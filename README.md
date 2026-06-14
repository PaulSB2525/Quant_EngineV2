# Quant Engine v2 — Multi-Asset Statistical Arbitrage System

> Wall Street-grade autonomous trading engine supporting simultaneous
> cross-asset strategies across crypto (Binance CEX) and US equities
> (NYSE/NASDAQ via Alpaca). Designed for institutional rigor on modest
> hardware: a Ryzen laptop hosts the engine, a Raspberry Pi serves the
> admin panel through a Cloudflare Zero Trust tunnel.

---

## Table of Contents

1. [What changed from v1](#what-changed-from-v1)
2. [Architecture](#architecture)
3. [Technology stack](#technology-stack)
4. [Mathematical core (bivariate & gap-aware)](#mathematical-core-bivariate--gap-aware)
5. [Risk Engine v2](#risk-engine-v2)
6. [LLM intelligence layer](#llm-intelligence-layer)
7. [Repository layout](#repository-layout)
8. [Quick start](#quick-start)
9. [Configuration](#configuration)
10. [Operating the system](#operating-the-system)
11. [Pair trading workflow](#pair-trading-workflow)
12. [Monitoring & observability](#monitoring--observability)
13. [Performance benchmarks](#performance-benchmarks)
14. [Scaling](#scaling)
15. [Security](#security)
16. [Troubleshooting](#troubleshooting)
17. [Roadmap and known TODOs](#roadmap-and-known-todos)
18. [Academic references](#academic-references)
19. [Regulatory rate disclaimer](#regulatory-rate-disclaimer)
20. [Trading disclaimer](#trading-disclaimer)

---

## What changed from v1

| Capability | v1 | v2 |
|---|---|---|
| Asset classes | Crypto only | **Crypto + US Equities** |
| Brokers | Binance (`ccxt.pro`) | **Binance + Alpaca** via abstract `BaseBrokerAdapter` |
| Symbol notation | `BTC/USDT` | **`CRYPTO:BTC/USDT` / `EQUITY:AAPL`** (v1 strings still work) |
| OU process | Univariate prices | **Univariate + bivariate cointegrated spreads** |
| Market hours | N/A (24/7) | **MarketSessionGuard** with NYSE/NASDAQ calendar 2024-2027 |
| Gap handling | None | **Kalman P/R inflation** on equity reopens |
| Risk namespaces | Single config | **`CryptoRiskConfig` (250% vol cap) + `EquityRiskConfig` (60% vol cap)** |
| TCA | bps flat | **bps for crypto, SEC §31 + FINRA TAF + half-spread + Almgren-Chriss for equity** |
| Correlation control | None | **Cross-asset Factor Exposure Penalty** + hard cutoff at \|ρ\| ≥ 0.90 |
| Concurrency | `asyncio` tasks | **`asyncio.TaskGroup` with `except*`** for structured concurrency |
| LLM validation | Single schema | **Differentiated `validate_crypto_thesis` / `validate_equity_thesis` / `validate_pair_thesis`** |

All v1 entry points remain backwards-compatible. The v1 bot can be
upgraded incrementally; nothing breaks.

---

## Architecture

### Hybrid two-node layout

```
┌─────────────────────────────────────┐    ┌──────────────────────────┐
│  Node A — The Engine                │    │  Node B — The Gatekeeper │
│  Ryzen 3 / 8 GB / SSD / Linux       │◀──▶│  Raspberry Pi 4 / 4 GB   │
│                                     │LAN │  - Streamlit dashboard   │
│  - bot_core (asyncio.TaskGroup)     │    │  - cloudflared tunnel    │
│  - QuestDB + Postgres + Redis       │    │  - Heartbeat monitor     │
│  - Engine matemático + LLMs         │    │                          │
└─────────────────────────────────────┘    └──────────────────────────┘
                                                       │
                                                       ▼
                                            Cloudflare Zero Trust
                                                       │
                                                       ▼
                                                 Your browser
```

### Asynchronous fan-out from core to brokers

```
                    ┌───────────────────────────────────┐
                    │       TradingBot core             │
                    │       (asyncio.TaskGroup)         │
                    └───────────────────────────────────┘
                       │                            │
                       │ asset_class=CRYPTO         │ asset_class=EQUITY
                       ▼                            ▼
        ┌─────────────────────────┐    ┌──────────────────────────────┐
        │  BinanceBrokerAdapter   │    │  MarketSessionGuard          │
        │  (ccxt.pro websockets)  │    │  - NYSE calendar 2024-2027   │
        │                         │    │  - DST via zoneinfo          │
        │  - watch_order_book     │    │  - Early closes (BlackFriday)│
        │  - submit_market_order  │    │                              │
        │  - OCO  (SL+TP)         │    │  Intercepta cada tick equity │
        └─────────────────────────┘    │  ┌────────────────────────┐  │
                       │               │  │ open ?                 │  │
                       ▼               │  │   yes → forward        │  │
                Binance CEX            │  │   no  → freeze buffer  │  │
                                       │  │                        │  │
                                       │  │ reopen transition ?    │  │
                                       │  │   → set gap_pending    │  │
                                       │  └────────────────────────┘  │
                                       └──────────────┬───────────────┘
                                                      │
                                                      ▼
                                         ┌─────────────────────────┐
                                         │  AlpacaBrokerAdapter    │
                                         │  (alpaca-py async)      │
                                         │                         │
                                         │  - subscribe_quotes(WS) │
                                         │  - bracket native       │
                                         │    (entry+SL+TP atomic) │
                                         └─────────────────────────┘
                                                      │
                                                      ▼
                                              NYSE / NASDAQ
```

The MarketSessionGuard sits **inline** in the equity ticker callback. Equity
ticks during closed hours never reach the math engine — buffers are frozen so
GARCH return matrices don't flood with zero-returns and OU/Kalman don't see
opening gap discontinuities as legitimate observations.

### Single-asset and pair execution flows

```
SINGLE-ASSET FLOW (per tick, decision_period_secs):
  Ticker → Kalman → OU/GARCH refit → z-score
       └→ RiskManager.evaluate()  ┐
                                  ├→ LLM veto (optional)
                                  ├→ Bracket order (entry + SL + TP atomic)
                                  └→ Persist + Telegram

PAIR FLOW (every pair_decision_period_secs, default 30s):
  StateA + StateB → cointegration refit (hourly)
                 → compute spread X_t = ln(A) − β·ln(B) − α
                 → OU(spread) → z-score
                 → RiskManager.evaluate_pair_trade()
                 → Validate both legs' markets open
                 → submit_market_order leg A + submit_market_order leg B
                 → Rollback leg A if leg B fails
                 → Persist with pair_partner cross-reference
```

---

## Technology stack

| Layer | Tool | Version | Why |
|---|---|---|---|
| Runtime | Python | 3.12+ | Native `asyncio.TaskGroup`, `except*` for structured exception propagation, faster startup |
| Crypto gateway | `ccxt.pro` | 4.3+ | Async WS over 100+ exchanges; uniform spot/futures API |
| Equity gateway | `alpaca-py` | 0.30+ | Official Alpaca SDK; native bracket orders; async data stream |
| Time zone | `zoneinfo` | stdlib | Deterministic DST handling for `America/New_York`; no third-party deps |
| Market calendar | embedded + optional `pandas_market_calendars` | — | NYSE holidays 2024-2027 embedded (≈2 KB); pandas calendar auto-detected if installed |
| TSDB | QuestDB | 7.4 | ILP TCP < 1 ms/tick, SQL-native, SSD-optimized |
| OLTP | PostgreSQL | 16 | Trades, equity snapshots, pair partner cross-references via JSONB |
| Cache / state | Redis | 7 | Day-lock SETNX, Kelly smoothing, rolling correlation ZSETs |
| Numerical | NumPy + SciPy | 1.26 / 1.13 | Vectorized OU, ADF, L-BFGS-B for GARCH MLE |
| LLM (fast) | Gemini 1.5 Flash | — | News sentiment, 1.5-2.5 s typical |
| LLM (deep) | DeepSeek-R1 | — | Trade thesis validation, chain-of-thought |
| Dashboard | Streamlit + Plotly | 1.36 / 5.22 | Equity curve, underwater, reasoning log; runs on Pi |
| Auth | streamlit-authenticator | 0.3.2 | bcrypt-hashed passwords, no OAuth infrastructure |
| Tunnel | cloudflared | latest | Zero Trust, email allowlist, no public ports |
| Containers | Docker Compose v2 | — | Per-service memory limits, healthchecks |

### Why `asyncio.TaskGroup`

Python 3.11+ introduced structured concurrency. In v2 the entire bot lives
inside a single `async with asyncio.TaskGroup() as tg:` block. Properties:

- If any task raises an unhandled exception, **all sibling tasks are
  cancelled** automatically.
- Exceptions are aggregated into an `ExceptionGroup` and handled with
  `except* SomeError:` syntax.
- Eliminates entire classes of "zombie task" bugs that plagued v1
  (orphaned WS loops after a crashed decision loop).

The bot's `start()` method maps cleanly:

```python
async with asyncio.TaskGroup() as tg:
    tg.create_task(self._wait_stop())
    for sym in self.cfg.symbols:
        tg.create_task(self._decision_loop(sym))
    if self.pairs:
        tg.create_task(self._pair_dispatcher())
    tg.create_task(self._equity_snapshot_loop())
    tg.create_task(self._daily_report_loop())
    tg.create_task(self._session_monitor_loop())
```

---

## Mathematical core (bivariate & gap-aware)

### Ornstein-Uhlenbeck (univariate, retained from v1)

```
dX_t = θ·(μ − X_t)·dt + σ·dW_t
```

Estimated via OLS on the **exact discretization** (not Euler-Maruyama):

```
X_{t+1} = a + b·X_t + ε,    b = exp(−θ·dt), a = μ(1−b),
                            Var(ε) = σ²(1−b²)/(2θ)
```

Validation (5,000 synthetic points, true θ=2.0, μ=100, σ=0.5):
```
θ̂ = 1.80,  μ̂ = 99.92,  σ̂ = 0.50,  half_life = 0.385
```

### Engle-Granger two-step cointegration (v2)

For a pair `(P_A, P_B)`, we estimate the cointegrating regression and the
**purified residual spread**:

```
Step 1 — OLS:    ln(P_A) = α + β·ln(P_B) + ε

Step 2 — Spread:    X_t = ln(P_A_t) − β·ln(P_B_t) − α

Step 3 — ADF on residuals:
                    Δr_t = γ·r_{t-1} + Σ φ_i·Δr_{t-i} + u_t
                    H₀: γ = 0  (unit root, NOT stationary)
                    H₁: γ < 0  (stationary, cointegrated)
```

Verdict enum:

| Verdict | Meaning |
|---|---|
| `COINTEGRATED` | ADF p < 0.05 — fully operable |
| `BORDERLINE` | 0.05 ≤ p < 0.10 — usable with caution |
| `NOT_COINTEGRATED` | p ≥ 0.10 — do not trade |
| `STRUCTURAL_BREAK` | β rolling deviates > 2σ from its rolling mean |

The OU/GARCH/Kalman v1 machinery is then applied **directly to the spread
series X_t** — the spread is univariate, so no new estimator infrastructure is
needed. This is the elegance of the Engle-Granger reduction.

### Gap-aware Kalman filter (v2)

Equity markets close ~16 hours per day plus full weekends. When a market
reopens, the first observation typically contains a **price jump** that
encodes 16+ hours of unobserved information. A naive Kalman update on this
observation would either over-trust the prior (rejecting the jump) or
under-trust the prior (chasing noise).

The gap-aware update inflates both the state variance `P` and the
observation variance `R`:

```
ΔP_gap = (σ_proc_per_√s)² · gap_seconds          (additive)

R_inflated = R_base · √(max(gap_seconds, 1s))    (multiplicative)
```

**Rationale:**

- `P` grows **linearly** in gap_seconds because the unobserved random-walk
  state diffuses with variance `Q·Δt`. This is mathematically exact.
- `R` grows with **√Δt** because the first post-open tick has more
  microstructure noise (low liquidity, gap-jump). The square-root scaling
  is empirically conservative — it inflates uncertainty without making the
  filter completely ignore the observation.
- After the gap-update, `R` is **restored to `R_base`** — only the first
  post-reopen tick gets the special treatment.

Validated behavior (180 → gap → 182.5 jump after 45.5 h close):

| | Tick 1 (post-gap) | Tick 21 |
|---|---|---|
| With gap update | x = 179.998, **P = 0.164** | x = **182.465** (converged) |
| Naive (no gap) | x = 180.120, P = 0.0019 | x = 181.608 (lagging) |

The gap-aware filter starts maximally uncertain, then converges rapidly
because the high P allows subsequent ticks to dominate.

### Dimensionally-corrected reversion alpha (Step 2 fix)

The v1 formula `alpha = 0.5 · |z| · √σ²_per_period` had a dimensional
inconsistency. The correct formulation scales σ² up to the **holding period
implied by the OU half-life**:

```
                                ┌──────────────────────────────┐
                                │            half_life         │
Alpha (bps) = 0.5 · |z| · √   │  σ²_GARCH  · ──────────────   │  · 10,000
                                │                  dt          │
                                └──────────────────────────────┘
```

Interpretation: a `|z| = 2.5` signal whose half-life is 30 minutes does not
deliver the same edge as a `|z| = 2.5` signal whose half-life is 5 minutes.
The longer half-life means more time for the realized return to materialize,
so the integrated variance over the holding period is larger and the alpha
in bps is larger. Conversely, the Kelly continuous formulation uses the
**holding-period variance** in its denominator, keeping dimensions consistent:

```
                  Alpha_decimal
Kelly_cont = ───────────────────────
              σ²_GARCH · (half_life / dt)
```

This correction was critical to make `evaluate_pair_trade()` produce
reasonable sizing for realistic spread half-lives.

### GARCH(1,1) (retained from v1)

```
r_t = μ + ε_t,    ε_t = σ_t · z_t,   z_t ~ N(0,1)
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
```

Constraints: `ω > 0, α,β ≥ 0, α+β < 1`. MLE via L-BFGS-B with soft
penalty on violations. h-step forecast in closed form
(Hamilton 1994, eq. 21.1.41):

```
E[σ²_{t+h} | F_t] = V_∞ + (α+β)^(h-1) · (σ²_{t+1} − V_∞)
```

---

## Risk Engine v2

Five gates evaluated in order. The trade is rejected at the first failure.

```
0. Market open?               ────────────► REJECT_MARKET_CLOSED
1. Day not locked?            ────────────► REJECT_DRAWDOWN
2. Vol regime within cap?     ────────────► REJECT_VOLATILITY_REGIME
3. Half-life coherent?        ────────────► REJECT_HALF_LIFE_INCOHERENT
4. TCA passes?                ────────────► REJECT_TCA
5. Correlation cluster OK?    ────────────► REJECT_CORRELATION_CLUSTER
6. Kelly > 0 after smoothing? ────────────► REJECT_KELLY_NEGATIVE
→ APPROVED with SL/TP from ATR multipliers
```

### Asset-class-differentiated configuration

```python
@dataclass
class RiskConfig:
    # Truly shared parameters at the top level
    kelly_fraction: float = 0.25
    kelly_smoothing_alpha: float = 0.3
    daily_drawdown_limit_pct: float = 0.02
    correlation_warning_threshold: float = 0.70
    correlation_hard_cutoff: float = 0.90
    correlation_lookback_seconds: int = 3600

    # Class-specific namespaces
    crypto: CryptoRiskConfig
    equity: EquityRiskConfig
```

| Parameter | Crypto | Equity |
|---|---|---|
| `max_acceptable_vol_annualized` | **2.50** (250%) | **0.60** (60%) |
| `max_position_pct_of_equity` | 0.30 | 0.20 |
| `per_trade_stop_loss_atr_mult` | 2.5 | 3.0 (equity gaps require wider SL) |
| `per_trade_take_profit_atr_mult` | 4.0 | 5.0 |
| `min_acceptable_half_life_sec` | 30 | 300 (5 min) |
| `max_acceptable_half_life_sec` | 21,600 (6 h) | 432,000 (5 days) |

**Why a 60% cap on equity?** Annualized volatility above this in a single
name signals one of three things: earnings manipulation, toxic corporate
event (FDA, fraud, lawsuit), or trading halt aftermath. None of these are
amenable to mean-reversion strategies; the right action is "step aside".
Crypto regimes routinely produce 100-200% annualized vol *legitimately*
(BTC during halving cycles, alts in bull-runs), hence the 250% cap.

### Crypto TCA

```
round_trip_bps = 2 · (maker_fee + spread + slippage)
threshold_bps  = round_trip_bps · safety_margin   (default 1.5×)
```

Default Binance spot: maker_fee=1.0 bps, expected_spread=2.0 bps,
slippage=3.0 bps → threshold = 18 bps round-trip.

### Equity TCA — full regulatory micro-structure

For US equities, "zero commission" does **not** mean zero cost. Three
regulatory fees apply (sale-only) plus market microstructure:

| Cost component | Rate | When | Source |
|---|---|---|---|
| SEC §31 fee | **$20.60 per $1,000,000 notional** | Sale only | SEC FY2026 Rate Advisory (effective 2026-04-04) |
| FINRA TAF | **$0.000195 per share, capped $9.79/trade** | Sale only | FINRA Schedule A §1 (effective 2026-01-01) |
| Half-spread | bid-ask / 2 | Both legs | Market microstructure |
| Slippage (Almgren-Chriss) | k · (volume / ADV) · σ | Both legs | Linear impact model |

Implementation in `EquityTradingCosts.round_trip_bps()`:

```python
# SEC §31 in basis points
sec_fee_usd = (notional_usd / 1_000_000) * sec_fee_per_million
sec_fee_bps = (sec_fee_usd / notional_usd) * 10_000

# FINRA TAF in basis points
shares = notional_usd / avg_share_price
taf_usd = min(shares * finra_taf_per_share, finra_taf_cap)
taf_bps = (taf_usd / notional_usd) * 10_000

# Half-spread (both legs, so doubled)
spread_cost_bps = 2 * expected_half_spread_bps

# Linear impact via Almgren-Chriss (simplified)
σ_per_sec_bps = (vol_annualized / √(252·6.5·3600)) * 10_000
slippage_bps_one_way = k · volume_pct_of_ADV · σ_per_sec_bps · 100

# Round trip
round_trip_bps = spread_cost_bps + sec_fee_bps + taf_bps + 2·slippage_bps_one_way
```

For a $10,000 AAPL position at $180 with 25% annualized vol: round-trip
**≈ 10.24 bps**, threshold (×1.5) **≈ 15.4 bps**.

### Factor Exposure Penalty (cross-asset)

A pair like `EQUITY:NVDA` and `CRYPTO:BTC/USDT` can exhibit high rolling
correlation during macro risk-off episodes, despite being "different
asset classes". v2 tracks **rolling 1-hour Pearson correlation** between
every actively-held symbol pair via Redis sorted sets.

Penalty curve (non-linear by design):

```
|ρ_max|         ≤  warning_threshold (0.70)  →  multiplier = 1.0  (no penalty)
warning < |ρ|  <  hard_cutoff       (0.90)  →  linear interp, multiplier ∈ (0, 1)
|ρ_max|         ≥  hard_cutoff       (0.90)  →  REJECT_CORRELATION_CLUSTER
```

The multiplier is applied to the smoothed Kelly fraction:

```
kelly_final = min(
    kelly_smoothed · kelly_fraction · correlation_multiplier,
    kelly_max_leverage,
    class_cfg.max_position_pct_of_equity
)
```

**Negative correlation is not penalized.** A strong negative correlation
between two open positions is a natural hedge (desirable), so only |ρ| is
considered. This is the standard treatment in factor risk models.

Validated behavior (synthetic NVDA + BTC at ρ ≈ 0.85, then test trade for
NVDA): measured ρ = 0.903 → `REJECT_CORRELATION_CLUSTER` triggered
correctly.

### Daily Drawdown Lock

At the first trade of each UTC day, the current equity is snapshotted into
Redis via `SETNX risk:daily_equity_start:YYYY-MM-DD`. Every evaluation
checks:

```
dd_pct = (start_equity − current_equity) / start_equity
if dd_pct ≥ 2%:
    redis.SET risk:day_locked:YYYY-MM-DD  (TTL 30h)
    REJECT all subsequent trades until 00:00 UTC
```

The lock survives bot restarts. A crashed-and-restarted bot **cannot**
keep losing past the daily cap because the Redis state is authoritative.

---

## LLM intelligence layer

The LLM is **veto-only**. It can reject a trade but never approve one that
the RiskManager rejected. Default behavior on timeout, network error, or
malformed JSON: **reject**.

| Method | Backend | Bucket | Purpose |
|---|---|---|---|
| `analyze_news_sentiment(text)` | Gemini 1.5 Flash | 15 rpm | Fast sentiment, ~2 s |
| `validate_crypto_thesis(thesis)` | DeepSeek-R1 | 30 rpm (light) | Technical thesis review |
| `validate_equity_thesis(thesis, context)` | DeepSeek-R1 | 15 rpm (heavy) | Adds fundamental context |
| `validate_pair_thesis(thesis, ctx_a, ctx_b)` | DeepSeek-R1 | 15 rpm (heavy) | Cointegration + leg fundamentals |

### Equity fundamental context

Optional structured payload passed to the LLM for equity validation:

```python
EquityFundamentalContext(
    fed_funds_rate_pct=5.25,
    is_fomc_week=True,
    hours_to_next_earnings=18.0,
    sector="Technology",
    sector_beta_rolling_60d=1.4,
    recent_8k_count_30d=2,
    short_interest_pct_of_float=8.5,
    avg_daily_volume_shares=50_000_000,
    is_pre_market=False,
)
```

**All fields are optional.** If the bot doesn't have a field (e.g. no
earnings calendar feed wired), it passes `None` and the LLM evaluates
only the technical signals — with an explicit prompt acknowledging the
limitation. This degrades gracefully instead of crashing.

The prompt builder emits explicit guidance: "If earnings <24h away →
typically reject; if FOMC this week and trade is rate-sensitive → caution;
high short interest >20% → check for squeeze setup conflict", etc.

---

## Repository layout

```
quant_system/
├── README.md                        ◀ this file
├── instructions.md                  Operational manual (Linux setup, Pi setup, tunnel)
│
├── engine_math.py        (495 LOC)  OU + Kalman + GARCH + Engle-Granger + gap handler
├── risk_manager.py       (562 LOC)  Asset-class split + TCA + correlation + pair eval
├── llm_bridge.py         (475 LOC)  Gemini + DeepSeek with differentiated schemas
├── broker_adapters.py    (480 LOC)  BaseBrokerAdapter ABC + Binance + Alpaca
├── market_session.py     (210 LOC)  NYSE calendar 2024-2027, DST, early closes
├── bot_core.py           (660 LOC)  Engine with TaskGroup, pair dispatcher
├── dashboard.py          (297 LOC)  Streamlit admin panel
├── test_integration.py   (110 LOC)  End-to-end tests with mock broker
│
├── docker-compose.yml               Full stack for Node A
├── Dockerfile.bot                   Engine image
├── Dockerfile.dashboard             Dashboard image (Pi-friendly)
│
├── requirements.txt                 Bot deps
├── requirements-dashboard.txt       Dashboard deps
│
└── .env.example                     Variable template
```

---

## Quick start

### Prerequisites

- Linux Node A (Ubuntu 22.04+ or Debian 12+), 8 GB RAM minimum
- Raspberry Pi OS 64-bit on Node B (optional)
- Docker + Docker Compose v2
- Python 3.12 (host parity with Dockerfile)
- Accounts: Binance subaccount, Alpaca paper account, Telegram bot,
  optionally Gemini + DeepSeek API keys
- Cloudflare-managed domain (optional, for remote dashboard access)

### Express install (Node A)

```bash
git clone <your-repo>.git quant
cd quant

cp .env.example .env
nano .env       # fill in credentials

docker compose up -d
docker compose logs -f bot
```

After ~25 minutes the GARCH buffer fills and the bot evaluates signals.
**Keep `PAPER_TRADING=true` for the first 14 days minimum.**

### Validate the math layer

```bash
python3 engine_math.py
```

Expected output (synthetic data smoke tests):

```
OU:     θ̂=1.80 (true 2.0), μ̂=99.92, σ̂=0.50
Kalman: RMSE 0.51 → 0.22 (-57.8% reduction)
GARCH:  α̂=0.080 (true 0.10), β̂=0.898 (true 0.85)

Cointegration tests:
  Pair cointegrated:     β̂=1.296 (true 1.30), ADF p=0.0010, verdict=cointegrated
  Pair not cointegrated: ADF p=0.239,           verdict=not_cointegrated
  Structural break:      β rolling std=8.86,    verdict=structural_break
```

### Run the test suite

The synchronization fixes (outbox, pair exit, startup reconciler) are covered
by an async test suite that exercises the real `bot_core` code against in-memory
doubles (fake broker + fake Postgres pool) — no exchange or DB needed:

```bash
pip install pytest pytest-asyncio   # included in requirements.txt
pytest tests/ -v
```

Expected: `8 passed` (scenarios A–D for the critical fixes, plus four
`test_startup_reconciler_*` for the restart gaps). Takes ~15s because two
scenarios exercise the real retry/backoff timers.

---

## Trade status lifecycle

Every order is persisted with the **outbox pattern**: a row is written
**before** the broker call and only confirmed afterwards, so a crash never
leaves a position the system can't see. The `status` column (TEXT) takes:

| status | Meaning | Who sets it | Resolution |
|---|---|---|---|
| `pending` | Row written before sending to broker | outbox INSERT | → `open` on fill, or resolved by startup reconciler |
| `open` | Fill confirmed, position live | `_confirm_entry_open` | Closed by reconciler (singles) / `_pair_exit_loop` (pairs) |
| `closed` | Round-trip done (see `exit_reason`) | finalize / reconciler / pair exit | Terminal |
| `canceled` | Order never executed (pair leg A failed, sibling aborted) | pair entry | Terminal |
| `failed_unprotected` | Entry OK but SL/TP placement failed | `_handle_unprotected_single` | Emergency close in progress |
| `orphaned` | Live position that could **not** be closed | emergency close exhausted | **Manual intervention** |
| `failed_no_position` | Pending with no live position and no close fill | startup reconciler | Terminal |
| `pair_leg_a_orphaned` | Pair leg B failed and rollback of A impossible | `_rollback_pair_leg_a` | **Manual intervention** |
| `pair_leg_b_close_failed` | Leg A closed but leg B couldn't | pair exit / startup | **Manual intervention** |

`exit_reason` values: `tp`, `sl`, `timeout`, `panic_close`, `emergency_close`,
`rollback`, `startup_closed`, `startup_partial_close`, `startup_external_close`.

Rows needing manual intervention (`orphaned`, `failed_unprotected`,
`pair_leg_*`) are re-alerted to Telegram every `UNRESOLVED_ALERT_SECS`
(default 600s) until resolved.

### Known open gaps (next sprint)

These are **not** addressed by the synchronization fixes and remain open:

- **A-2 — PnL ignores fees.** `pnl_quote` is `direction·(exit−entry)·size`;
  broker fees/commissions are not subtracted. Reported PnL is optimistic.
- **M-2 — Money uses `float`, not `Decimal`.** DB columns are `NUMERIC(24,12)`
  but Python computes in float; rounding error can accumulate.
- **A-4 — Irregular sampling treated as 1s.** OU/GARCH and annualization assume
  one sample/second, but ticks arrive at variable rate → half-life, vol caps
  and alpha are mis-scaled.
- **A-5 — GARCH variance frozen between refits.** `last_sigma2`/`last_eps` only
  update on the hourly refit, so the forecast can use up-to-1h-stale variance.

---

## Configuration

### Critical environment variables (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `PAPER_TRADING` | `true` | Keep true for 14+ days minimum |
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | — | IP-whitelisted, **no withdraw** |
| `ALPACA_API_KEY` / `ALPACA_API_SECRET` | — | Paper or live |
| `SYMBOLS` | `CRYPTO:BTC/USDT,CRYPTO:ETH/USDT` | Prefix-tagged |
| `PAIR_SYMBOLS` | (empty) | Pipe-separated, comma-list (see below) |
| `ALLOW_PRE_MARKET` | `false` | Equity 04:00-09:30 ET |
| `ALLOW_POST_MARKET` | `false` | Equity 16:00-20:00 ET |
| `ENABLE_LLM_VALIDATION` | `true` | Disable to operate fully without LLMs |
| `GEMINI_API_KEY` / `DEEPSEEK_API_KEY` | — | Optional |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | — | Alerts and daily reports |
| `ADMIN_PASSWORD_HASH` | — | bcrypt; generate with `streamlit-authenticator.Hasher.hash()` |
| `AUTH_COOKIE_KEY` | — | ≥32 chars random |

### Per-class risk tuning (Python config)

```python
# More conservative (start here)
RiskConfig(
    kelly_fraction=0.10,
    daily_drawdown_limit_pct=0.01,        # 1% daily cap
    correlation_hard_cutoff=0.80,         # tighter
    crypto=CryptoRiskConfig(
        max_acceptable_vol_annualized=2.0,
        max_position_pct_of_equity=0.20,
    ),
    equity=EquityRiskConfig(
        max_acceptable_vol_annualized=0.50,
        max_position_pct_of_equity=0.15,
        block_if_earnings_within_hours=48,
    ),
)
```

---

## Operating the system

### Daily checklist

```
[ ] docker compose ps             → all UP, no restart loops
[ ] Telegram                       → 🟢 Bot started message received
[ ] Dashboard                      → equity_now correct, day not locked
[ ] Underwater chart               → intraday DD < 1%
[ ] Reasoning log                  → recent decisions make sense
[ ] Equity market hours respected  → no trades during NYSE close
```

### Going live (post-paper-trading)

Do not flip `PAPER_TRADING=false` until **all** of these are true:

| Metric | Minimum |
|---|---|
| Days in paper | ≥ 14 (ideally 30) |
| Annualized Sharpe (paper) | > 1.0 |
| Max DD (paper) | < 4% |
| Closed trades | ≥ 30 |
| Profit factor | > 1.3 |
| Cointegration tests passing for configured pairs | 100% |

Then:

```bash
# 1. Edit .env:  PAPER_TRADING=false
# 2. Start with 10% of target capital
docker compose down bot && docker compose up -d bot
```

### Backups (cron on Node A)

```cron
0 4 * * * docker exec quant_postgres pg_dump -U quant quant | gzip > /backup/pg-$(date +\%F).sql.gz
0 4 * * * docker exec quant_redis redis-cli SAVE && cp /var/lib/docker/volumes/quant_redis_data/_data/dump.rdb /backup/
```

---

## Pair trading workflow

### Step 1 — Define a pair

In `.env`, list pairs as `SYMBOL_A|SYMBOL_B` separated by commas:

```bash
SYMBOLS=CRYPTO:BTC/USDT,CRYPTO:ETH/USDT,EQUITY:SMH,EQUITY:NVDA
PAIR_SYMBOLS=CRYPTO:ETH/USDT|CRYPTO:BTC/USDT,EQUITY:NVDA|EQUITY:SMH
```

Convention: **A is the asset where alpha is expected, B is the more
liquid hedger**. For `NVDA|SMH`, NVDA is A (single stock with idiosyncratic
alpha), SMH is B (semiconductor ETF as the hedge).

For cross-asset pairs (highly experimental), the system still works but the
LLM and the `MarketSessionGuard` will heavily restrict execution windows.
Example:

```bash
PAIR_SYMBOLS=EQUITY:COIN|CRYPTO:BTC/USDT
```

This pair can only trade during NYSE hours (when both COIN and BTC are
streaming). The bot enforces this automatically.

### Step 2 — Pre-flight cointegration check

On startup, the `_pair_dispatcher` task validates each configured pair:

```
[PAIR ETH/USDT/BTC/USDT] refit: β=0.9743 verdict=cointegrated
[PAIR NVDA/SMH] refit: β=1.4521 verdict=cointegrated
[PAIR COIN/BTC/USDT] refit: β=0.8132 verdict=borderline
```

Pairs marked `not_cointegrated` or `structural_break` are skipped but
logged. They are re-evaluated hourly — if the relationship recovers, they
become operable automatically.

### Step 3 — Live execution

When `|z_spread| ≥ 2.0`:
1. `RiskManager.evaluate_pair_trade()` validates the full pair stack
   (drawdown, vol, half-life, TCA on both legs, dimensional alpha).
2. Optional LLM validation via `validate_pair_thesis`.
3. **Leg A** submitted first (market order).
4. **Leg B** submitted second. If it fails, leg A is rolled back
   immediately to avoid a directional residual position.
5. Both legs persist to Postgres with `pair_partner` cross-references.

Leg sizing maintains hedge neutrality:
```
leg_a_size_quote = kelly_final · current_equity
leg_b_size_quote = leg_a_size_quote · |β|
```

### Step 4 — Position monitoring (automated)

The `pair.open_position` dict stores the entry spread, SL/TP **in spread
units**, and metadata. A dedicated `_pair_exit_loop` (every
`PAIR_EXIT_CHECK_SECS`, default 5s) closes pair positions automatically on:

- **Take-profit** — `|z| < PAIR_TP_Z_THRESHOLD` (default 0.3), the spread
  reverted to the mean.
- **Stop-loss** — the spread moved beyond `sl_spread_distance` (in spread
  log-units) relative to `entry_spread`.
- **Timeout** — held longer than `PAIR_MAX_HOLD_HOURS` (default 48h).

Both legs are closed with verified retries; if leg A fails to close, leg B is
left untouched and the close is retried next cycle; if leg B fails after A
closed, the row is flagged `pair_leg_b_close_failed`, the pair is blocked and a
Telegram alert is raised for manual intervention.

**Restart safety:** on startup, `_startup_reconciler` rehydrates `open` pairs
from the DB (`risk_metrics._recon`) so the exit loop resumes managing them.
**Residual limit:** right after a restart, exit-by-z only resumes once price
buffers refill (Kalman needs fresh ticks); until then the position is tracked
but not actively exited.

---

## Monitoring & observability

### Dashboard tabs (Streamlit on Node B)

| Tab | Content |
|---|---|
| 📈 Equity & Drawdown | Equity curve + underwater chart |
| 📜 Reasoning Log | Per-trade RiskManager reason + Kelly metrics |
| 💼 Trades | Last N trades with asset_class filter |
| 🔗 Pair Trades | Spread visualization, β over time, leg P&L |
| 🩺 System Health | Redis keys, day-lock status, broker connectivity |

### Header KPIs

| KPI | Computation |
|---|---|
| Equity | Latest snapshot from `equity_snapshots` |
| Sharpe (ann.) | `(μ_returns / σ_returns) · √(periods/year)` |
| Max DD | `max((peak − eq) / peak)` |
| Win Rate | `wins / closed_trades` |
| Profit Factor | `Σwins / |Σlosses|` |
| Pair Win Rate | Same, filtered to `pair_partner IS NOT NULL` |

### Telegram alert types

- 🟢 / 🔴 — Bot start/stop
- 📈 — Single-asset entry (symbol, side, size, SL, TP, reason)
- 📊 — Pair entry (both legs, β, z-score, reason)
- 🔔 — Equity market reopen (gap handler armed)
- 🚨 — Execution error (WS down, SL/TP failed, etc.)
- 📊 — Daily 24h report (PnL breakdown by asset class)

### QuestDB SQL examples

```sql
-- Mid vs Kalman over the last hour for BTC
SELECT timestamp, mid, kalman_mid
FROM orderbook
WHERE symbol = 'CRYPTO__BTC_USDT'
  AND timestamp > dateadd('h', -1, now())
SAMPLE BY 1m;

-- Average spread per asset class today
SELECT asset_class, avg(spread)
FROM orderbook
WHERE timestamp > dateadd('d', -1, now())
GROUP BY asset_class;
```

---

## Performance benchmarks

Measured on Ryzen 3 5300U, 8 GB RAM, NVMe SSD:

| Operation | Latency |
|---|---|
| Kalman update (single tick) | ~0.8 µs |
| Kalman gap-update | ~1.2 µs |
| OU refit (240 points) | ~0.15 ms |
| GARCH refit (1,500 returns) | 80-200 ms |
| GARCH forecast 1-step | ~5 µs |
| Cointegration refit (500 points) | 3-5 ms |
| ADF test (single) | 0.5-1 ms |
| Rolling β over 500-point window | ~2 ms |
| `RiskManager.evaluate()` (no LLM) | 2-5 ms |
| `RiskManager.evaluate_pair_trade()` | 4-7 ms |
| `RiskManager.evaluate()` (with DeepSeek) | 2-5 s |
| Gemini sentiment | 1.5-2.5 s |
| Tick → decision (full pipeline, no LLM) | < 10 ms |
| Bracket order submit (Binance OCO) | 50-150 ms |
| Bracket order submit (Alpaca native) | 80-200 ms |

### Memory footprint

| Component | RAM in use |
|---|---|
| Bot (asyncio + numpy buffers) | 200-350 MB |
| QuestDB (with ~1M ticks) | 400-800 MB |
| PostgreSQL | 80-150 MB |
| Redis | 20-50 MB |
| Dashboard | 150-250 MB |
| **Node A total** | **~1.2-1.6 GB** |
| **Node B (Pi) total** | **~250-400 MB** |

---

## Scaling

| Capital | Recommended setup |
|---|---|
| ≤ $1k USD | 1-2 symbols, `kelly_fraction=0.10`, paper or microsize |
| $1k-$50k | 2-4 single symbols, 1 pair, strict vol filter |
| $50k-$500k | Move engine to VPS near exchange (Binance ↔ AWS Tokyo). Same `docker-compose`. Add liquidity-aware sizing via real ADV feeds. |
| $500k-$5M | Multi-strategy, derivatives (perp futures, equity options). Reduce `decision_period_secs` to 0.5. Implement Almgren-Chriss slicing for entries > 5% of ADV. |
| > $5M | Multi-broker, multi-region. Each strategy = separate compose stack. FIX protocol direct to ECNs. Co-location. |

What **never changes** when scaling:
- The math layer (OU, Kalman, GARCH, cointegration)
- The six risk gates
- The asset-class differentiation in TCA and vol caps
- The LLM-as-veto-only contract

What changes is parameter tuning and execution venue.

---

## Security

### Network

- UFW on Node A: SSH and database ports only from LAN
- No public ports on Node A
- Cloudflare Zero Trust in front of dashboard (email allowlist)

### Credentials

- Binance API key: IP-whitelisted, **no withdraw permission, no margin
  enable**
- Alpaca API key: same — read + trade only
- `.env` file: `chmod 600`, never committed
- Dashboard password: bcrypt-hashed via streamlit-authenticator
- Cookie key: ≥32 chars random

### Container hardening

- Non-root `quant` user inside containers
- Explicit resource limits in compose
- Log rotation (`max-size: 10m, max-file: 5`)
- No host-mounted directories except read-only configs

### Application invariants

- **Idempotency**: `client_order_id = SHA256(symbol|side|ns_timestamp)`,
  Postgres `ON CONFLICT DO NOTHING`. Duplicate entries are impossible
  across crashes.
- **Day-lock persistence**: Redis SETNX with TTL means restarting the bot
  cannot reset the daily drawdown counter.
- **Exchange-side SL/TP**: protective orders live on the exchange. If
  the bot dies, positions remain protected.
- **LLM cannot approve**: silence, errors, or malformed JSON all result
  in rejection. Only an explicit `{"verdict": "accept"}` lets a trade
  through.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `WS error … 1006` | Internet flap | Adapter reconnects with exponential backoff |
| `GARCH fit failed` | Outlier returns | Winsorize input; check data feed integrity |
| `Day locked: drawdown 2.X%` | Daily DD cap hit | Wait until 00:00 UTC |
| `Bot at 100% CPU` | OU refit loop on corrupted buffer | `docker compose restart bot` |
| `REJECT_HALF_LIFE_INCOHERENT` constant | Mismatch between data frequency and `dt` in OU fit | Verify `garch_periods_per_year` matches sampling |
| `REJECT_MARKET_CLOSED` during equity trading hours | DST transition issue | Confirm host timezone; `MarketSessionGuard` uses ET internally |
| `REJECT_PAIR_NOT_OPERABLE` | Cointegration broken | Check verdict — `structural_break` means pause; `not_cointegrated` means find new pair |
| Telegram silent | Bot never received `/start` | Send `/start` to your bot once |
| Alpaca `alpaca-py missing` | SDK not installed | `pip install alpaca-py>=0.30` |
| Dashboard `connection refused` from Pi | UFW blocking Pi IP | `sudo ufw allow from PI_IP to any port 5432` |

---

## Roadmap and known TODOs

### Critical TODOs inherited from v2 — must be closed before serious capital

These are honest gaps in the current implementation. None of them are
mathematical or risk-management omissions; they are **execution-side
automation** items that v2 still does by manual oversight.

| TODO | Severity | Description |
|---|---|---|
| **Pair SL/TP monitor loop** | **High** | Single-asset positions have exchange-side OCO protecting them automatically. Pair positions have SL/TP defined **in spread units**, which neither broker can monitor natively. A `_pair_position_monitor()` task must poll the live spread and trigger leg-A + leg-B closes when threshold hit. Until v3 ships this, pair trades require human oversight. |
| **Earnings calendar feed** | **Medium** | `EquityFundamentalContext.hours_to_next_earnings` is currently always `None` unless injected manually. v3 will add a background task polling Alpaca's `/v2/calendar` or IEX Cloud earnings endpoint. Without this, the LLM equity validation gets weaker signals. |
| **Live regulatory rate sync** | **Medium** | SEC §31 and FINRA TAF rates are hardcoded. They change annually (sometimes mid-year). v3 will fetch from `/data/sec_fee_rate.json` weekly and fail loud if rates have changed. |
| **Position reconciliation loop** | **High** | When a SL or TP fills on the exchange, the bot currently does not know — the `trades.status` stays at `open` until manually closed. v3 will add an `_order_status_monitor()` task that polls fills and updates Postgres. |
| **Pair entry rollback hardening** | **Medium** | If leg B submission times out (vs. errors cleanly), the rollback path is not exercised. v3 will use `asyncio.shield()` and a timeout-with-cancel pattern around the leg B submit. |

### v3 confirmed scope

- [ ] Pair SL/TP monitor (above)
- [ ] Order reconciliation loop (above)
- [ ] Earnings calendar feed via Alpaca + IEX Cloud
- [ ] Live SEC/FINRA rate sync
- [ ] Multi-strategy isolation: one strategy per Docker stack, shared risk overlay
- [ ] Kalman smoother RTS for post-hoc analysis in dashboard

### v3 research items (no commitment)

- [ ] Regime-switching GARCH (HMM-driven α/β switching)
- [ ] Johansen test for >2 cointegrated assets (n-dimensional baskets)
- [ ] Execution-aware Almgren-Chriss slicing for entries > 5% ADV
- [ ] Direct WebSocket (no ccxt wrapper) for sub-ms tick latency on top venues

---

## Academic references

**Stochastic processes**
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton UP.
  Chapter 13 (state-space) and 21 (GARCH).
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional
  Heteroskedasticity." *J. Econometrics* 31.

**Cointegration and pairs trading**
- Engle, R. F., & Granger, C. W. J. (1987). "Co-integration and error
  correction: representation, estimation, and testing." *Econometrica*
  55(2).
- MacKinnon, J. G. (1996). "Numerical Distribution Functions for Unit
  Root and Cointegration Tests." *J. Applied Econometrics*.
- Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). "Pairs
  Trading: Performance of a Relative-Value Arbitrage Rule." *Review of
  Financial Studies* 19(3).
- Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and
  Analysis*. Wiley.

**Statistical arbitrage**
- Avellaneda, M., & Lee, J. H. (2010). "Statistical Arbitrage in the
  U.S. Equities Market." *Quantitative Finance* 10(7).
- Pole, A. (2007). *Statistical Arbitrage: Algorithmic Trading
  Insights*. Wiley.

**Kelly criterion**
- Kelly, J. L. (1956). "A New Interpretation of Information Rate."
  *Bell System TJ* 35.
- Thorp, E. O. (2006). "The Kelly Criterion in Blackjack, Sports
  Betting, and the Stock Market."

**Execution**
- Almgren, R., & Chriss, N. (2001). "Optimal Execution of Portfolio
  Transactions." *J. Risk* 3.
- Cartea, Á., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and
  High-Frequency Trading*. Cambridge UP.

---

## Regulatory rate disclaimer

The SEC §31 fee rate ($20.60 per $1M effective 2026-04-04) and the FINRA
Trading Activity Fee ($0.000195/share, $9.79 cap, effective 2026-01-01)
hardcoded in `EquityTradingCosts` are valid **at the time of writing**.
These rates change annually and sometimes mid-year. Authoritative sources:

- SEC Fee Rate Advisories:
  https://www.sec.gov/rules-regulations/fee-rate-advisories
- FINRA TAF schedule:
  https://www.finra.org/rules-guidance/guidance/trading-activity-fee
- FINRA Information Notices:
  https://www.finra.org/rules-guidance/notices

Verify both rates before going live and update `EquityTradingCosts`
accordingly. A 20-30% rate change is unusual but historically possible
(SEC §31 dropped to $0.00 between May 2025 and April 2026). Treat the
hardcoded values as a starting point, not as ground truth.

---

## Trading disclaimer

Algorithmic trading carries **real risk of total capital loss**. This
code is provided as-is, with no warranties. Before deploying real
capital:

1. Read and understand every line of `engine_math.py` and
   `risk_manager.py`.
2. Audit the bot's decisions for ≥ 2 weeks in paper trading.
3. Start with < 5% of your trading capital.
4. Define a-priori the maximum amount you can afford to lose.
5. Consult an accredited financial advisor in your jurisdiction.

The author is not a financial advisor. Nothing in this code constitutes
investment advice. Past performance does not guarantee future returns.
Statistical models can break in out-of-sample market conditions.

---

## License

Choose a license before publishing the repository. Suggestions:

- **MIT** — maximum adoption
- **AGPL-3.0** — open derivatives only
- **Proprietary** — for commercialization

Default in the template: none. Add before pushing public.
