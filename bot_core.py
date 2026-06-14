"""
bot_core.py  (v2)
=================
Engine multi-asset multi-broker.

Componentes integrados:
    - broker_adapters.{Binance,Alpaca} para data + ejecución.
    - market_session.MarketSessionGuard para horarios equity.
    - engine_math v2 con cointegración y gap handling de Kalman.
    - risk_manager v2 con namespaces crypto/equity.
    - llm_bridge v2 con validación diferenciada.

Loops:
    1. Per-asset: ingesta de ticker, actualización de buffers, decisión single-asset.
    2. Per-pair: cada N segundos, evalúa cointegración + spread y posibles entries.
    3. Session monitor: detecta transiciones open↔closed para gap handling.
    4. Equity reconciliation: snapshots de equity periódicos.
    5. Daily report: PnL/win-rate vía Telegram al final del día UTC.

Diseño:
    - asyncio.TaskGroup (Python 3.11+) para que crashes de hijas cancelen al grupo.
    - Buffers congelados durante closes (no append flat-line).
    - Gap handler invocado automáticamente en la primera observación post-reopen.
    - Reconexión y backoff manejados por los adapters; bot core solo recibe callbacks.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import logging
import math
import os
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from zoneinfo import ZoneInfo

import asyncpg
import httpx
import numpy as np
import redis.asyncio as aioredis

from broker_adapters import (
    BaseBrokerAdapter,
    BinanceBrokerAdapter,
    AlpacaBrokerAdapter,
    TickerData,
)
from engine_math import (
    CointegrationParams,
    CointegrationVerdict,
    GARCHParams,
    OUParams,
    compute_pair_spread,
    fit_cointegration_ols,
    fit_garch,
    fit_ornstein_uhlenbeck,
    garch_forecast_variance,
    init_kalman,
    ou_zscore,
    pair_spread_zscore,
    _garch_recursion,
)
from llm_bridge import (
    EquityFundamentalContext,
    LLMManager,
    ValidationVerdict,
)
from market_session import MarketSessionGuard
from risk_manager import (
    AssetClass,
    CryptoTradingCosts,
    EquityTradingCosts,
    RiskConfig,
    RiskManager,
    RiskVerdict,
    parse_asset_class,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bot_core")

UTC = ZoneInfo("UTC")


# =============================================================================
# Config
# =============================================================================

@dataclass
class BotConfig:
    # Crypto broker
    binance_api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    binance_api_secret: str = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    # Equity broker
    alpaca_api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    alpaca_api_secret: str = field(default_factory=lambda: os.getenv("ALPACA_API_SECRET", ""))
    paper_trading: bool = field(default_factory=lambda: os.getenv("PAPER_TRADING", "true").lower() == "true")

    # Símbolos con prefijo (CRYPTO:BTC/USDT, EQUITY:AAPL)
    symbols: tuple = field(default_factory=lambda: tuple(
        os.getenv("SYMBOLS", "CRYPTO:BTC/USDT,CRYPTO:ETH/USDT").split(",")
    ))

    # Pares cointegrados: "A|B,C|D" - se valida cointegración al arranque
    pair_symbols: tuple = field(default_factory=lambda: tuple(
        p for p in os.getenv("PAIR_SYMBOLS", "").split(",") if p
    ))

    # Sesión
    allow_pre_market: bool = field(default_factory=lambda: os.getenv("ALLOW_PRE_MARKET", "false").lower() == "true")
    allow_post_market: bool = field(default_factory=lambda: os.getenv("ALLOW_POST_MARKET", "false").lower() == "true")

    # Ventanas
    ou_window: int = 240
    garch_window: int = 1500
    garch_refit_secs: int = 3600
    decision_period_secs: float = 2.0
    pair_decision_period_secs: float = 30.0
    reconciliation_period_secs: float = 12.0   # cierre SL/TP: poll cada 10-15s

    # Outbox / resolución de estados problemáticos (C-3/C-4/C-5)
    unresolved_alert_secs: int = field(
        default_factory=lambda: int(os.getenv("UNRESOLVED_ALERT_SECS", "600")))
    # Cierre de pares (C-1/C-2)
    pair_exit_check_secs: float = field(
        default_factory=lambda: float(os.getenv("PAIR_EXIT_CHECK_SECS", "5.0")))
    pair_tp_z_threshold: float = field(
        default_factory=lambda: float(os.getenv("PAIR_TP_Z_THRESHOLD", "0.3")))
    pair_max_hold_hours: float = field(
        default_factory=lambda: float(os.getenv("PAIR_MAX_HOLD_HOURS", "48.0")))
    # Edad máxima del precio en caché para considerar fresca una señal de salida
    pair_price_max_age_secs: float = field(
        default_factory=lambda: float(os.getenv("PAIR_PRICE_MAX_AGE_SECS", "30.0")))

    # Quote currencies por asset class
    crypto_quote: str = "USDT"
    equity_quote: str = "USD"

    # Infra
    questdb_host: str = field(default_factory=lambda: os.getenv("QUESTDB_HOST", "questdb"))
    questdb_ilp_port: int = field(default_factory=lambda: int(os.getenv("QUESTDB_ILP_PORT", "9009")))
    postgres_dsn: str = field(default_factory=lambda: os.getenv(
        "POSTGRES_DSN", "postgresql://quant:quant@postgres:5432/quant"))
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://redis:6379/0"))

    # Telegram
    telegram_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))

    # LLMs
    enable_llm_validation: bool = field(default_factory=lambda: os.getenv("ENABLE_LLM_VALIDATION", "true").lower() == "true")
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))


# =============================================================================
# Persistencia
# =============================================================================

class QuestDBWriter:
    """Cliente ILP TCP. Reconnect on demand."""
    def __init__(self, host: str, port: int):
        self.host, self.port = host, port
        self._writer: Optional[asyncio.StreamWriter] = None
        self._lock = asyncio.Lock()

    async def _connect(self):
        _, self._writer = await asyncio.open_connection(self.host, self.port)

    async def write(self, table: str, tags: dict, fields: dict,
                     ts_ns: Optional[int] = None):
        async with self._lock:
            if self._writer is None or self._writer.is_closing():
                try:
                    await self._connect()
                except OSError as e:
                    logger.warning("QuestDB no disponible: %s", e)
                    return
            line = self._format_line(table, tags, fields, ts_ns)
            try:
                self._writer.write(line.encode("utf-8"))
                await self._writer.drain()
            except (ConnectionError, BrokenPipeError) as e:
                logger.warning("Conexión QuestDB rota: %s", e)
                self._writer = None

    @staticmethod
    def _format_line(table, tags, fields, ts_ns):
        tag_str = ",".join(f"{k}={v}" for k, v in tags.items())
        parts = []
        for k, v in fields.items():
            if isinstance(v, str):
                parts.append(f'{k}="{v}"')
            elif isinstance(v, bool):
                parts.append(f'{k}={"t" if v else "f"}')
            else:
                parts.append(f"{k}={v}")
        field_str = ",".join(parts)
        ts = ts_ns if ts_ns is not None else time.time_ns()
        prefix = f"{table},{tag_str}" if tag_str else table
        return f"{prefix} {field_str} {ts}\n"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL PRIMARY KEY,
    client_order_id TEXT UNIQUE NOT NULL,
    symbol TEXT NOT NULL,
    asset_class TEXT NOT NULL DEFAULT 'crypto',
    side TEXT NOT NULL,
    pair_partner TEXT,
    entry_price NUMERIC(24, 12),
    exit_price NUMERIC(24, 12),
    size_quote NUMERIC(24, 12),
    size_base NUMERIC(24, 12),
    pnl_quote NUMERIC(24, 12),
    fees_quote NUMERIC(24, 12),
    opened_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'open',
    exit_reason TEXT,
    notes TEXT,
    reasoning JSONB,
    risk_metrics JSONB
);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_opened ON trades(symbol, opened_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_asset_class ON trades(asset_class);
CREATE INDEX IF NOT EXISTS idx_trades_status_partner ON trades(status, pair_partner);

-- Migración 002 (idempotente) para DBs creadas antes del patrón outbox:
ALTER TABLE trades ADD COLUMN IF NOT EXISTS exit_reason TEXT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS notes TEXT;

CREATE TABLE IF NOT EXISTS equity_snapshots (
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    equity_quote NUMERIC(24, 12) NOT NULL,
    available_quote NUMERIC(24, 12) NOT NULL,
    open_positions_value NUMERIC(24, 12) NOT NULL,
    unrealized_pnl NUMERIC(24, 12) NOT NULL,
    quote_currency TEXT NOT NULL DEFAULT 'USDT'
);
CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_snapshots(ts DESC);
"""


# =============================================================================
# Telegram
# =============================================================================

class TelegramNotifier:
    def __init__(self, token, chat_id):
        self.token, self.chat_id = token, chat_id
        self._client = httpx.AsyncClient(timeout=10.0)
        self._enabled = bool(token and chat_id)

    # Caracteres reservados de MarkdownV2 (Telegram Bot API). Cualquier
    # variable dinámica interpolada en un mensaje MarkdownV2 debe escaparlos
    # o Telegram responde HTTP 400 "can't parse entities".
    _MDV2_SPECIALS = r"_*[]()~`>#+-=|{}.!"

    @classmethod
    def escape_mdv2(cls, text) -> str:
        """Escapa todos los reservados de MarkdownV2 en una variable dinámica."""
        s = str(text)
        for ch in cls._MDV2_SPECIALS:
            s = s.replace(ch, "\\" + ch)
        return s

    @staticmethod
    def code_block(text) -> str:
        """
        Envuelve texto técnico (errores de broker, IDs de orden, dumps de ccxt
        con llaves/guiones/comillas, dict de fills) en un bloque ```text ... ```
        para que Telegram lo trate como texto plano y NO parsee entidades
        Markdown/MarkdownV2 — la causa típica del HTTP 400 'can't parse
        entities'. Neutraliza los backticks que cerrarían el bloque antes.
        """
        safe = str(text).replace("```", "ʼʼʼ").replace("`", "ʼ")
        return f"```text\n{safe}\n```"

    async def send(self, text, parse_mode="Markdown"):
        if not self._enabled:
            return
        # Blindaje: la mensajería NUNCA debe poder detener el loop de trading.
        # Capturamos CUALQUIER excepción (red, timeout, JSON, etc.) y además
        # inspeccionamos el status: un 400/403/5xx no lanza en httpx, así que
        # hay que mirarlo explícitamente.
        # parse_mode=None debe viajar como AUSENCIA del campo, no como
        # "parse_mode": null en el JSON: Telegram rechaza un parse_mode no
        # reconocido con 400 'unsupported parse_mode'. Omitir la clave fuerza
        # texto plano puro (mismo criterio que el reintento de abajo).
        payload = {"chat_id": self.chat_id, "text": text}
        if parse_mode is not None:
            payload["parse_mode"] = parse_mode
        try:
            resp = await self._client.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json=payload,
            )
            if resp.status_code >= 400:
                logger.error(
                    "Fallo no bloqueante en notificación de Telegram: HTTP %s %s",
                    resp.status_code, resp.text[:300])
                # Si fue un 400 de parseo (formato Markdown roto), reintentamos
                # una vez en texto plano para no perder el alert.
                if parse_mode is not None and resp.status_code == 400:
                    try:
                        await self._client.post(
                            f"https://api.telegram.org/bot{self.token}/sendMessage",
                            json={"chat_id": self.chat_id, "text": text},
                        )
                    except Exception as e:
                        logger.error(
                            "Fallo no bloqueante en notificación de Telegram: %s", e)
        except Exception as e:
            logger.error("Fallo no bloqueante en notificación de Telegram: %s", e)
            return

    async def close(self):
        await self._client.aclose()


# =============================================================================
# Asset state
# =============================================================================

class AssetState:
    """
    Estado unificado por activo. Crypto y equity comparten todos los campos;
    los equity-only (last_tick_ts, gap_pending) son ignorados silenciosamente
    para crypto.
    """
    def __init__(self, asset_string: str, ou_window: int, garch_window: int):
        self.asset_string = asset_string
        self.asset_class, _ = parse_asset_class(asset_string)

        # Buffers
        self.mid_prices: deque[float] = deque(maxlen=ou_window)
        self.kalman_prices: deque[float] = deque(maxlen=ou_window)
        self.log_returns: deque[float] = deque(maxlen=garch_window)
        self.atr_window: deque[float] = deque(maxlen=14)

        # Modelos
        self.kalman = None
        self.ou_params: Optional[OUParams] = None
        self.garch_params: Optional[GARCHParams] = None
        self.last_garch_fit_ts: float = 0.0
        self.last_sigma2: float = 0.0
        self.last_eps: float = 0.0

        # Equity-only: tracking de gap
        self.last_tick_ts: Optional[dt.datetime] = None
        self.gap_pending: bool = False    # True si próximo tick es post-cierre

        # Posición abierta
        self.open_position: Optional[dict] = None

    @property
    def ready(self) -> bool:
        return (
            self.kalman is not None
            and self.ou_params is not None
            and self.ou_params.is_operable
            and self.garch_params is not None
            and len(self.atr_window) >= 5
        )

    def atr_estimate(self) -> float:
        if not self.atr_window:
            return 0.0
        return float(np.mean(self.atr_window))


# =============================================================================
# Pair state
# =============================================================================

class PairState:
    """Estado de un par cointegrado."""
    def __init__(self, sym_a: str, sym_b: str):
        self.sym_a = sym_a
        self.sym_b = sym_b
        self.coint_params: Optional[CointegrationParams] = None
        self.spread_buffer: deque[float] = deque(maxlen=500)
        self.spread_ou: Optional[OUParams] = None
        self.spread_garch: Optional[GARCHParams] = None
        self.last_refit_ts: float = 0.0
        self.open_position: Optional[dict] = None


# =============================================================================
# TradingBot
# =============================================================================

class TradingBot:
    def __init__(self, config: BotConfig):
        self.cfg = config
        self.brokers: dict[AssetClass, BaseBrokerAdapter] = {}
        self.redis: Optional[aioredis.Redis] = None
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.qdb: Optional[QuestDBWriter] = None
        self.risk: Optional[RiskManager] = None
        self.llm: Optional[LLMManager] = None
        self.tg: Optional[TelegramNotifier] = None
        self.session_guard = MarketSessionGuard(
            allow_pre_market=config.allow_pre_market,
            allow_post_market=config.allow_post_market,
        )
        self.states: dict[str, AssetState] = {}
        self.pairs: dict[tuple[str, str], PairState] = {}
        self._stop = asyncio.Event()
        # Referencias fuertes a tareas fire-and-forget (retries de confirmación
        # de DB) para que el GC no las destruya antes de completarse.
        self._bg_tasks: set[asyncio.Task] = set()

    # ---- Lifecycle ----

    async def start(self):
        logger.info("Iniciando TradingBot multi-asset (paper=%s)",
                     self.cfg.paper_trading)

        # Persistencia
        self.redis = aioredis.from_url(self.cfg.redis_url, decode_responses=True)
        await self.redis.ping()
        self.qdb = QuestDBWriter(self.cfg.questdb_host, self.cfg.questdb_ilp_port)
        self.pg_pool = await asyncpg.create_pool(self.cfg.postgres_dsn,
                                                   min_size=1, max_size=4)
        async with self.pg_pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)

        # Risk / LLM / Telegram
        # paper_trading selecciona el perfil de costos TCA crypto: en paper se
        # relaja a ~3 bps (baja fricción de testnet); en live usa los costos
        # reales de Binance spot (~18 bps). Ver CryptoTradingCosts.for_paper_trading.
        self.risk = RiskManager(self.redis, RiskConfig(),
                                paper_trading=self.cfg.paper_trading)
        logger.info("RiskManager TCA crypto profile: %s (umbral maker=%.1f bps)",
                     "paper_sim" if self.cfg.paper_trading else "production",
                     self.risk.crypto_costs.round_trip_bps(False)
                     * self.risk.cfg.crypto.tca_safety_margin)
        if self.cfg.enable_llm_validation:
            self.llm = LLMManager(self.cfg.gemini_api_key,
                                    self.cfg.deepseek_api_key)
        self.tg = TelegramNotifier(self.cfg.telegram_token,
                                     self.cfg.telegram_chat_id)

        # Brokers (solo los necesarios según los símbolos)
        await self._init_brokers()

        # Estados
        for sym in self.cfg.symbols:
            self.states[sym] = AssetState(sym, self.cfg.ou_window,
                                            self.cfg.garch_window)

        # Suscribir tickers
        for sym in self.cfg.symbols:
            ac, _ = parse_asset_class(sym)
            broker = self.brokers[ac]
            await broker.subscribe_ticker(sym, self._on_ticker)
            logger.info("Subscribed: %s via %s", sym, broker.__class__.__name__)

        # Pares
        for pair_str in self.cfg.pair_symbols:
            if "|" not in pair_str:
                logger.warning("Pair mal formado: %s. Usar 'A|B'", pair_str)
                continue
            a, b = pair_str.split("|", 1)
            self.pairs[(a, b)] = PairState(a, b)

        await self.tg.send(
            f"🟢 *Bot iniciado*\nAssets: {len(self.cfg.symbols)}\n"
            f"Pairs: {len(self.pairs)}\nPaper: {self.cfg.paper_trading}"
        )

        # Loops principales en TaskGroup
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._wait_stop(), name="wait_stop")
                for sym in self.cfg.symbols:
                    tg.create_task(self._decision_loop(sym),
                                     name=f"dec:{sym}")
                if self.pairs:
                    tg.create_task(self._pair_dispatcher(), name="pairs")
                tg.create_task(self._equity_snapshot_loop(), name="equity_snap")
                tg.create_task(self._daily_report_loop(), name="daily_report")
                tg.create_task(self._session_monitor_loop(), name="session_mon")
                tg.create_task(self._order_reconciliation_loop(),
                                 name="order_reconcile")
                tg.create_task(self._unresolved_alert_loop(),
                                 name="unresolved_alert")
                if self.pairs:
                    tg.create_task(self._pair_exit_loop(), name="pair_exit")
        except* Exception as eg:
            # ExceptionGroup en Python 3.11+
            for e in eg.exceptions:
                logger.exception("Task crashed: %r", e)
            raise

    async def _wait_stop(self):
        await self._stop.wait()
        # Sale del TaskGroup limpiamente
        raise asyncio.CancelledError("stop signal received")

    async def stop(self):
        logger.info("Deteniendo bot...")
        self._stop.set()
        for broker in self.brokers.values():
            try:
                await broker.close()
            except Exception:
                pass
        if self.llm:
            await self.llm.close()
        if self.tg:
            await self.tg.send("🔴 *Bot detenido.*")
            await self.tg.close()
        if self.pg_pool:
            await self.pg_pool.close()
        if self.redis:
            await self.redis.close()

    async def _init_brokers(self):
        needs_crypto = False
        needs_equity = False
        for sym in self.cfg.symbols:
            ac, _ = parse_asset_class(sym)
            if ac == AssetClass.CRYPTO:
                needs_crypto = True
            elif ac == AssetClass.EQUITY:
                needs_equity = True

        if needs_crypto:
            b = BinanceBrokerAdapter(
                api_key=self.cfg.binance_api_key,
                api_secret=self.cfg.binance_api_secret,
                paper_trading=self.cfg.paper_trading,
                quote_currency=self.cfg.crypto_quote,
            )
            await b.connect()
            self.brokers[AssetClass.CRYPTO] = b

        if needs_equity:
            a = AlpacaBrokerAdapter(
                api_key=self.cfg.alpaca_api_key,
                api_secret=self.cfg.alpaca_api_secret,
                paper_trading=self.cfg.paper_trading,
                quote_currency=self.cfg.equity_quote,
                redis_client=self.redis,
            )
            await a.connect()
            self.brokers[AssetClass.EQUITY] = a

    # ---- Ticker callback ----

    async def _on_ticker(self, ticker: TickerData):
        """Callback invocado por los adapters en cada tick."""
        state = self.states.get(ticker.asset_string)
        if state is None:
            return

        # ---- Si es equity, verificar mercado abierto ----
        if state.asset_class == AssetClass.EQUITY:
            ts_dt = dt.datetime.fromtimestamp(ticker.timestamp_ns / 1e9, tz=UTC)
            if not self.session_guard.is_open(ticker.asset_string, ts_dt):
                # Mercado cerrado. Si llega tick (pre/post extended sin permiso),
                # ignorar y NO contaminar buffers.
                return

        # ---- Gap handling ----
        if state.gap_pending and state.kalman is not None:
            gap_secs = self.session_guard.gap_duration_seconds(
                ticker.asset_string,
                state.last_tick_ts or dt.datetime.now(UTC),
                dt.datetime.fromtimestamp(ticker.timestamp_ns / 1e9, tz=UTC),
            )
            if gap_secs > 60:    # gaps <60s no merecen tratamiento especial
                gap_vol = self._estimate_gap_volatility(state)
                state.kalman.update_after_gap(ticker.mid, gap_secs, gap_vol)
                logger.info("[%s] Gap handler: %.0fs gap, vol/√s=%.5f. Kalman P inflada.",
                              ticker.asset_string, gap_secs, gap_vol)
                state.gap_pending = False
                state.last_tick_ts = dt.datetime.fromtimestamp(
                    ticker.timestamp_ns / 1e9, tz=UTC)
                # NO ejecutamos update normal porque update_after_gap ya lo hizo.
                # Solo actualizamos buffers.
                self._update_buffers_after_kalman(state, ticker)
                return
            else:
                state.gap_pending = False

        # ---- Update normal ----
        self._update_state(state, ticker)

        # ---- Persistir tick (best-effort) ----
        await self.qdb.write(
            table="orderbook",
            tags={
                "symbol": ticker.asset_string.replace("/", "_").replace(":", "__"),
                "asset_class": state.asset_class.value,
            },
            fields={
                "bid": ticker.bid, "ask": ticker.ask, "mid": ticker.mid,
                "spread": ticker.spread,
                "kalman_mid": state.kalman.x if state.kalman else ticker.mid,
            },
            ts_ns=ticker.timestamp_ns,
        )

        # ---- Alimentar correlation tracker ----
        if state.log_returns:
            await self.risk.record_price_observation(
                ticker.asset_string, state.log_returns[-1])

    @staticmethod
    def _estimate_gap_volatility(state: AssetState) -> float:
        """
        Estima vol del proceso para inflación de Kalman.P durante un gap.
        Usa GARCH unconditional std si disponible; fallback a std muestral
        de los log_returns recientes.
        """
        if state.garch_params is not None:
            uv = state.garch_params.unconditional_variance
            if math.isfinite(uv) and uv > 0:
                # GARCH var es per-period (typically 1 segundo). El gap está
                # en segundos, así que vol per sqrt(sec) es sqrt(uv).
                return math.sqrt(uv)
        if state.log_returns:
            arr = np.array(state.log_returns, dtype=np.float64)
            return float(np.std(arr, ddof=1))
        return 0.001   # fallback conservador

    def _update_state(self, state: AssetState, ticker: TickerData):
        mid = ticker.mid
        spread = ticker.spread

        # Init Kalman lazy
        if state.kalman is None:
            obs_var = (spread / 2) ** 2 + 1e-12
            state.kalman = init_kalman(mid, obs_variance=obs_var,
                                         process_variance=1e-8)
        state.kalman.update(mid)

        self._update_buffers_after_kalman(state, ticker)

    def _update_buffers_after_kalman(self, state: AssetState,
                                       ticker: TickerData):
        state.mid_prices.append(ticker.mid)
        state.kalman_prices.append(state.kalman.x)
        state.last_tick_ts = dt.datetime.fromtimestamp(
            ticker.timestamp_ns / 1e9, tz=UTC)

        if len(state.kalman_prices) >= 2:
            prev = state.kalman_prices[-2]
            curr = state.kalman_prices[-1]
            r = math.log(curr / prev) if prev > 0 else 0.0
            state.log_returns.append(r)
            state.atr_window.append(abs(curr - prev))

        # OU refit cuando el buffer está lleno
        if len(state.kalman_prices) == state.kalman_prices.maxlen:
            try:
                state.ou_params = fit_ornstein_uhlenbeck(
                    np.array(state.kalman_prices, dtype=np.float64),
                    dt=1.0 / (252 * 24 * 60 * 60),
                )
            except ValueError:
                pass

        # GARCH refit periódico
        now = time.time()
        if (len(state.log_returns) >= 500
                and now - state.last_garch_fit_ts > self.cfg.garch_refit_secs):
            try:
                arr = np.array(state.log_returns, dtype=np.float64)
                state.garch_params = fit_garch(arr)
                state.last_eps = float(arr[-1] - state.garch_params.mu)
                sig2_path = _garch_recursion(
                    arr, state.garch_params.mu, state.garch_params.omega,
                    state.garch_params.alpha, state.garch_params.beta,
                )
                state.last_sigma2 = float(sig2_path[-1])
                state.last_garch_fit_ts = now
                logger.info("GARCH refit %s: α=%.4f β=%.4f persist=%.4f",
                              state.asset_string,
                              state.garch_params.alpha,
                              state.garch_params.beta,
                              state.garch_params.persistence)
            except Exception as e:
                # state.garch_params se reasigna SOLO dentro del try, así que
                # ante un fallo de convergencia los últimos parámetros válidos
                # quedan intactos en memoria. Demoted a DEBUG para no inundar
                # la consola con cada refit en ventanas con poca variabilidad
                # (típico de Testnet de Binance en ticks rápidos).
                logger.debug("GARCH fit %s falló (asimila últimos params "
                              "válidos): %s", state.asset_string, e)

    # ---- Decision loop single-asset ----

    async def _decision_loop(self, asset_string: str):
        state = self.states[asset_string]
        while not self._stop.is_set():
            try:
                await asyncio.sleep(self.cfg.decision_period_secs)

                # ¿Mercado abierto?
                if not self.session_guard.is_open(asset_string):
                    continue

                if not state.ready or state.open_position is not None:
                    continue

                await self._maybe_enter_single(state)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error decision_loop %s", asset_string)

    async def _maybe_enter_single(self, state: AssetState):
        ou = state.ou_params
        garch = state.garch_params
        current_price = state.kalman.x
        z = ou_zscore(current_price, ou)

        if abs(z) < 2.0:
            return

        side = "long" if z < 0 else "short"

        # Binance Spot Testnet (paper) no permite cortos: no hay margen, así que
        # un SHORT cripto falla con InsufficientFunds. Lo bloqueamos aquí, antes
        # de gastar ciclos en la evaluación de riesgo y el envío al adaptador.
        if (self.cfg.paper_trading
                and state.asset_class == AssetClass.CRYPTO
                and side == "short"):
            logger.info("[CRYPTO] Señal SHORT descartada preventivamente en modo "
                        "Paper Trading para evitar InsufficientFunds.")
            return

        sigma2_forecast = garch_forecast_variance(
            garch, state.last_eps, state.last_sigma2, horizon=1)[0]

        gap = (ou.mu - current_price) if side == "long" \
              else (current_price - ou.mu)
        expected_alpha_bps = (0.5 * gap / current_price) * 10000.0

        from scipy.stats import norm
        win_prob = float(norm.cdf(abs(z)) * 0.8 + 0.1)

        equity = await self._fetch_equity(state.asset_class)
        atr = state.atr_estimate() or current_price * 0.001

        other_open = [s for s, st in self.states.items()
                       if st.open_position is not None and s != state.asset_string]

        decision = await self.risk.evaluate(
            symbol=state.asset_string,
            current_equity=equity,
            current_price=current_price,
            expected_alpha_bps=expected_alpha_bps,
            win_prob=win_prob,
            win_loss_ratio=1.6,
            garch_variance_per_period=sigma2_forecast,
            garch_periods_per_year=252 * 24 * 60 * 60
                if state.asset_class == AssetClass.CRYPTO else 252 * 6.5 * 3600,
            atr=atr,
            side=side,
            ou_half_life_sec=ou.half_life * (252 * 24 * 60 * 60),
            market_is_open=True,    # ya verificado arriba
            other_open_symbols=other_open,
        )

        if not decision.approved:
            logger.info("[%s] Rejected: %s", state.asset_string, decision.reason)
            return

        # LLM opcional para alpha modesto
        if self.llm and expected_alpha_bps < 30.0:
            thesis = {
                "symbol": state.asset_string, "side": side,
                "ou_zscore": z,
                "ou_half_life_sec": decision.metrics.get("ou_half_life_sec"),
                "garch_vol_annualized": decision.metrics.get("vol_annualized"),
                "kelly_final_fraction": decision.metrics.get("kelly_final"),
                "expected_alpha_bps": expected_alpha_bps,
                "tca_threshold_bps": decision.metrics.get("threshold_bps"),
            }
            try:
                if state.asset_class == AssetClass.CRYPTO:
                    validation = await asyncio.wait_for(
                        self.llm.validate_crypto_thesis(thesis), timeout=15.0)
                else:
                    validation = await asyncio.wait_for(
                        self.llm.validate_equity_thesis(thesis, None),
                        timeout=20.0,
                    )
                if validation.verdict != ValidationVerdict.ACCEPT:
                    logger.info("[%s] LLM vetó: %s",
                                  state.asset_string, validation.reasoning)
                    return
            except asyncio.TimeoutError:
                logger.warning("LLM timeout, continuando sin validación")

        await self._execute_single_entry(state, side, decision, current_price)

    async def _execute_single_entry(self, state: AssetState, side: str,
                                       decision, current_price: float):
        broker = self.brokers[state.asset_class]
        client_id = broker.make_client_order_id(state.asset_string, side)
        # Coerción explícita a float Python puro (no np.float64): Alpaca's
        # Pydantic models tipan qty como Real y aceptan numpy, pero la
        # serialización JSON de algunos numpy scalars produce 'NaN' literal
        # cuando la conversión falla silenciosamente. Guard finitude antes
        # de mandar al broker — los adaptadores tienen su propio guard pero
        # cortar acá evita gastar un client_order_id en una orden inválida.
        try:
            size_base = float(decision.size_quote) / float(current_price)
        except (TypeError, ValueError, ZeroDivisionError) as e:
            logger.error("[%s] size_base no computable (%s); orden descartada.",
                         state.asset_string, e)
            return
        if not math.isfinite(size_base) or size_base <= 0:
            logger.error(
                "[%s] size_base no finito o ≤0 (size_quote=%s price=%s); "
                "orden descartada.",
                state.asset_string, decision.size_quote, current_price)
            return
        broker_side = "buy" if side == "long" else "sell"

        # ---- OUTBOX 1A: registrar 'pending' ANTES de enviar al broker ----
        # Si esta escritura falla, NO se envía nada al broker: preferimos no
        # operar a operar sin registro (C-3/C-4). entry_price se guarda como
        # precio estimado y se sobreescribe con el fill real al confirmar 1B.
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO trades
                    (client_order_id, symbol, asset_class, side, entry_price,
                     size_quote, size_base, status, reasoning, risk_metrics)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,'pending',$8,$9)
                    ON CONFLICT (client_order_id) DO NOTHING
                """, client_id, state.asset_string, state.asset_class.value,
                    side, current_price, decision.size_quote, size_base,
                    json.dumps({"reason": decision.reason}),
                    json.dumps(decision.metrics))
        except Exception as e:
            logger.exception(
                "[%s] No se pudo registrar trade 'pending'; NO se envía la "
                "orden al broker (sin registro = sin operación).",
                state.asset_string)
            await self.tg.send(
                f"🚨 DB no disponible: orden {state.asset_string} {side} "
                f"ABORTADA antes de enviar.\n{e}", parse_mode=None)
            return

        result = await broker.submit_bracket_orders(
            asset_string=state.asset_string,
            side=broker_side,
            size_base=size_base,
            stop_loss=decision.stop_loss_price,
            take_profit=decision.take_profit_price,
            client_order_id=client_id,
            # Precio de referencia para que Alpaca clampe el TP a la
            # microestructura (base_price + buffer). Binance lo ignora.
            base_price=current_price,
        )

        # ---- 1C: rutas de fallo explícitas ----
        if not result.success:
            # Alerta como TEXTO PLANO (el error de ccxt trae llaves/guiones).
            await self.tg.send(
                f"🚨 Ejecución falló {state.asset_string}:\n{result.error}",
                parse_mode=None)
            entry_px = result.avg_price or current_price
            if result.panic_closed:
                # Round-trip real (entrada + cierre de pánico). Confirmamos la
                # fila 'pending' -> 'closed' con el PnL real.
                exit_px = result.exit_price
                pnl_quote = None
                if exit_px is not None:
                    direction = 1.0 if side == "long" else -1.0
                    pnl_quote = direction * (exit_px - entry_px) * size_base
                await self._finalize_trade_closed(
                    client_id, entry_px, exit_px, pnl_quote,
                    exit_reason="panic_close",
                    notes=(result.error or "")[:500])
            else:
                # Posición potencialmente ABIERTA sin protección (C-3).
                await self._handle_unprotected_single(
                    state, side, broker, client_id, size_base,
                    entry_px=entry_px, err=result.error)
            return

        # ---- 1B: éxito → confirmar 'pending' -> 'open' ----
        entry_px = result.avg_price or current_price
        position = {
            "client_id": client_id,
            "broker_id": result.broker_order_id,
            "side": side,
            "entry": entry_px,
            "size_base": size_base,
            "size_quote": decision.size_quote,
            "sl": decision.stop_loss_price,
            "tp": decision.take_profit_price,
            "opened_at": time.time(),
        }
        # Marcamos open_position SIEMPRE tras un fill exitoso (la posición existe
        # y está protegida en el exchange) para impedir doble entrada en el
        # símbolo, incluso si el UPDATE de confirmación falla.
        state.open_position = position
        confirmed = await self._confirm_entry_open(client_id, entry_px)
        if confirmed:
            # El símbolo va en un inline code span (`...`): contenido literal en
            # Markdown legacy y MarkdownV2; inmune a '_'/'*'/'.' sin escape.
            await self.tg.send(
                f"📈 *ENTRADA* `{state.asset_string}` `{side.upper()}`\n"
                f"Size: `${decision.size_quote:,.2f}` ({size_base:.6f})\n"
                f"Entry: `{entry_px:.4f}` SL: `{decision.stop_loss_price:.4f}` "
                f"TP: `{decision.take_profit_price:.4f}`\n"
                + self.tg.code_block(decision.reason[:200])
            )
        else:
            # Broker OK pero Postgres no confirmó: la posición está viva y
            # protegida; solo la DB quedó en 'pending'. Reintentamos en
            # background y mantenemos el símbolo bloqueado hasta confirmar (C-4).
            position["_db_unconfirmed"] = True
            logger.critical(
                "[%s] Orden %s ejecutada y protegida, pero UPDATE 'open' falló; "
                "reintentando en background. Símbolo bloqueado hasta confirmar.",
                state.asset_string, client_id)
            await self.tg.send(
                f"🚨 CRÍTICO {state.asset_string}: orden ejecutada pero la DB no "
                f"confirmó 'open' (coid={client_id}). Reintentando en background; "
                f"símbolo bloqueado.", parse_mode=None)
            self._spawn_bg(
                self._retry_confirm_open(client_id, entry_px, state))

    # ---- Helpers outbox / confirmación / cierre (C-3, C-4) ----

    def _spawn_bg(self, coro) -> None:
        """Lanza una corrutina fire-and-forget conservando referencia fuerte."""
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _confirm_entry_open(self, client_id: str,
                                   entry_px: Optional[float]) -> bool:
        """UPDATE 'pending' -> 'open' con el precio de fill real. True si OK."""
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE trades
                       SET status='open',
                           entry_price=COALESCE($2, entry_price)
                     WHERE client_order_id=$1 AND status='pending'
                """, client_id, entry_px)
            return True
        except Exception:
            logger.exception("confirm 'open' falló coid=%s", client_id)
            return False

    async def _retry_confirm_open(self, client_id: str,
                                   entry_px: Optional[float],
                                   state: "AssetState") -> None:
        """Reintenta el UPDATE 'open' cada 30s hasta lograrlo (o stop)."""
        while not self._stop.is_set():
            await asyncio.sleep(30)
            if await self._confirm_entry_open(client_id, entry_px):
                logger.info("[%s] confirm 'open' reintentado OK coid=%s",
                            state.asset_string, client_id)
                pos = state.open_position
                if pos is not None and pos.get("client_id") == client_id:
                    pos.pop("_db_unconfirmed", None)
                await self.tg.send(
                    f"✅ DB reconciliada: {state.asset_string} coid={client_id} "
                    f"ahora 'open'.", parse_mode=None)
                return

    async def _finalize_trade_closed(self, client_id: str,
                                      entry_px: Optional[float],
                                      exit_px: Optional[float],
                                      pnl_quote: Optional[float], *,
                                      exit_reason: str, notes: str = "") -> bool:
        """UPDATE 'pending'/'open' -> 'closed'. Un reintento ante fallo."""
        for attempt in range(2):
            try:
                async with self.pg_pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE trades
                           SET status='closed',
                               entry_price=COALESCE($2, entry_price),
                               exit_price=$3, pnl_quote=$4, closed_at=NOW(),
                               exit_reason=$5, notes=$6
                         WHERE client_order_id=$1
                           AND status IN ('pending','open','failed_unprotected')
                    """, client_id, entry_px, exit_px, pnl_quote,
                        exit_reason, notes)
                return True
            except Exception:
                logger.exception("finalize 'closed' falló coid=%s (intento %d)",
                                 client_id, attempt + 1)
                await asyncio.sleep(2)
        return False

    async def _mark_status(self, client_id: str, status: str, *,
                            notes: Optional[str] = None) -> bool:
        """UPDATE de status simple (failed_unprotected/orphaned/...)."""
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE trades
                       SET status=$2, notes=COALESCE($3, notes)
                     WHERE client_order_id=$1
                """, client_id, status, notes)
            return True
        except Exception:
            logger.exception("mark_status %s falló coid=%s", status, client_id)
            return False

    async def _handle_unprotected_single(self, state: "AssetState", side: str,
                                          broker, client_id: str,
                                          size_base: float,
                                          entry_px: Optional[float],
                                          err: Optional[str]) -> None:
        """
        C-3: entry OK pero SL/TP falló y el pánico del adapter también. La
        posición puede estar ABIERTA sin protección. Marcamos el estado,
        alertamos en CRÍTICO e intentamos un cierre de emergencia (3 intentos,
        backoff 2s). Si todo falla → 'orphaned' + símbolo bloqueado.
        """
        await self._mark_status(
            client_id, "failed_unprotected",
            notes=("SL/TP placement failed: " + (err or ""))[:500])
        await self.tg.send(
            f"🚨🚨 CRÍTICO {state.asset_string} {side}: posición ABIERTA sin "
            f"protección (coid={client_id}). Cierre de emergencia en curso.",
            parse_mode=None)

        opposite = "sell" if side == "long" else "buy"
        backoff = 2.0
        for attempt in range(3):
            res = await broker.submit_market_order(
                state.asset_string, opposite, size_base,
                broker.make_client_order_id(state.asset_string, opposite))
            if res.success:
                exit_px = res.avg_price
                pnl_quote = None
                if exit_px is not None and entry_px is not None:
                    direction = 1.0 if side == "long" else -1.0
                    pnl_quote = direction * (exit_px - entry_px) * size_base
                await self._finalize_trade_closed(
                    client_id, entry_px, exit_px, pnl_quote,
                    exit_reason="emergency_close",
                    notes="closed after SL/TP placement failure")
                await self.tg.send(
                    f"✅ Cierre de emergencia OK {state.asset_string} "
                    f"coid={client_id} exit={exit_px}", parse_mode=None)
                return
            logger.error("[%s] cierre de emergencia intento %d/3 falló: %s",
                         state.asset_string, attempt + 1, res.error)
            await asyncio.sleep(backoff)
            backoff *= 2

        # Todos los intentos fallaron: posición huérfana, bloquear símbolo.
        await self._mark_status(
            client_id, "orphaned",
            notes="emergency close failed; manual intervention required")
        state.open_position = {"client_id": client_id, "status": "orphaned",
                               "side": side, "size_base": size_base,
                               "opened_at": time.time()}
        await self.tg.send(
            f"🆘 ORPHANED {state.asset_string} {side} size={size_base} "
            f"coid={client_id}: posición abierta NO cerrable. Requiere "
            f"intervención manual.", parse_mode=None)

    async def _unresolved_alert_loop(self) -> None:
        """
        Cada `unresolved_alert_secs` reporta a Telegram los trades en estados
        problemáticos (huérfanos / fallos de protección / pares partidos) que
        requieren intervención manual, hasta que se resuelvan.
        """
        problem = ("failed_unprotected", "orphaned",
                   "pair_leg_a_orphaned", "pair_leg_b_close_failed")
        while not self._stop.is_set():
            try:
                await asyncio.sleep(self.cfg.unresolved_alert_secs)
                if self.pg_pool is None:
                    continue
                rows = await self.pg_pool.fetch("""
                    SELECT client_order_id, symbol, side, status, size_base
                      FROM trades
                     WHERE status = ANY($1::text[])
                """, list(problem))
                if not rows:
                    continue
                lines = [
                    f"{r['status']} {r['symbol']} {r['side']} "
                    f"size={r['size_base']} coid={r['client_order_id']}"
                    for r in rows
                ]
                await self.tg.send(
                    "🆘 Trades sin resolver (intervención manual):\n"
                    + self.tg.code_block("\n".join(lines)))
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("unresolved_alert_loop")

    # ---- Reconciliación de órdenes (cierre natural por SL/TP) ----

    async def _order_reconciliation_loop(self):
        """
        Task de background NO bloqueante. Cada `reconciliation_period_secs`
        busca trades 'open' en Postgres y, por cada uno, comprueba si sus
        órdenes de protección (SL/TP) siguen vivas en el exchange.

        Semántica OCO: cuando una pierna (SL o TP) se ejecuta, la otra se
        cancela atómicamente; si NO queda ninguna orden abierta para el
        símbolo, el round-trip se cerró de forma natural. Entonces capturamos
        el exit fill, calculamos el PnL y marcamos el trade 'closed'.

        Toda la I/O de red se aísla por-trade en try/except: un fallo nunca
        congela este loop ni los WS principales (corren en tasks separadas).
        """
        while not self._stop.is_set():
            try:
                await asyncio.sleep(self.cfg.reconciliation_period_secs)
                if self.pg_pool is None:
                    continue
                rows = await self.pg_pool.fetch("""
                    SELECT client_order_id, symbol, asset_class, side,
                           entry_price, size_base,
                           EXTRACT(EPOCH FROM opened_at) AS opened_epoch
                      FROM trades
                     WHERE status = 'open' AND pair_partner IS NULL
                """)
                for row in rows:
                    await self._reconcile_one_trade(row)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error en _order_reconciliation_loop")

    async def _reconcile_one_trade(self, row) -> None:
        """Reconcilia un único trade 'open' contra el exchange (I/O aislada)."""
        symbol = row["symbol"]
        coid = row["client_order_id"]
        try:
            broker = next((b for ac, b in self.brokers.items()
                           if ac.value == row["asset_class"]), None)
            if broker is None:
                return

            open_orders = await broker.fetch_open_orders(symbol)
            if open_orders is None:
                return                      # estado desconocido → no cerrar
            if len(open_orders) > 0:
                return                      # protección aún viva → sigue abierto

            # Sin protección viva: SL o TP se ejecutó. Reconstruimos el exit.
            opened_epoch = row["opened_epoch"]
            since_ms = int(float(opened_epoch) * 1000) if opened_epoch else None
            # El fill de salida es el lado OPUESTO al de entrada y posterior a la
            # apertura: así no confundimos la entrada con la salida (A-6).
            exit_side = "sell" if row["side"] == "long" else "buy"
            exit_px = await broker.fetch_recent_fill_price(
                symbol, since_ms, after_timestamp_ms=since_ms,
                expected_side=exit_side)

            entry_px = (float(row["entry_price"])
                        if row["entry_price"] is not None else None)
            size_base = (float(row["size_base"])
                         if row["size_base"] is not None else None)
            pnl_quote = None
            if None not in (exit_px, entry_px, size_base):
                direction = 1.0 if row["side"] == "long" else -1.0
                pnl_quote = direction * (exit_px - entry_px) * size_base

            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE trades
                       SET status='closed', exit_price=$2, pnl_quote=$3,
                           closed_at=NOW()
                     WHERE client_order_id=$1 AND status='open'
                """, coid, exit_px, pnl_quote)

            # Liberar el slot en memoria si correspondía a este trade.
            st = self.states.get(symbol)
            if (st is not None and st.open_position is not None
                    and st.open_position.get("client_id") == coid):
                st.open_position = None

            logger.info("[RECONCILE] %s cerrado por SL/TP: exit=%s pnl_quote=%s "
                        "(coid=%s)", symbol, exit_px, pnl_quote, coid)
            await self.tg.send(
                f"✅ Cierre conciliado `{symbol}`\n"
                + self.tg.code_block(
                    f"exit={exit_px} pnl_quote={pnl_quote} coid={coid}"))
        except Exception:
            logger.exception("Error reconciliando trade coid=%s (%s)", coid, symbol)

    # ---- Pair dispatcher ----

    async def _pair_dispatcher(self):
        """
        Loop separado para pares. Cada N segundos:
            1. Reajusta cointegración si toca.
            2. Para cada par operable, evalúa señal y posibles entries.
        """
        while not self._stop.is_set():
            try:
                await asyncio.sleep(self.cfg.pair_decision_period_secs)
                for key, pair_state in self.pairs.items():
                    if pair_state.open_position is not None:
                        continue
                    await self._evaluate_pair(pair_state)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error pair_dispatcher")

    async def _evaluate_pair(self, pair: PairState):
        state_a = self.states.get(pair.sym_a)
        state_b = self.states.get(pair.sym_b)
        if state_a is None or state_b is None:
            return
        if not state_a.ready or not state_b.ready:
            return

        # Ambos mercados abiertos?
        open_a = self.session_guard.is_open(pair.sym_a)
        open_b = self.session_guard.is_open(pair.sym_b)
        if not (open_a and open_b):
            return

        # Refit cointegración cada hora
        now = time.time()
        if now - pair.last_refit_ts > 3600 or pair.coint_params is None:
            log_a = np.log(np.array(state_a.kalman_prices, dtype=np.float64))
            log_b = np.log(np.array(state_b.kalman_prices, dtype=np.float64))
            min_len = min(log_a.size, log_b.size)
            if min_len < 100:
                return
            log_a = log_a[-min_len:]; log_b = log_b[-min_len:]
            try:
                pair.coint_params = fit_cointegration_ols(
                    log_a, log_b, rolling_window=60)
                spread = compute_pair_spread(log_a, log_b, pair.coint_params)
                pair.spread_buffer.extend(spread.tolist())
                pair.last_refit_ts = now

                # OU + GARCH sobre spread
                if pair.coint_params.is_operable:
                    pair.spread_ou = fit_ornstein_uhlenbeck(
                        np.array(pair.spread_buffer, dtype=np.float64),
                        dt=1.0 / (252 * 24 * 60 * 60),
                    )
                    if len(pair.spread_buffer) >= 200:
                        spread_returns = np.diff(np.array(pair.spread_buffer,
                                                            dtype=np.float64))
                        if spread_returns.size >= 200:
                            # try/except local SOLO para el GARCH del spread:
                            # ante no-convergencia (frecuente con ticks rápidos
                            # de Testnet) preservamos el último spread_garch
                            # válido y bajamos el log a DEBUG. El except externo
                            # del refit sigue cubriendo errores reales de
                            # cointegración/OU como WARNING.
                            try:
                                pair.spread_garch = fit_garch(spread_returns)
                            except Exception as ge:
                                logger.debug(
                                    "[PAIR %s/%s] spread GARCH fit falló "
                                    "(asimila últimos params válidos): %s",
                                    pair.sym_a, pair.sym_b, ge)

                logger.info("[PAIR %s/%s] refit: β=%.4f verdict=%s",
                              pair.sym_a, pair.sym_b,
                              pair.coint_params.beta,
                              pair.coint_params.verdict.value)
            except Exception as e:
                logger.warning("Pair refit %s/%s falló: %s",
                                 pair.sym_a, pair.sym_b, e)
                return

        if pair.coint_params is None or not pair.coint_params.is_operable:
            return
        if pair.spread_ou is None or pair.spread_garch is None:
            return

        # Z-score actual
        latest_la = math.log(state_a.kalman.x)
        latest_lb = math.log(state_b.kalman.x)
        z = pair_spread_zscore(latest_la, latest_lb,
                                  pair.coint_params, pair.spread_ou)
        if abs(z) < 2.0:
            return

        # ATR aproximado del spread
        spread_arr = np.array(pair.spread_buffer, dtype=np.float64)[-20:]
        atr_spread = float(np.mean(np.abs(np.diff(spread_arr)))) \
                       if spread_arr.size >= 2 else 0.0

        # Equity total (usar el crypto broker como referencia agregada)
        equity = await self._fetch_equity(AssetClass.CRYPTO)

        decision = await self.risk.evaluate_pair_trade(
            symbol_a=pair.sym_a, symbol_b=pair.sym_b,
            current_equity=equity,
            price_a=state_a.kalman.x, price_b=state_b.kalman.x,
            beta_hedge=pair.coint_params.beta,
            spread_zscore=z,
            spread_half_life_sec=pair.spread_ou.half_life
                                   * (252 * 24 * 60 * 60),
            spread_garch_variance=pair.spread_garch.unconditional_variance,
            spread_garch_periods_per_year=252 * 24 * 60 * 60,
            atr_spread=atr_spread,
            cointegration_verdict=pair.coint_params.verdict.value,
            market_is_open_a=open_a, market_is_open_b=open_b,
        )

        if not decision.approved:
            logger.info("[PAIR %s/%s] Rejected: %s",
                          pair.sym_a, pair.sym_b, decision.reason)
            return

        # LLM opcional pair validation
        if self.llm:
            thesis = {
                "symbol_a": pair.sym_a, "symbol_b": pair.sym_b,
                "asset_class_a": state_a.asset_class.value,
                "asset_class_b": state_b.asset_class.value,
                "beta_hedge": pair.coint_params.beta,
                "alpha_intercept": pair.coint_params.alpha,
                "cointegration_verdict": pair.coint_params.verdict.value,
                "adf_pvalue": pair.coint_params.adf_pvalue,
                "beta_rolling_std": pair.coint_params.beta_rolling_std,
                "spread_zscore": z,
                **decision.metrics,
            }
            try:
                validation = await asyncio.wait_for(
                    self.llm.validate_pair_thesis(thesis), timeout=20.0)
                if validation.verdict != ValidationVerdict.ACCEPT:
                    logger.info("[PAIR %s/%s] LLM vetó: %s",
                                  pair.sym_a, pair.sym_b, validation.reasoning)
                    return
            except asyncio.TimeoutError:
                logger.warning("LLM pair timeout, continuando")

        await self._execute_pair_entry(pair, state_a, state_b, decision)

    async def _execute_pair_entry(self, pair: PairState,
                                     state_a: AssetState, state_b: AssetState,
                                     decision):
        """
        long_spread:  long A, short B
        short_spread: short A, long B
        """
        is_long_spread = decision.metrics.get("pair_side") == "long_spread"
        side_a_broker = "buy" if is_long_spread else "sell"
        side_b_broker = "sell" if is_long_spread else "buy"

        # Mismo guard que en la entrada single-asset: cortar antes de gastar
        # client_order_ids si un leg arroja un size_base degenerado.
        try:
            size_a_base = float(decision.leg_a_size_quote) / float(state_a.kalman.x)
            size_b_base = float(decision.leg_b_size_quote) / float(state_b.kalman.x)
        except (TypeError, ValueError, ZeroDivisionError) as e:
            logger.error("[PAIR %s/%s] size_base no computable (%s); descartado.",
                         pair.sym_a, pair.sym_b, e)
            return
        if not (math.isfinite(size_a_base) and size_a_base > 0
                and math.isfinite(size_b_base) and size_b_base > 0):
            logger.error(
                "[PAIR %s/%s] size_base no finito o ≤0 (a=%s b=%s); descartado.",
                pair.sym_a, pair.sym_b, size_a_base, size_b_base)
            return

        broker_a = self.brokers[state_a.asset_class]
        broker_b = self.brokers[state_b.asset_class]
        cid_a = broker_a.make_client_order_id(pair.sym_a, side_a_broker)
        cid_b = broker_b.make_client_order_id(pair.sym_b, side_b_broker)
        side_a = "long" if side_a_broker == "buy" else "short"
        side_b = "long" if side_b_broker == "buy" else "short"

        # ---- OUTBOX (C-4): ambos legs 'pending' ANTES de enviar nada ----
        try:
            async with self.pg_pool.acquire() as conn:
                for sym, ac, side, cid, entry, size_q, size_b, partner in [
                    (pair.sym_a, state_a.asset_class.value, side_a,
                     cid_a, state_a.kalman.x, decision.leg_a_size_quote,
                     size_a_base, pair.sym_b),
                    (pair.sym_b, state_b.asset_class.value, side_b,
                     cid_b, state_b.kalman.x, decision.leg_b_size_quote,
                     size_b_base, pair.sym_a),
                ]:
                    await conn.execute("""
                        INSERT INTO trades
                        (client_order_id, symbol, asset_class, side, pair_partner,
                         entry_price, size_quote, size_base, status,
                         reasoning, risk_metrics)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,'pending',$9,$10)
                        ON CONFLICT (client_order_id) DO NOTHING
                    """, cid, sym, ac, side, partner, entry, size_q, size_b,
                        json.dumps({"reason": decision.reason}),
                        json.dumps(decision.metrics))
        except Exception as e:
            logger.exception("[PAIR %s/%s] No se pudo registrar 'pending'; "
                             "no se envían órdenes.", pair.sym_a, pair.sym_b)
            await self.tg.send(
                f"🚨 DB no disponible: pair {pair.sym_a}/{pair.sym_b} ABORTADO "
                f"antes de enviar.\n{e}", parse_mode=None)
            return

        # Legs sin bracket: el SL/TP es sobre el spread, no sobre cada leg.
        result_a = await broker_a.submit_market_order(
            pair.sym_a, side_a_broker, size_a_base, cid_a)
        if not result_a.success:
            # Nada ejecutado: cancelar ambas filas pending (no quedan posiciones).
            await self.tg.send(f"🚨 Pair leg A falló:\n{result_a.error}",
                               parse_mode=None)
            await self._mark_status(cid_a, "canceled",
                                    notes=("leg A submit failed: "
                                           + (result_a.error or ""))[:500])
            await self._mark_status(cid_b, "canceled",
                                    notes="sibling leg A failed before submit")
            return

        result_b = await broker_b.submit_market_order(
            pair.sym_b, side_b_broker, size_b_base, cid_b)
        if not result_b.success:
            # Leg A ejecutado, leg B no. Rollback de A CON VERIFICACIÓN (C-5).
            await self.tg.send(
                f"🚨 Pair leg B falló, rollback de leg A en curso:\n{result_b.error}",
                parse_mode=None)
            await self._mark_status(cid_b, "canceled",
                                    notes=("leg B submit failed: "
                                           + (result_b.error or ""))[:500])
            entry_a_px = result_a.avg_price or state_a.kalman.x
            await self._rollback_pair_leg_a(
                pair, broker_a, cid_a, side_a, size_a_base, entry_a_px)
            return

        # ---- Ambos legs OK → confirmar 'pending' -> 'open' ----
        entry_a_px = result_a.avg_price or state_a.kalman.x
        entry_b_px = result_b.avg_price or state_b.kalman.x
        pair.open_position = {
            "client_id_a": cid_a, "client_id_b": cid_b,
            "broker_id_a": result_a.broker_order_id,
            "broker_id_b": result_b.broker_order_id,
            "side": "long_spread" if is_long_spread else "short_spread",
            "side_a": side_a, "side_b": side_b,
            "entry_a": entry_a_px, "entry_b": entry_b_px,
            "size_a_base": size_a_base, "size_b_base": size_b_base,
            "size_a_quote": decision.leg_a_size_quote,
            "size_b_quote": decision.leg_b_size_quote,
            "entry_spread": math.log(state_a.kalman.x) - pair.coint_params.beta
                              * math.log(state_b.kalman.x) - pair.coint_params.alpha,
            "sl_spread_distance": decision.stop_loss_price,
            "tp_spread_distance": decision.take_profit_price,
            "opened_at": time.time(),
        }
        ok_a = await self._confirm_entry_open(cid_a, entry_a_px)
        ok_b = await self._confirm_entry_open(cid_b, entry_b_px)
        if ok_a and ok_b:
            await self.tg.send(
                f"📊 *PAIR ENTRY* `{pair.sym_a}/{pair.sym_b}`\n"
                f"Side: `{decision.metrics.get('pair_side')}`\n"
                f"Leg A: `${decision.leg_a_size_quote:,.2f}` @ {entry_a_px:.4f}\n"
                f"Leg B: `${decision.leg_b_size_quote:,.2f}` @ {entry_b_px:.4f}\n"
                + self.tg.code_block(decision.reason[:200])
            )
        else:
            pair.open_position["_db_unconfirmed"] = True
            logger.critical(
                "[PAIR %s/%s] legs ejecutados pero confirmación DB incompleta "
                "(a=%s b=%s); reintento en background.",
                pair.sym_a, pair.sym_b, ok_a, ok_b)
            await self.tg.send(
                f"🚨 CRÍTICO pair {pair.sym_a}/{pair.sym_b}: legs ejecutados pero "
                f"DB no confirmó 'open'. Reintentando; par bloqueado.",
                parse_mode=None)
            self._spawn_bg(self._retry_confirm_pair_open(
                cid_a, cid_b, entry_a_px, entry_b_px, pair))

    async def _rollback_pair_leg_a(self, pair: "PairState", broker_a,
                                    cid_a: str, side_a: str,
                                    size_a_base: float,
                                    entry_a_px: Optional[float]) -> None:
        """
        C-5: el leg B falló; revierte el leg A con reintentos verificados
        (5 intentos, backoff exponencial). Si todos fallan → leg A queda
        'pair_leg_a_orphaned', el par se bloquea indefinidamente y se alerta.
        """
        opp = "sell" if side_a == "long" else "buy"
        backoff = 2.0
        for attempt in range(5):
            res = await broker_a.submit_market_order(
                pair.sym_a, opp, size_a_base,
                broker_a.make_client_order_id(pair.sym_a, opp))
            if res.success:
                exit_px = res.avg_price
                pnl_quote = None
                if exit_px is not None and entry_a_px is not None:
                    direction = 1.0 if side_a == "long" else -1.0
                    pnl_quote = direction * (exit_px - entry_a_px) * size_a_base
                await self._finalize_trade_closed(
                    cid_a, entry_a_px, exit_px, pnl_quote,
                    exit_reason="rollback",
                    notes="leg A rolled back after leg B failure")
                await self.tg.send(
                    f"✅ Rollback leg A OK {pair.sym_a} coid={cid_a} exit={exit_px}",
                    parse_mode=None)
                return
            logger.error("[PAIR %s] rollback leg A intento %d/5 falló: %s",
                         pair.sym_a, attempt + 1, res.error)
            await asyncio.sleep(backoff)
            backoff *= 2

        # Rollback imposible: leg A queda direccional naked. Bloquear el par.
        await self._mark_status(
            cid_a, "pair_leg_a_orphaned",
            notes="rollback failed; naked leg A; manual intervention required")
        pair.open_position = {"status": "pair_leg_a_orphaned",
                              "client_id_a": cid_a, "side_a": side_a,
                              "size_a_base": size_a_base,
                              "opened_at": time.time()}
        await self.tg.send(
            f"🆘 PAIR LEG A ORPHANED {pair.sym_a} {side_a} size={size_a_base} "
            f"coid={cid_a}: rollback imposible, posición naked. Par bloqueado, "
            f"requiere intervención manual.", parse_mode=None)

    async def _retry_confirm_pair_open(self, cid_a: str, cid_b: str,
                                        entry_a_px: Optional[float],
                                        entry_b_px: Optional[float],
                                        pair: "PairState") -> None:
        """Reintenta confirmar 'open' de ambos legs cada 30s hasta lograrlo."""
        done_a = done_b = False
        while not self._stop.is_set() and not (done_a and done_b):
            await asyncio.sleep(30)
            if not done_a:
                done_a = await self._confirm_entry_open(cid_a, entry_a_px)
            if not done_b:
                done_b = await self._confirm_entry_open(cid_b, entry_b_px)
        if done_a and done_b:
            if pair.open_position is not None:
                pair.open_position.pop("_db_unconfirmed", None)
            logger.info("[PAIR] confirm 'open' reintentado OK (%s,%s)",
                        cid_a, cid_b)
            await self.tg.send(
                f"✅ DB reconciliada: pair legs {cid_a}/{cid_b} ahora 'open'.",
                parse_mode=None)

    # ---- Cierre de pares (C-1) ----

    async def _pair_exit_loop(self) -> None:
        """
        Loop dedicado de SALIDA de pares (C-1). Cada `pair_exit_check_secs`
        revisa cada par con posición viva y la cierra si revierte a la media
        (TP), si el spread se aleja más allá del SL, o por timeout.

        Gestiona la posición en memoria (pair.open_position), que tiene todos
        los datos de entrada. NOTA de límite: tras un reinicio del proceso esta
        memoria se pierde; los pares 'open' en DB quedan sin gestionar hasta una
        reentrada (ver sección de honestidad del reporte).
        """
        while not self._stop.is_set():
            try:
                await asyncio.sleep(self.cfg.pair_exit_check_secs)
                for pair in self.pairs.values():
                    pos = pair.open_position
                    if pos is None:
                        continue
                    # Saltar sentinelas de error y posiciones no confirmadas.
                    if pos.get("status") == "pair_leg_a_orphaned":
                        continue
                    if pos.get("_db_unconfirmed"):
                        continue
                    if "side" not in pos:   # no es una posición gestionable
                        continue
                    await self._maybe_exit_pair(pair)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("pair_exit_loop")

    async def _maybe_exit_pair(self, pair: "PairState") -> None:
        pos = pair.open_position
        state_a = self.states.get(pair.sym_a)
        state_b = self.states.get(pair.sym_b)
        if state_a is None or state_b is None:
            return
        if state_a.kalman is None or state_b.kalman is None:
            return
        if pair.coint_params is None or pair.spread_ou is None:
            return

        # Frescura de precios: si algún leg no tiene tick reciente, no operar.
        now_dt = dt.datetime.now(UTC)
        for st in (state_a, state_b):
            if st.last_tick_ts is None:
                return
            age = (now_dt - st.last_tick_ts).total_seconds()
            if age > self.cfg.pair_price_max_age_secs:
                logger.debug("[PAIR %s/%s] precio stale (%.0fs); skip salida.",
                             pair.sym_a, pair.sym_b, age)
                return

        # Mismo cálculo de spread/z que la entrada.
        latest_la = math.log(state_a.kalman.x)
        latest_lb = math.log(state_b.kalman.x)
        current_spread = (latest_la - pair.coint_params.beta * latest_lb
                          - pair.coint_params.alpha)
        z = pair_spread_zscore(latest_la, latest_lb,
                                pair.coint_params, pair.spread_ou)

        side = pos["side"]                      # 'long_spread' | 'short_spread'
        entry_spread = pos.get("entry_spread")
        sl_dist = pos.get("sl_spread_distance") or 0.0
        opened_at = pos.get("opened_at", now_dt.timestamp())

        exit_reason = None
        # (a) TAKE PROFIT: el spread revirtió a la media.
        if abs(z) < self.cfg.pair_tp_z_threshold:
            exit_reason = "tp"
        # (b) STOP LOSS en unidades de spread, relativo a la entrada.
        elif sl_dist > 0 and entry_spread is not None:
            if side == "long_spread" and current_spread <= entry_spread - sl_dist:
                exit_reason = "sl"
            elif side == "short_spread" and current_spread >= entry_spread + sl_dist:
                exit_reason = "sl"
        # (c) TIMEOUT.
        if exit_reason is None:
            held_h = (time.time() - opened_at) / 3600.0
            if held_h >= self.cfg.pair_max_hold_hours:
                exit_reason = "timeout"

        if exit_reason is None:
            return

        await self._close_pair(pair, pos, exit_reason, z)

    async def _close_pair(self, pair: "PairState", pos: dict,
                           exit_reason: str, z: float) -> None:
        """Cierra ambos legs con reintentos verificados y persiste el cierre."""
        state_a = self.states.get(pair.sym_a)
        state_b = self.states.get(pair.sym_b)
        broker_a = self.brokers[state_a.asset_class]
        broker_b = self.brokers[state_b.asset_class]

        logger.info("[PAIR %s/%s] Cerrando (%s, z=%.3f).",
                    pair.sym_a, pair.sym_b, exit_reason, z)

        # 3a/3c — Cerrar leg A (3 intentos). Si falla, NO tocamos leg B (evita
        # dejar B naked); reintentaremos el par completo en el próximo ciclo.
        ok_a, exit_a = await self._close_pair_leg(
            broker_a, pair.sym_a, pos["side_a"], pos["size_a_base"],
            pos["opened_at"])
        if not ok_a:
            await self.tg.send(
                f"🚨 PAIR {pair.sym_a}/{pair.sym_b}: cierre de leg A falló tras "
                f"3 intentos; reintento en el próximo ciclo.", parse_mode=None)
            return

        # 3b/3d — Cerrar leg B (3 intentos). Si falla tras cerrar A, el par
        # queda PARCIALMENTE cerrado: leg B abierto y naked.
        ok_b, exit_b = await self._close_pair_leg(
            broker_b, pair.sym_b, pos["side_b"], pos["size_b_base"],
            pos["opened_at"])

        cid_a = pos["client_id_a"]
        cid_b = pos["client_id_b"]
        pnl_a = self._leg_pnl(pos["side_a"], pos.get("entry_a"), exit_a,
                              pos["size_a_base"])

        if not ok_b:
            # Leg A cerrado, leg B no → estado partido, bloquear par.
            await self._finalize_trade_closed(
                cid_a, pos.get("entry_a"), exit_a, pnl_a,
                exit_reason=exit_reason, notes="leg A closed; leg B close FAILED")
            await self._mark_status(
                cid_b, "pair_leg_b_close_failed",
                notes="leg A closed but leg B close failed; naked leg B")
            pair.open_position = {"status": "pair_leg_b_close_failed",
                                  "client_id_b": cid_b, "side_b": pos["side_b"],
                                  "size_b_base": pos["size_b_base"],
                                  "opened_at": time.time()}
            await self.tg.send(
                f"🆘 PAIR LEG B CLOSE FAILED {pair.sym_b} {pos['side_b']} "
                f"size={pos['size_b_base']} coid={cid_b}: leg A cerrado, leg B "
                f"naked. Par bloqueado, requiere intervención manual.",
                parse_mode=None)
            return

        # Ambos legs cerrados: persistir y liberar el slot SOLO tras confirmar.
        pnl_b = self._leg_pnl(pos["side_b"], pos.get("entry_b"), exit_b,
                              pos["size_b_base"])
        done_a = await self._finalize_trade_closed(
            cid_a, pos.get("entry_a"), exit_a, pnl_a, exit_reason=exit_reason)
        done_b = await self._finalize_trade_closed(
            cid_b, pos.get("entry_b"), exit_b, pnl_b, exit_reason=exit_reason)
        if done_a and done_b:
            pair.open_position = None
        else:
            # El cierre en el exchange ocurrió pero la DB no confirmó; reintento
            # en background para no perder el registro ni reabrir el par.
            self._spawn_bg(self._retry_finalize_pair_closed(
                pair, cid_a, cid_b, pos, exit_a, exit_b, pnl_a, pnl_b,
                exit_reason, done_a, done_b))
        total_pnl = (pnl_a or 0.0) + (pnl_b or 0.0)
        await self.tg.send(
            f"✅ *PAIR EXIT* `{pair.sym_a}/{pair.sym_b}` ({exit_reason})\n"
            + self.tg.code_block(
                f"z={z:.3f} exit_a={exit_a} exit_b={exit_b} "
                f"pnl≈{total_pnl:.2f}"))

    async def _close_pair_leg(self, broker, sym: str, entry_side: str,
                               size_base: float, opened_at: float,
                               attempts: int = 3) -> tuple[bool, Optional[float]]:
        """Cierra un leg (lado opuesto al de entrada) con reintentos backoff."""
        opp = "sell" if entry_side == "long" else "buy"
        backoff = 2.0
        for attempt in range(attempts):
            res = await broker.submit_market_order(
                sym, opp, size_base, broker.make_client_order_id(sym, opp))
            if res.success:
                exit_px = res.avg_price
                if exit_px is None:
                    # Fallback: buscar el fill de cierre (lado opuesto, posterior).
                    exit_px = await broker.fetch_recent_fill_price(
                        sym, after_timestamp_ms=int(opened_at * 1000),
                        expected_side=opp)
                return True, exit_px
            logger.error("[PAIR] cierre leg %s intento %d/%d falló: %s",
                         sym, attempt + 1, attempts, res.error)
            await asyncio.sleep(backoff)
            backoff *= 2
        return False, None

    @staticmethod
    def _leg_pnl(entry_side: str, entry_px: Optional[float],
                  exit_px: Optional[float], size_base: float) -> Optional[float]:
        if entry_px is None or exit_px is None:
            return None
        direction = 1.0 if entry_side == "long" else -1.0
        return direction * (exit_px - entry_px) * size_base

    async def _retry_finalize_pair_closed(self, pair, cid_a, cid_b, pos,
                                           exit_a, exit_b, pnl_a, pnl_b,
                                           exit_reason, done_a, done_b) -> None:
        while not self._stop.is_set() and not (done_a and done_b):
            await asyncio.sleep(30)
            if not done_a:
                done_a = await self._finalize_trade_closed(
                    cid_a, pos.get("entry_a"), exit_a, pnl_a,
                    exit_reason=exit_reason)
            if not done_b:
                done_b = await self._finalize_trade_closed(
                    cid_b, pos.get("entry_b"), exit_b, pnl_b,
                    exit_reason=exit_reason)
        if done_a and done_b:
            pair.open_position = None
            logger.info("[PAIR %s/%s] cierre persistido tras reintento.",
                        pair.sym_a, pair.sym_b)

    # ---- Equity, snapshots, session monitor, daily ----

    async def _fetch_equity(self, asset_class: AssetClass) -> float:
        broker = self.brokers.get(asset_class)
        if broker is None:
            return 0.0
        try:
            bal = await broker.fetch_balance()
            eq = bal.total_quote
            # El adaptador devuelve NaN como "indeterminado" cuando hay un
            # fallo transitorio de red y NO hay cache previa. NUNCA propagar
            # NaN al RiskManager: usar el último _last_known_equity en lugar
            # de 0.0 (que dispararía un falso DD del 100%).
            if eq != eq or eq is None:  # NaN-check sin importar math
                last = getattr(self, "_last_known_equity", {}).get(asset_class)
                if last is not None and last > 0:
                    logger.warning(
                        "[%s] equity indeterminada; usando último known=%.2f",
                        asset_class, last)
                    return last
                logger.error(
                    "[%s] equity indeterminada y sin known previo; "
                    "devolviendo 0 (downstream debe abstenerse de operar).",
                    asset_class)
                return 0.0
            # Cachear último valor conocido por asset class.
            if not hasattr(self, "_last_known_equity"):
                self._last_known_equity = {}
            if eq > 0:
                self._last_known_equity[asset_class] = eq
            return eq
        except Exception as e:
            logger.warning("fetch_balance %s falló: %s", asset_class, e)
            last = getattr(self, "_last_known_equity", {}).get(asset_class)
            if last is not None and last > 0:
                return last
            return 0.0

    async def _equity_snapshot_loop(self):
        while not self._stop.is_set():
            try:
                await asyncio.sleep(60)
                # Snapshot por asset class
                for ac, broker in self.brokers.items():
                    try:
                        bal = await broker.fetch_balance()
                        async with self.pg_pool.acquire() as conn:
                            await conn.execute("""
                                INSERT INTO equity_snapshots
                                (equity_quote, available_quote,
                                 open_positions_value, unrealized_pnl,
                                 quote_currency)
                                VALUES ($1, $2, 0, 0, $3)
                            """, bal.total_quote, bal.available_quote,
                                bal.quote_currency)
                    except Exception as e:
                        logger.warning("snapshot %s falló: %s", ac, e)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("equity_snapshot_loop")

    async def _session_monitor_loop(self):
        """
        Cada minuto verifica si algún equity transitó closed→open. Si sí,
        marca gap_pending=True para que el próximo tick reciba gap-handling.
        """
        last_open_state: dict[str, bool] = {}
        while not self._stop.is_set():
            try:
                await asyncio.sleep(30)
                for sym, state in self.states.items():
                    if state.asset_class != AssetClass.EQUITY:
                        continue
                    is_open_now = self.session_guard.is_open(sym)
                    was_open = last_open_state.get(sym, is_open_now)
                    if not was_open and is_open_now:
                        # Transición closed → open
                        state.gap_pending = True
                        logger.info("[%s] Market reopened. gap_pending=True", sym)
                        await self.tg.send(f"🔔 {sym} mercado reabrió. "
                                              f"Gap handler armado para próximo tick.")
                    last_open_state[sym] = is_open_now
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("session_monitor_loop")

    async def _daily_report_loop(self):
        while not self._stop.is_set():
            try:
                now = time.time()
                gm = time.gmtime(now)
                secs_to_midnight = (24 - gm.tm_hour) * 3600 \
                                     - gm.tm_min * 60 - gm.tm_sec + 300
                await asyncio.sleep(secs_to_midnight)
                await self._send_daily_report()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("daily_report_loop")

    async def _send_daily_report(self):
        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT asset_class, symbol, side, pnl_quote, opened_at, closed_at
                FROM trades
                WHERE closed_at >= NOW() - INTERVAL '24 hours'
                ORDER BY closed_at DESC
            """)
        if not rows:
            await self.tg.send("📊 *Reporte 24h*: sin trades cerrados.")
            return
        total = sum(float(r["pnl_quote"] or 0) for r in rows)
        n = len(rows)
        win = sum(1 for r in rows if (r["pnl_quote"] or 0) > 0)
        by_class = {}
        for r in rows:
            ac = r["asset_class"]
            by_class.setdefault(ac, {"n": 0, "pnl": 0.0})
            by_class[ac]["n"] += 1
            by_class[ac]["pnl"] += float(r["pnl_quote"] or 0)

        breakdown = "\n".join(
            f"  • {ac}: {d['n']} trades, PnL `${d['pnl']:,.2f}`"
            for ac, d in by_class.items()
        )
        msg = (
            f"📊 *Reporte 24h*\nTrades: {n} | Win: {win}/{n} "
            f"({(win/n*100 if n else 0):.1f}%)\n"
            f"PnL total: `${total:,.2f}`\n{breakdown}"
        )
        await self.tg.send(msg)


# =============================================================================
# Entry point
# =============================================================================

async def main():
    cfg = BotConfig()
    bot = TradingBot(cfg)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(bot.stop()))

    try:
        await bot.start()
    except* asyncio.CancelledError:
        pass    # stop signal
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
