"""
tests/test_fixes.py — Tests de los fixes críticos C-1..C-5 + reconciliador de
arranque (GAP 1/GAP 2).

Ejercita el CÓDIGO REAL de bot_core con dobles en memoria (fake broker + fake
pool Postgres), sin tocar exchanges ni DB.

Ejecutar:  pytest tests/ -v
(asyncio_mode=auto en pytest.ini → los `async def test_*` se ejecutan directo.)
"""

from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

from broker_adapters import OrderResult
from bot_core import AssetState, BotConfig, PairState, TradingBot, UTC
from engine_math import CointegrationParams, CointegrationVerdict, OUParams
from risk_manager import AssetClass, RiskDecision, RiskVerdict


# ----------------------------------------------------------------------------
# Dobles
# ----------------------------------------------------------------------------

class FakeTelegram:
    def __init__(self):
        self.messages: list[str] = []

    async def send(self, text, parse_mode="Markdown"):
        self.messages.append(text)

    @staticmethod
    def code_block(text):
        return str(text)

    async def close(self):
        pass


class FakeConn:
    def __init__(self, pool):
        self.pool = pool

    async def execute(self, sql, *args):
        self.pool._apply(sql, args)


class FakeAcquire:
    def __init__(self, pool):
        self.pool = pool

    async def __aenter__(self):
        return FakeConn(self.pool)

    async def __aexit__(self, *exc):
        return False


class FakePool:
    """DB en memoria. Honra el WHERE relevante de cada query del código."""
    def __init__(self, fail_confirm_open=False):
        self.rows: dict[str, dict] = {}
        self.fail_confirm_open = fail_confirm_open

    def acquire(self):
        return FakeAcquire(self)

    def seed(self, **row):
        full = {
            "client_order_id": None, "symbol": None, "asset_class": "crypto",
            "side": "long", "status": "open", "pair_partner": None,
            "entry_price": 100.0, "size_base": 1.0, "size_quote": 100.0,
            "risk_metrics": None, "opened_epoch": 1_700_000_000.0,
        }
        full.update(row)
        self.rows[full["client_order_id"]] = full

    def _match(self, sql, args):
        s = " ".join(sql.split()).lower()
        out = []
        for r in self.rows.values():
            if "where client_order_id=$1" in s and r["client_order_id"] != args[0]:
                continue
            if "status='pending'" in s and r.get("status") != "pending":
                continue
            if "status='open'" in s and r.get("status") != "open":
                continue
            if "pair_partner is null" in s and r.get("pair_partner") is not None:
                continue
            if "pair_partner is not null" in s and r.get("pair_partner") is None:
                continue
            out.append(r)
        return out

    def _apply(self, sql, args):
        s = " ".join(sql.split()).lower()
        coid = args[0]
        if "insert into trades" in s:
            self.rows.setdefault(coid, {})
            self.rows[coid].update({
                "client_order_id": coid, "symbol": args[1],
                "status": "pending",
                "pair_partner": args[4] if "pair_partner" in s else None,
            })
            return
        if "set status='closed'" in s:
            if coid not in self.rows:
                return
            if "exit_reason=" in s:           # _finalize_trade_closed
                self.rows[coid].update(status="closed", exit_price=args[2],
                                       pnl_quote=args[3], exit_reason=args[4])
            else:                             # reconciliador de singles
                self.rows[coid].update(status="closed", exit_price=args[1],
                                       pnl_quote=args[2])
            return
        if "set status='open'" in s:          # _confirm_entry_open
            if self.fail_confirm_open:
                raise RuntimeError("simulated Postgres down on confirm 'open'")
            if coid in self.rows:
                self.rows[coid]["status"] = "open"
            return
        if "set status=$2" in s:              # _mark_status
            if coid in self.rows:
                self.rows[coid]["status"] = args[1]
            return

    async def fetch(self, sql, *args):
        if "status = any" in " ".join(sql.split()).lower():
            problem = set(args[0])            # _unresolved_alert_loop
            return [r for r in self.rows.values() if r.get("status") in problem]
        return self._match(sql, args)

    async def fetchrow(self, sql, *args):
        rows = self._match(sql, args)
        return rows[0] if rows else None


class FakeBroker:
    def __init__(self, *, bracket_ok=True, market_ok=True, open_orders=None,
                 positions=None):
        self.bracket_ok = bracket_ok
        self.market_ok = market_ok
        self._open_orders = open_orders if open_orders is not None else []
        self.positions = positions or {}     # asset_string -> qty (None=unknown)
        self.market_calls: list[tuple] = []

    async def fetch_position(self, asset_string):
        return self.positions.get(asset_string)

    @staticmethod
    def make_client_order_id(asset_string, side):
        return f"qt-{asset_string}-{side}-{id(object()):x}"

    async def submit_bracket_orders(self, **kw):
        if self.bracket_ok:
            return OrderResult(success=True, broker_order_id="b1",
                               client_order_id=kw["client_order_id"],
                               avg_price=kw.get("base_price") or 100.0)
        return OrderResult(success=False, broker_order_id=None,
                           client_order_id=kw["client_order_id"],
                           error="bracket failed", panic_closed=False)

    async def submit_market_order(self, asset_string, side, size_base, coid):
        self.market_calls.append((asset_string, side, size_base))
        if self.market_ok:
            return OrderResult(success=True, broker_order_id="m1",
                               client_order_id=coid, avg_price=100.0)
        return OrderResult(success=False, broker_order_id=None,
                           client_order_id=coid, error="market failed")

    async def fetch_open_orders(self, asset_string):
        return self._open_orders

    async def fetch_recent_fill_price(self, asset_string, since_ms=None,
                                       after_timestamp_ms=None,
                                       expected_side=None):
        return 101.0


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _bot(pool):
    import asyncio
    bot = TradingBot(BotConfig())
    bot.pg_pool = pool
    bot.tg = FakeTelegram()
    bot._stop = asyncio.Event()
    return bot


def _decision():
    return RiskDecision(
        verdict=RiskVerdict.APPROVED, size_quote=1000.0,
        stop_loss_price=95.0, take_profit_price=110.0,
        reason="test", metrics={})


def _check(name, cond):
    assert cond, name


def _coint():
    return CointegrationParams(
        beta=1.0, alpha=0.0, adf_statistic=-4.0, adf_pvalue=0.01,
        spread_std=0.01, verdict=CointegrationVerdict.COINTEGRATED)


def _ou_at(mu):
    return OUParams(theta=1.0, mu=mu, sigma=0.1, half_life=0.69, dt=1.0)


def _pair_with_position(broker_b_ok=True):
    pool = FakePool()
    bot = _bot(pool)
    broker_a = FakeBroker(market_ok=True)
    broker_b = FakeBroker(market_ok=broker_b_ok)
    bot.brokers = {AssetClass.CRYPTO: broker_a}
    sa = AssetState("CRYPTO:ETH/USDT", 240, 1500)
    sb = AssetState("CRYPTO:BTC/USDT", 240, 1500)
    sa.kalman = SimpleNamespace(x=100.0)
    sb.kalman = SimpleNamespace(x=100.0)
    now = dt.datetime.now(UTC)
    sa.last_tick_ts = now
    sb.last_tick_ts = now
    bot.states[sa.asset_string] = sa
    bot.states[sb.asset_string] = sb
    pair = PairState(sa.asset_string, sb.asset_string)
    pair.coint_params = _coint()
    current_spread = 0.0
    pair.spread_ou = _ou_at(current_spread)
    pair.open_position = {
        "client_id_a": "cidA", "client_id_b": "cidB",
        "side": "long_spread", "side_a": "long", "side_b": "short",
        "entry_a": 100.0, "entry_b": 100.0,
        "size_a_base": 1.0, "size_b_base": 1.0,
        "entry_spread": current_spread,
        "sl_spread_distance": 0.05, "tp_spread_distance": 0.1,
        "opened_at": now.timestamp(),
    }
    pool.seed(client_order_id="cidA", symbol=sa.asset_string, status="open",
              pair_partner=sb.asset_string)
    pool.seed(client_order_id="cidB", symbol=sb.asset_string, status="open",
              pair_partner=sa.asset_string)
    bot.pairs[(sa.asset_string, sb.asset_string)] = pair
    return bot, pool, pair, broker_a, broker_b


def _recon_json():
    recon = {
        "pair_side": "long_spread", "sym_a": "CRYPTO:ETH/USDT",
        "sym_b": "CRYPTO:BTC/USDT", "cid_a": "cidA", "cid_b": "cidB",
        "side_a": "long", "side_b": "short",
        "entry_a": 100.0, "entry_b": 100.0,
        "size_a_base": 1.0, "size_b_base": 1.0,
        "size_a_quote": 100.0, "size_b_quote": 100.0,
        "entry_spread": 0.0, "sl_spread_distance": 0.05,
        "tp_spread_distance": 0.1,
        "beta": 1.0, "alpha": 0.0,
        "ou_theta": 1.0, "ou_mu": 0.0, "ou_sigma": 0.1,
        "ou_half_life": 0.69, "ou_dt": 1.0,
    }
    return json.dumps({"_recon": recon})


# ----------------------------------------------------------------------------
# Tests — fixes C-1..C-5
# ----------------------------------------------------------------------------

async def test_scenario_a_outbox_single_db_confirm_fails():
    pool = FakePool(fail_confirm_open=True)
    bot = _bot(pool)
    bot.brokers = {AssetClass.CRYPTO: FakeBroker(bracket_ok=True)}
    st = AssetState("CRYPTO:BTC/USDT", 240, 1500)
    bot.states["CRYPTO:BTC/USDT"] = st

    await bot._execute_single_entry(st, "long", _decision(), 100.0)

    coid = st.open_position["client_id"] if st.open_position else None
    _check("fila quedó 'pending'",
           coid is not None and pool.rows[coid]["status"] == "pending")
    _check("open_position bloquea reentrada",
           st.open_position is not None
           and st.open_position.get("_db_unconfirmed") is True)
    _check("alerta CRÍTICA enviada", any("CRÍTICO" in m for m in bot.tg.messages))
    bot._stop.set()
    for t in list(bot._bg_tasks):
        t.cancel()


async def test_scenario_b_reconciler_skips_pairs():
    import asyncio
    pool = FakePool()
    pool.seed(client_order_id="single1", symbol="CRYPTO:BTC/USDT",
              status="open", pair_partner=None)
    pool.seed(client_order_id="pairlegA", symbol="CRYPTO:ETH/USDT",
              status="open", pair_partner="CRYPTO:BTC/USDT")
    bot = _bot(pool)
    bot.brokers = {AssetClass.CRYPTO: FakeBroker(open_orders=[])}
    bot.cfg.reconciliation_period_secs = 0.02

    task = asyncio.create_task(bot._order_reconciliation_loop())
    await asyncio.sleep(0.15)
    bot._stop.set()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    _check("single SÍ se cerró", pool.rows["single1"]["status"] == "closed")
    _check("pair leg NO se tocó", pool.rows["pairlegA"]["status"] == "open")


async def test_scenario_c_pair_exit_tp():
    bot, pool, pair, broker_a, _ = _pair_with_position()
    bot.brokers = {AssetClass.CRYPTO: broker_a}

    await bot._maybe_exit_pair(pair)

    _check("leg A closed/tp", pool.rows["cidA"]["status"] == "closed"
           and pool.rows["cidA"].get("exit_reason") == "tp")
    _check("leg B closed/tp", pool.rows["cidB"]["status"] == "closed"
           and pool.rows["cidB"].get("exit_reason") == "tp")
    _check("open_position liberado", pair.open_position is None)
    _check("≥2 market de cierre", len(broker_a.market_calls) >= 2)


async def test_scenario_d_pair_leg_b_close_fail():
    bot, pool, pair, _, _ = _pair_with_position()

    class SplitBroker(FakeBroker):
        async def submit_market_order(self, asset_string, side, size_base, coid):
            self.market_calls.append((asset_string, side, size_base))
            if asset_string == pair.sym_b:
                return OrderResult(success=False, broker_order_id=None,
                                   client_order_id=coid, error="leg B down")
            return OrderResult(success=True, broker_order_id="m1",
                               client_order_id=coid, avg_price=100.0)

    bot.brokers = {AssetClass.CRYPTO: SplitBroker()}

    await bot._maybe_exit_pair(pair)

    _check("leg A cerrado", pool.rows["cidA"]["status"] == "closed")
    _check("leg B pair_leg_b_close_failed",
           pool.rows["cidB"]["status"] == "pair_leg_b_close_failed")
    _check("par bloqueado",
           pair.open_position is not None
           and pair.open_position.get("status") == "pair_leg_b_close_failed")
    _check("alerta enviada",
           any("LEG B CLOSE FAILED" in m for m in bot.tg.messages))


# ----------------------------------------------------------------------------
# Tests — reconciliador de arranque (GAP 1/GAP 2)
# ----------------------------------------------------------------------------

async def test_startup_reconciler_pending_resolves_to_open():
    pool = FakePool()
    pool.seed(client_order_id="p1", symbol="CRYPTO:BTC/USDT", side="long",
              status="pending", pair_partner=None)
    bot = _bot(pool)
    bot.brokers = {AssetClass.CRYPTO: FakeBroker(
        positions={"CRYPTO:BTC/USDT": 1.0})}
    bot.states["CRYPTO:BTC/USDT"] = AssetState("CRYPTO:BTC/USDT", 240, 1500)

    await bot._startup_reconciler()

    _check("pending → open", pool.rows["p1"]["status"] == "open")
    _check("open_position reconstruido",
           bot.states["CRYPTO:BTC/USDT"].open_position is not None)


async def test_startup_reconciler_pending_no_position_marks_failed():
    pool = FakePool()
    pool.seed(client_order_id="p2", symbol="CRYPTO:BTC/USDT", side="long",
              status="pending", pair_partner=None)
    bot = _bot(pool)

    class NoFillBroker(FakeBroker):
        async def fetch_recent_fill_price(self, *a, **k):
            return None

    bot.brokers = {AssetClass.CRYPTO: NoFillBroker(
        positions={"CRYPTO:BTC/USDT": 0.0})}
    bot.states["CRYPTO:BTC/USDT"] = AssetState("CRYPTO:BTC/USDT", 240, 1500)

    await bot._startup_reconciler()

    _check("pending → failed_no_position",
           pool.rows["p2"]["status"] == "failed_no_position")


async def test_startup_reconciler_pair_reconstructed_in_memory():
    pool = FakePool()
    pool.seed(client_order_id="cidA", symbol="CRYPTO:ETH/USDT", side="long",
              status="open", pair_partner="CRYPTO:BTC/USDT",
              risk_metrics=_recon_json())
    pool.seed(client_order_id="cidB", symbol="CRYPTO:BTC/USDT", side="short",
              status="open", pair_partner="CRYPTO:ETH/USDT",
              risk_metrics=_recon_json())
    bot = _bot(pool)
    bot.brokers = {AssetClass.CRYPTO: FakeBroker(positions={
        "CRYPTO:ETH/USDT": 1.0, "CRYPTO:BTC/USDT": 1.0})}
    pair = PairState("CRYPTO:ETH/USDT", "CRYPTO:BTC/USDT")
    bot.pairs[("CRYPTO:ETH/USDT", "CRYPTO:BTC/USDT")] = pair

    await bot._startup_reconciler()

    _check("pair.open_position reconstruido",
           pair.open_position is not None
           and pair.open_position.get("side") == "long_spread")
    _check("coint_params rehidratado", pair.coint_params is not None)
    _check("spread_ou rehidratado", pair.spread_ou is not None)


async def test_startup_reconciler_pair_partially_closed_externally():
    pool = FakePool()
    pool.seed(client_order_id="cidA", symbol="CRYPTO:ETH/USDT", side="long",
              status="open", pair_partner="CRYPTO:BTC/USDT",
              risk_metrics=_recon_json())
    pool.seed(client_order_id="cidB", symbol="CRYPTO:BTC/USDT", side="short",
              status="open", pair_partner="CRYPTO:ETH/USDT",
              risk_metrics=_recon_json())
    bot = _bot(pool)
    broker = FakeBroker(positions={"CRYPTO:ETH/USDT": 1.0,
                                   "CRYPTO:BTC/USDT": 0.0})
    bot.brokers = {AssetClass.CRYPTO: broker}
    pair = PairState("CRYPTO:ETH/USDT", "CRYPTO:BTC/USDT")
    bot.pairs[("CRYPTO:ETH/USDT", "CRYPTO:BTC/USDT")] = pair

    await bot._startup_reconciler()

    _check("leg A closed/startup_partial_close",
           pool.rows["cidA"]["status"] == "closed"
           and pool.rows["cidA"].get("exit_reason") == "startup_partial_close")
    _check("leg B cerrado", pool.rows["cidB"]["status"] == "closed")
    _check("market de cierre del leg vivo",
           any(c[0] == "CRYPTO:ETH/USDT" for c in broker.market_calls))
    _check("open_position liberado", pair.open_position is None)
