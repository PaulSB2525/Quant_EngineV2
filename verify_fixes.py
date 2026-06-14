"""
verify_fixes.py — Verificación de los fixes críticos C-1..C-5 (FIX 3).

Ejercita el CÓDIGO REAL de bot_core con dobles en memoria (fake broker + fake
pool Postgres) sin tocar exchanges ni DB. Cubre:

    A. Outbox single-asset: broker OK pero confirmación DB falla → fila queda
       'pending', open_position seteado (bloquea reentrada), alerta enviada.
    B. Reconciliador de singles NO cierra pares (status sigue 'open').
    C. Cierre de par por TP: ambos legs cerrados, 'closed' exit_reason='tp',
       open_position liberado.
    D. Fallo al cerrar leg B tras cerrar leg A → 'pair_leg_b_close_failed',
       par bloqueado, alerta.

Uso:  python3 verify_fixes.py     (sale 0 si todo pasa)
"""

from __future__ import annotations

import asyncio
import datetime as dt
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

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
    """
    DB en memoria. Tabla `trades` = dict[client_order_id] -> row dict.
    Honra el WHERE relevante de cada query usada por el código.
    """
    def __init__(self, fail_confirm_open=False):
        self.rows: dict[str, dict] = {}
        self.fail_confirm_open = fail_confirm_open

    def acquire(self):
        return FakeAcquire(self)

    def seed(self, **row):
        self.rows[row["client_order_id"]] = row

    def _apply(self, sql, args):
        # Normalizamos para discriminar por la cláusula SET (varios statements
        # comparten subcadenas como status='open' en su WHERE).
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
        if "status = ANY" in sql:            # _unresolved_alert_loop
            problem = set(args[0])
            return [r for r in self.rows.values() if r.get("status") in problem]
        # _order_reconciliation_loop
        out = []
        for r in self.rows.values():
            if r.get("status") != "open":
                continue
            if "pair_partner IS NULL" in sql and r.get("pair_partner") is not None:
                continue
            out.append({
                "client_order_id": r["client_order_id"], "symbol": r["symbol"],
                "asset_class": r.get("asset_class", "crypto"),
                "side": r.get("side", "long"),
                "entry_price": r.get("entry_price", 100.0),
                "size_base": r.get("size_base", 1.0),
                "opened_epoch": 1_700_000_000.0,
            })
        return out


class FakeBroker:
    def __init__(self, *, bracket_ok=True, market_ok=True, open_orders=None):
        self.bracket_ok = bracket_ok
        self.market_ok = market_ok
        self._open_orders = open_orders if open_orders is not None else []
        self.market_calls: list[tuple] = []

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


def _bot(pool):
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
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond


# ----------------------------------------------------------------------------
# Escenarios
# ----------------------------------------------------------------------------

async def scenario_A():
    print("Escenario A — outbox single: confirmación DB falla")
    pool = FakePool(fail_confirm_open=True)
    bot = _bot(pool)
    broker = FakeBroker(bracket_ok=True)
    bot.brokers = {AssetClass.CRYPTO: broker}
    st = AssetState("CRYPTO:BTC/USDT", 240, 1500)
    bot.states["CRYPTO:BTC/USDT"] = st

    await bot._execute_single_entry(st, "long", _decision(), 100.0)

    coid = st.open_position["client_id"] if st.open_position else None
    ok = True
    ok &= _check("fila quedó 'pending' (no 'open')",
                 coid is not None and pool.rows[coid]["status"] == "pending")
    ok &= _check("open_position seteado (bloquea reentrada)",
                 st.open_position is not None
                 and st.open_position.get("_db_unconfirmed") is True)
    ok &= _check("alerta CRÍTICA enviada",
                 any("CRÍTICO" in m for m in bot.tg.messages))
    bot._stop.set()
    for t in list(bot._bg_tasks):
        t.cancel()
    return ok


async def scenario_B():
    print("Escenario B — reconciliador no toca pares")
    pool = FakePool()
    pool.seed(client_order_id="single1", symbol="CRYPTO:BTC/USDT",
              asset_class="crypto", side="long", status="open",
              pair_partner=None, entry_price=100.0, size_base=1.0)
    pool.seed(client_order_id="pairlegA", symbol="CRYPTO:ETH/USDT",
              asset_class="crypto", side="long", status="open",
              pair_partner="CRYPTO:BTC/USDT", entry_price=50.0, size_base=2.0)
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

    ok = True
    ok &= _check("single SÍ se cerró (sin protección viva)",
                 pool.rows["single1"]["status"] == "closed")
    ok &= _check("pair leg NO se tocó (sigue 'open')",
                 pool.rows["pairlegA"]["status"] == "open")
    return ok


def _coint():
    return CointegrationParams(
        beta=1.0, alpha=0.0, adf_statistic=-4.0, adf_pvalue=0.01,
        spread_std=0.01, verdict=CointegrationVerdict.COINTEGRATED)


def _ou_at(mu):
    # OU operable centrado en mu → z(mu)=0 (dispara TP).
    return OUParams(theta=1.0, mu=mu, sigma=0.1, half_life=0.69, dt=1.0)


def _pair_with_position(broker_b_ok=True):
    pool = FakePool()
    bot = _bot(pool)
    broker_a = FakeBroker(market_ok=True)
    broker_b = FakeBroker(market_ok=broker_b_ok)
    bot.brokers = {AssetClass.CRYPTO: broker_a}   # se sobreescribe abajo
    sa = AssetState("CRYPTO:ETH/USDT", 240, 1500)
    sb = AssetState("CRYPTO:BTC/USDT", 240, 1500)
    sa.kalman = SimpleNamespace(x=100.0)
    sb.kalman = SimpleNamespace(x=100.0)
    now = dt.datetime.now(UTC)
    sa.last_tick_ts = now
    sb.last_tick_ts = now
    bot.states[sa.asset_string] = sa
    bot.states[sb.asset_string] = sb
    # Brokers distintos por símbolo: ambos crypto, pero queremos control
    # independiente del leg B. Usamos un broker por asset_class; para distinguir
    # legs forzamos que el cierre de B falle vía un broker dedicado.
    pair = PairState(sa.asset_string, sb.asset_string)
    pair.coint_params = _coint()
    current_spread = 0.0  # log(100)-1*log(100)-0 = 0
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


async def scenario_C():
    print("Escenario C — cierre de par por TP")
    bot, pool, pair, broker_a, broker_b = _pair_with_position()
    # Ambos legs son crypto → mismo broker; basta uno que cierre OK.
    bot.brokers = {AssetClass.CRYPTO: broker_a}

    await bot._maybe_exit_pair(pair)

    ok = True
    ok &= _check("leg A 'closed' exit_reason='tp'",
                 pool.rows["cidA"]["status"] == "closed"
                 and pool.rows["cidA"].get("exit_reason") == "tp")
    ok &= _check("leg B 'closed' exit_reason='tp'",
                 pool.rows["cidB"]["status"] == "closed"
                 and pool.rows["cidB"].get("exit_reason") == "tp")
    ok &= _check("open_position liberado (None)", pair.open_position is None)
    ok &= _check("se enviaron ≥2 market de cierre",
                 len(broker_a.market_calls) >= 2)
    return ok


async def scenario_D():
    print("Escenario D — fallo al cerrar leg B tras cerrar leg A")
    bot, pool, pair, broker_a, broker_b = _pair_with_position(broker_b_ok=False)

    # Broker que cierra A OK pero B falla. Como ambos son crypto, usamos un
    # broker cuyo submit_market_order falle SOLO para el símbolo del leg B.
    class SplitBroker(FakeBroker):
        async def submit_market_order(self, asset_string, side, size_base, coid):
            self.market_calls.append((asset_string, side, size_base))
            if asset_string == pair.sym_b:
                return OrderResult(success=False, broker_order_id=None,
                                   client_order_id=coid, error="leg B down")
            return OrderResult(success=True, broker_order_id="m1",
                               client_order_id=coid, avg_price=100.0)

    sb_broker = SplitBroker()
    bot.brokers = {AssetClass.CRYPTO: sb_broker}

    await bot._maybe_exit_pair(pair)

    ok = True
    ok &= _check("leg A cerrado", pool.rows["cidA"]["status"] == "closed")
    ok &= _check("leg B marcado 'pair_leg_b_close_failed'",
                 pool.rows["cidB"]["status"] == "pair_leg_b_close_failed")
    ok &= _check("par bloqueado (sentinela)",
                 pair.open_position is not None
                 and pair.open_position.get("status") == "pair_leg_b_close_failed")
    ok &= _check("alerta enviada",
                 any("LEG B CLOSE FAILED" in m for m in bot.tg.messages))
    return ok


async def main():
    results = []
    for fn in (scenario_A, scenario_B, scenario_C, scenario_D):
        results.append(await fn())
        print()
    total = all(results)
    print("=" * 60)
    print(f"RESULTADO: {sum(results)}/{len(results)} escenarios PASS")
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
