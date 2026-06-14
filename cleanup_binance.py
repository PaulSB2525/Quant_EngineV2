"""
cleanup_binance.py — Liquidación quirúrgica de Binance Testnet.

Diseñado como script one-shot, asíncrono y idempotente:
    1. Conecta a Binance SPOT Testnet con set_sandbox_mode(True).
    2. load_markets() para tener precisión amount/price disponible.
    3. Para cada par objetivo (BTC/USDT, ETH/USDT):
         a) fetch_open_orders() y cancel_order() en bloque.
         b) fetch_balance(): si hay base remanente > minNotional, MARKET SELL.
    4. Reporta estado final por pantalla.

Diseño defensivo:
    - amount_to_precision() ANTES del SELL para evitar -1013 LOT_SIZE.
    - El script NO opera apalancamiento ni futures. SPOT-only.
    - Cualquier excepción se loguea por símbolo y NO interrumpe la limpieza
      del resto: una orden inválida en BTC no bloquea la liquidación de ETH.
"""

from __future__ import annotations

import asyncio
import os
import sys

import ccxt.pro as ccxtpro


PAIRS = ["BTC/USDT", "ETH/USDT"]


async def cleanup() -> int:
    api_key = os.environ.get("BINANCE_API_KEY", "").strip()
    api_secret = os.environ.get("BINANCE_API_SECRET", "").strip()
    if not api_key or not api_secret:
        print("[ERROR] BINANCE_API_KEY/SECRET vacíos en el entorno.",
              file=sys.stderr)
        return 1

    ex = ccxtpro.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    ex.set_sandbox_mode(True)
    print("[INFO] Conectado a Binance SPOT TESTNET.")

    rc = 0
    try:
        await ex.load_markets()
        print("[INFO] Mercados cargados.")

        # ---------- (a) Cancelar órdenes abiertas por símbolo ----------
        for sym in PAIRS:
            try:
                open_orders = await ex.fetch_open_orders(sym)
            except Exception as e:
                print(f"[WARN] fetch_open_orders({sym}) falló: {e}")
                rc = 2
                continue

            if not open_orders:
                print(f"[OK] {sym}: sin órdenes abiertas.")
                continue

            print(f"[INFO] {sym}: cancelando {len(open_orders)} órdenes...")
            for od in open_orders:
                oid = od.get("id")
                otype = od.get("type")
                oside = od.get("side")
                try:
                    await ex.cancel_order(oid, sym)
                    print(f"  - canceled {oside} {otype} id={oid}")
                except Exception as e:
                    print(f"  ! cancel id={oid} falló: {e}")
                    rc = 2

        # ---------- (b) Liquidar inventario base remanente ----------
        try:
            bal = await ex.fetch_balance()
        except Exception as e:
            print(f"[ERROR] fetch_balance falló: {e}", file=sys.stderr)
            return 3

        for sym in PAIRS:
            base = sym.split("/")[0]
            free = float(bal.get("free", {}).get(base, 0.0) or 0.0)
            total = float(bal.get("total", {}).get(base, 0.0) or 0.0)
            print(f"[BAL] {base}: free={free} total={total}")

            if total <= 0:
                continue

            # Quantize a la precisión de amount del mercado para no fallar
            # con -1013 LOT_SIZE. Usamos `total` (no `free`) por si hay
            # locked legítimo que tras cancel ya pasó a free.
            try:
                amount_str = ex.amount_to_precision(sym, total)
                amount = float(amount_str)
            except Exception as e:
                print(f"  ! amount_to_precision({sym}, {total}) falló: {e}")
                rc = 2
                continue

            if amount <= 0:
                print(f"  - {sym}: amount cuantizado = 0; nada que liquidar.")
                continue

            # Verificar minNotional. En Binance Testnet el filtro NOTIONAL
            # rechaza órdenes pequeñas con -1013 "MIN_NOTIONAL". Si el polvo
            # remanente no llega al mínimo, lo dejamos y reportamos.
            mkt = ex.market(sym)
            limits = (mkt or {}).get("limits") or {}
            cost_min = ((limits.get("cost") or {}).get("min") or 0.0)
            # ticker last para estimar notional
            try:
                t = await ex.fetch_ticker(sym)
                last = float(t.get("last") or t.get("close") or 0.0)
            except Exception:
                last = 0.0
            est_notional = amount * last if last else 0.0
            if cost_min and est_notional and est_notional < cost_min:
                print(f"  - {sym}: dust amount={amount} (~{est_notional:.4f} "
                      f"USDT) < minNotional={cost_min}; sin liquidar.")
                continue

            try:
                order = await ex.create_order(sym, "market", "sell", amount)
                avg = order.get("average") or order.get("price")
                print(f"  ✓ {sym}: SELL MARKET {amount} ejecutada. avg={avg}")
            except Exception as e:
                print(f"  ! {sym}: SELL falló: {e}")
                rc = 2

        # ---------- Reporte final ----------
        try:
            bal_final = await ex.fetch_balance()
            print("\n[FINAL BALANCES]")
            for ccy in ("USDT", "BTC", "ETH"):
                tot = bal_final.get("total", {}).get(ccy, 0.0)
                free = bal_final.get("free", {}).get(ccy, 0.0)
                print(f"  {ccy}: free={free} total={tot}")
        except Exception as e:
            print(f"[WARN] fetch_balance final falló: {e}")
    finally:
        try:
            await ex.close()
        except Exception:
            pass

    return rc


if __name__ == "__main__":
    sys.exit(asyncio.run(cleanup()))
