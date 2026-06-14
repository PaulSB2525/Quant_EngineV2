"""
broker_adapters.py  (v2)
========================
Abstracción multi-broker para ejecución y data streaming.

Arquitectura:
    BaseBrokerAdapter (ABC)
        ├── BinanceBrokerAdapter      (ccxt.pro websockets)
        └── AlpacaBrokerAdapter       (alpaca-py async, fallback a REST+SSE)

Cada adapter encapsula:
    - Conexión y reconexión (WS o SSE).
    - Routing de símbolos nativos (BTC/USDT vs AAPL).
    - Submit/cancel de órdenes con su semántica nativa.
    - Bracket orders (entry + SL + TP) con la mejor primitiva disponible.

Contrato:
    El bot core nunca sabe si el adapter usa REST o WS. Solo recibe callbacks
    `on_ticker(asset_string, data)` donde `asset_string` lleva el prefijo
    "CRYPTO:" o "EQUITY:" para que el RiskManager pueda diferenciar.
"""

from __future__ import annotations

import abc
import asyncio
import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


# Errores transitorios de red / DNS / pool / 5xx en los que NO debemos
# tratar el balance como 0. Lista por nombre para evitar imports duros
# que rompan el cold-start si alpaca-py / requests no están disponibles.
_TRANSIENT_BALANCE_EXC_NAMES = {
    "ConnectionError", "ConnectionRefusedError", "ConnectionResetError",
    "ConnectTimeout", "ConnectTimeoutError", "ReadTimeout", "ReadTimeoutError",
    "Timeout", "TimeoutError", "MaxRetryError", "NewConnectionError",
    "ProtocolError", "RemoteDisconnected", "ChunkedEncodingError",
    "SSLError", "APIError", "HTTPError", "ClientConnectionError",
    "ClientConnectorError", "ServerDisconnectedError", "TooManyRedirects",
    "RetryError",
}


def _is_transient_balance_error(exc: BaseException) -> bool:
    """True si la excepción es claramente transitoria (red / 5xx)."""
    name = type(exc).__name__
    if name in _TRANSIENT_BALANCE_EXC_NAMES:
        return True
    msg = str(exc).lower()
    transient_signals = (
        "connection refused", "max retries exceeded", "timed out",
        "temporarily unavailable", "service unavailable", "bad gateway",
        "gateway timeout", "name or service not known", "no route to host",
        "connection reset", "remote end closed", "broken pipe",
    )
    return any(sig in msg for sig in transient_signals)


# =============================================================================
# DTOs unificados
# =============================================================================

@dataclass
class TickerData:
    """Snapshot de un ticker normalizado entre brokers."""
    asset_string: str          # "CRYPTO:BTC/USDT" o "EQUITY:AAPL"
    bid: float
    ask: float
    last: float
    timestamp_ns: int          # ns desde epoch
    bid_size: float = 0.0
    ask_size: float = 0.0

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class OrderResult:
    """Resultado de un submit_order normalizado."""
    success: bool
    broker_order_id: Optional[str]
    client_order_id: str
    error: Optional[str] = None
    fills: list[dict] = field(default_factory=list)
    avg_price: Optional[float] = None    # precio medio de fill (entrada market)
    exit_price: Optional[float] = None   # precio del cierre de pánico (si lo hubo)
    panic_closed: bool = False           # True solo si el cierre de pánico se confirmó


@dataclass
class BalanceData:
    """Balance en la moneda quote."""
    total_quote: float
    available_quote: float
    quote_currency: str        # 'USDT', 'USD', etc.


# Tipo del callback de ticker
TickerCallback = Callable[[TickerData], Awaitable[None]]


# =============================================================================
# Base ABC
# =============================================================================

class BaseBrokerAdapter(abc.ABC):
    """
    Contrato común a todos los brokers. Las implementaciones encapsulan
    diferencias de API. Métodos abstractos son los mínimos imprescindibles;
    métodos concretos proveen helpers (idempotency, retry de bajo nivel).
    """

    def __init__(self, quote_currency: str = "USDT"):
        self.quote_currency = quote_currency
        self._ticker_callbacks: dict[str, TickerCallback] = {}
        self._stop_event = asyncio.Event()

    # ---- Lifecycle ----

    @abc.abstractmethod
    async def connect(self) -> None: ...

    @abc.abstractmethod
    async def close(self) -> None: ...

    # ---- Streaming ----

    @abc.abstractmethod
    async def subscribe_ticker(self, asset_string: str,
                                 callback: TickerCallback) -> None:
        """Inicia el stream WS y registra el callback. Maneja reconnect."""

    @abc.abstractmethod
    async def unsubscribe_ticker(self, asset_string: str) -> None:
        """Cierra el stream. Útil cuando un equity cierra para liberar WS."""

    # ---- Execution ----

    @abc.abstractmethod
    async def submit_market_order(self, asset_string: str, side: str,
                                    size_base: float,
                                    client_order_id: str) -> OrderResult:
        """side ∈ {'buy', 'sell'}."""

    @abc.abstractmethod
    async def submit_bracket_orders(self, asset_string: str, side: str,
                                      size_base: float, stop_loss: float,
                                      take_profit: float,
                                      client_order_id: str,
                                      base_price: Optional[float] = None
                                      ) -> OrderResult:
        """
        Envia entry + SL + TP. Cada broker usa su mejor primitiva:
            - Binance: order de entrada market, luego OCO con SL/TP.
            - Alpaca: bracket order nativa.

        `base_price` es el precio de referencia del mercado al disparar la
        entrada (el tick que la motivó). Alpaca lo usa para clampar las piernas
        de protección a la microestructura del validador; Binance lo ignora
        porque deriva su referencia del fill real de la entrada market.
        """

    @abc.abstractmethod
    async def cancel_order(self, asset_string: str,
                           broker_order_id: str) -> bool: ...

    @abc.abstractmethod
    async def fetch_balance(self) -> BalanceData: ...

    # ---- Reconciliación de órdenes (default opcional por adapter) ----

    async def fetch_open_orders(self, asset_string: str) -> Optional[list[dict]]:
        """
        Órdenes abiertas del símbolo, incluidas las condicionales SL/TP.

        Devuelve None si el adapter NO soporta la consulta o si hubo un error
        de red: el reconciliador debe tratarlo como 'estado desconocido' y NO
        cerrar el trade. Una lista vacía [] sí significa 'no quedan
        protecciones vivas' (en OCO, una pierna ejecutada cancela la otra).
        """
        return None

    async def fetch_recent_fill_price(self, asset_string: str,
                                       since_ms: Optional[int] = None,
                                       after_timestamp_ms: Optional[int] = None,
                                       expected_side: Optional[str] = None
                                       ) -> Optional[float]:
        """
        Precio del fill más reciente (exit del round-trip). None si se ignora.

        after_timestamp_ms : solo considerar fills con timestamp >= a este valor
                             (evita confundir el fill de ENTRADA con el de SALIDA).
        expected_side      : 'buy'/'sell' — solo fills de este lado (el lado de
                             cierre es el OPUESTO al de entrada).
        """
        return None

    # ---- Helpers concretos ----

    @staticmethod
    def make_client_order_id(asset_string: str, side: str) -> str:
        payload = f"{asset_string}|{side}|{time.time_ns()}".encode()
        return "qt-" + hashlib.sha256(payload).hexdigest()[:24]


# =============================================================================
# BinanceBrokerAdapter (ccxt.pro)
# =============================================================================

class BinanceBrokerAdapter(BaseBrokerAdapter):
    """
    Implementación usando ccxt.pro. Soporta spot por defecto; cambia
    `default_type` a 'future' para perp futures.

    Reconnect: ccxt.pro maneja reconnect de bajo nivel; nosotros añadimos
    backoff exponencial sobre el watch_order_book loop.
    """

    def __init__(self, api_key: str, api_secret: str,
                 paper_trading: bool = True,
                 quote_currency: str = "USDT",
                 max_reconnect_backoff: float = 30.0):
        super().__init__(quote_currency=quote_currency)
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper_trading = paper_trading
        self.max_reconnect_backoff = max_reconnect_backoff
        self._exchange = None
        self._stream_tasks: dict[str, asyncio.Task] = {}

    async def connect(self) -> None:
        import ccxt.pro as ccxtpro
        cls = getattr(ccxtpro, "binance")
        self._exchange = cls({
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        if self.paper_trading:
            self._exchange.set_sandbox_mode(True)
        # load_markets() materializa los filtros LOT_SIZE/PRICE_FILTER/NOTIONAL
        # que necesitamos para amount_to_precision/price_to_precision al enviar
        # órdenes. Sin esto, ccxt manda floats con decimales arbitrarios y el
        # exchange rechaza con -1013. Aislado en try/except: un fallo transitorio
        # de red no debe abortar el cold-start; los streams ws cargan markets
        # bajo demanda más adelante, y un retry tardío rellena la caché.
        try:
            await self._exchange.load_markets()
        except Exception as e:
            logger.warning(
                "Binance load_markets() falló en connect(): %s. Los streams "
                "cargarán mercados bajo demanda; las primeras órdenes podrían "
                "rebotar con -1013 hasta que la caché se rellene.", e)
        logger.info("Binance connected (paper=%s)", self.paper_trading)

    async def close(self) -> None:
        self._stop_event.set()
        for task in self._stream_tasks.values():
            task.cancel()
        await asyncio.gather(*self._stream_tasks.values(), return_exceptions=True)
        if self._exchange:
            await self._exchange.close()

    async def subscribe_ticker(self, asset_string: str,
                                 callback: TickerCallback) -> None:
        self._ticker_callbacks[asset_string] = callback
        if asset_string not in self._stream_tasks:
            task = asyncio.create_task(self._ticker_loop(asset_string),
                                         name=f"binance:{asset_string}")
            self._stream_tasks[asset_string] = task

    async def unsubscribe_ticker(self, asset_string: str) -> None:
        task = self._stream_tasks.pop(asset_string, None)
        if task:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._ticker_callbacks.pop(asset_string, None)

    async def _ticker_loop(self, asset_string: str):
        """
        WS loop con reconnect exponencial.

        Las AuthenticationError de ccxt son terminales (key inválida, IP no
        whitelisted, permisos faltantes) — reintentarlas no las repara y
        consume cuota de rate-limit. Detectamos por nombre de la excepción
        para evitar un import duro de ccxt.base.errors que retrasaría el
        cold-start; el contenedor seguirá vivo y otros símbolos podrán
        seguir streaming.
        """
        native = self._native_symbol(asset_string)
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                ob = await self._exchange.watch_order_book(native, limit=20)
                backoff = 1.0   # reset
                bids = ob["bids"]; asks = ob["asks"]
                if not bids or not asks:
                    continue
                ticker = TickerData(
                    asset_string=asset_string,
                    bid=float(bids[0][0]), ask=float(asks[0][0]),
                    last=0.5 * (float(bids[0][0]) + float(asks[0][0])),
                    timestamp_ns=time.time_ns(),
                    bid_size=float(bids[0][1]), ask_size=float(asks[0][1]),
                )
                cb = self._ticker_callbacks.get(asset_string)
                if cb is not None:
                    await cb(ticker)
            except asyncio.CancelledError:
                break
            except Exception as e:
                exc_name = type(e).__name__
                if exc_name in ("AuthenticationError", "PermissionDenied",
                                "InvalidNonce"):
                    logger.error(
                        "Binance WS %s: error TERMINAL (%s): %s. Abortando "
                        "loop de este símbolo; no se reintentará hasta restart.",
                        asset_string, exc_name, e)
                    return
                logger.error("Binance WS error %s: %s. Backoff %.1fs",
                              asset_string, e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.max_reconnect_backoff)

    # ---- Helpers de cuantización (LOT_SIZE / PRICE_FILTER) ----
    # Binance rechaza con -1013 cualquier qty cuyo step no calce con stepSize
    # o cualquier price cuyo tick no calce con tickSize. ccxt expone
    # amount_to_precision/price_to_precision que truncan al múltiplo válido
    # cuando load_markets() ya pobló los filtros. Si load_markets() aún no
    # ha completado (cold-start con red lenta), caemos al valor crudo: el
    # exchange dirá si lo rechaza y dejamos la huella en el log de error.

    def _quantize_amount(self, native: str, amount: float) -> float:
        try:
            return float(self._exchange.amount_to_precision(native, amount))
        except Exception as e:
            logger.debug("amount_to_precision(%s, %s) falló: %s; usando crudo.",
                          native, amount, e)
            return float(amount)

    def _quantize_price(self, native: str, price: float) -> float:
        try:
            return float(self._exchange.price_to_precision(native, price))
        except Exception as e:
            logger.debug("price_to_precision(%s, %s) falló: %s; usando crudo.",
                          native, price, e)
            return float(price)

    async def submit_market_order(self, asset_string: str, side: str,
                                    size_base: float,
                                    client_order_id: str) -> OrderResult:
        native = self._native_symbol(asset_string)
        # Guard NaN/Inf y cuantización a stepSize. amount_to_precision puede
        # truncar a 0 si stepSize es mayor que size_base: en ese caso abortamos
        # antes de enviar para no recibir un -1013 vacío.
        try:
            amount_f = float(size_base)
        except (TypeError, ValueError) as e:
            return OrderResult(success=False, broker_order_id=None,
                               client_order_id=client_order_id,
                               error=f"size_base no numérico: {e}")
        if not math.isfinite(amount_f) or amount_f <= 0:
            return OrderResult(success=False, broker_order_id=None,
                               client_order_id=client_order_id,
                               error=f"size_base no finito o ≤0: {amount_f}")
        amount = self._quantize_amount(native, amount_f)
        if amount <= 0:
            return OrderResult(
                success=False, broker_order_id=None,
                client_order_id=client_order_id,
                error=f"size_base {amount_f} cuantizado a 0 por stepSize de {native}")
        try:
            order = await self._exchange.create_order(
                native, "market", side, amount, None,
                {"clientOrderId": client_order_id},
            )
            avg = order.get("average") or order.get("price")
            return OrderResult(
                success=True, broker_order_id=order.get("id"),
                client_order_id=client_order_id,
                fills=order.get("trades", []),
                avg_price=float(avg) if avg else None,
            )
        except Exception as e:
            logger.exception("Binance submit_market_order falló: %s", e)
            return OrderResult(success=False, broker_order_id=None,
                               client_order_id=client_order_id, error=str(e))

    async def submit_bracket_orders(self, asset_string: str, side: str,
                                      size_base: float, stop_loss: float,
                                      take_profit: float,
                                      client_order_id: str,
                                      base_price: Optional[float] = None
                                      ) -> OrderResult:
        """
        Binance NO tiene bracket nativa para spot. Estrategia:
            1. Market entry.
            2. Protección SL/TP en el lado opuesto.

        IMPORTANTE (restricción del exchange):
            - Binance SPOT no soporta `STOP_MARKET` ni el flag `reduceOnly`
              (ambos son exclusivos de USDⓈ-M / COIN-M futures). Enviarlos a
              spot produce un -1106 "parameter reduceOnly sent when not
              required" o -1116 "Invalid orderType".
            - Para spot usamos un OCO nativo (liga TP-limit + SL-stop-limit;
              al ejecutarse una pierna, la otra se cancela atómicamente). Si
              el símbolo/cuenta no admite OCO, hacemos fallback a dos órdenes:
              STOP_LOSS_LIMIT (SL) + LIMIT (TP).
            - En futures conservamos STOP_MARKET + reduceOnly (válido allí).

        Si toda la protección falla, cerramos la posición.

        `base_price` se acepta por compatibilidad de interfaz pero NO se usa
        aquí: el clamp de las piernas spot toma su referencia del precio medio
        del fill de entrada (entry_result.avg_price), más preciso que el tick
        de señal.
        """
        entry_result = await self.submit_market_order(
            asset_string, side, size_base, client_order_id)
        if not entry_result.success:
            return entry_result

        native = self._native_symbol(asset_string)
        opposite = "sell" if side == "buy" else "buy"
        is_spot = (self._exchange.options or {}).get("defaultType", "spot") == "spot"
        # Cuantizamos UNA sola vez aquí para que el size de las piernas de
        # protección case con el LOT_SIZE del par: si llegan con decimales no
        # alineados a stepSize, Binance las rechaza con -1013 al armar el OCO.
        size_q = self._quantize_amount(native, float(size_base))
        try:
            if is_spot:
                # Latencia cero: el precio de referencia para el clamp del stop
                # sale del fill de entrada que acabamos de ejecutar, sin REST.
                await self._submit_spot_protection(
                    native, opposite, size_q, stop_loss, take_profit,
                    entry_price=entry_result.avg_price)
            else:
                # Futures: STOP_MARKET + reduceOnly siguen siendo válidos.
                # Aquí también pasamos por price_to_precision para que el
                # stopPrice/take_profit calcen con el tickSize del contrato.
                await self._exchange.create_order(
                    native, "stop_market", opposite, size_q, None,
                    {"stopPrice": self._quantize_price(native, stop_loss),
                     "reduceOnly": True},
                )
                await self._exchange.create_order(
                    native, "limit", opposite, size_q,
                    self._quantize_price(native, take_profit),
                    {"reduceOnly": True},
                )
            return entry_result
        except Exception as e:
            logger.error("Binance SL/TP falló: %s. Cerrando posición.", e)
            try:
                close_order = await self._exchange.create_order(
                    native, "market", opposite, size_q)
                # Exponemos el precio de cierre para que bot_core (dueño de la
                # DB) registre el round-trip como 'closed' con su PnL real. El
                # adapter NO escribe en Postgres: solo entrega el dato.
                close_px = close_order.get("average") or close_order.get("price")
                entry_result.exit_price = float(close_px) if close_px else None
                entry_result.panic_closed = True
                entry_result.error = f"SL/TP failed and position closed: {e}"
            except Exception as close_err:
                logger.exception("CRITICAL: no se pudo cerrar tras SL/TP fail: %s",
                                  close_err)
                # El cierre de pánico falló: la posición sigue ABIERTA en el
                # exchange. NO marcamos panic_closed para no registrar en la DB
                # un cierre que no ocurrió.
                entry_result.panic_closed = False
                entry_result.error = (
                    f"SL/TP failed AND panic close failed (POSICIÓN ABIERTA en "
                    f"exchange): sltp={e}; close={close_err}")
            entry_result.success = False
            return entry_result

    # --- Distancias de protección para las piernas stop en spot (NO confundir) ---
    # (1) Distancia mínima TRIGGER (stopPrice) ↔ PRECIO DE MERCADO. Binance Spot
    #     rechaza la condicional con "Stop price would trigger immediately" si el
    #     trigger queda del lado equivocado o demasiado pegado al last price.
    #     30 bps dan holgura al matching engine para aceptarla sin gatillarla.
    _SPOT_STOP_MIN_DISTANCE = 0.003   # 30 bps  (mercado ↔ trigger)
    # (2) Buffer TRIGGER ↔ LÍMITE del STOP_LOSS_LIMIT. Coloca el límite algo peor
    #     que el trigger para asegurar el fill una vez disparado, a costa de algo
    #     de slippage. Trade-off inherente a no tener STOP_MARKET en spot.
    _SPOT_STOP_LIMIT_BUFFER = 0.001   # 10 bps  (trigger ↔ límite)

    async def _submit_spot_protection(self, native: str, side: str,
                                       size_base: float, stop_loss: float,
                                       take_profit: float,
                                       entry_price: Optional[float] = None) -> None:
        """
        Protección SL/TP para Binance SPOT sin STOP_MARKET ni reduceOnly.

        Intenta primero un OCO nativo (bracket transparente para el exchange);
        si no está disponible, cae a STOP_LOSS_LIMIT + LIMIT separados.

        `entry_price` es el precio medio del fill de entrada (ruta de latencia
        cero: lo pasa submit_bracket_orders, sin ninguna llamada REST).

        Dos distancias en juego (no confundir):
          - mercado→trigger (_SPOT_STOP_MIN_DISTANCE): el stopPrice debe quedar
            suficientemente lejos del precio de entrada o Binance rechaza con
            "Stop price would trigger immediately". Aquí se clampa el trigger.
          - trigger→límite (_SPOT_STOP_LIMIT_BUFFER): el límite se coloca algo
            peor que el trigger para asegurar el fill una vez disparado.
        """
        # Clamp del TRIGGER usando el precio de entrada fresco del fill (sin
        # REST). Garantiza distancia mínima respecto al mercado en el lado
        # protector; solo aleja el trigger si está demasiado cerca — si el risk
        # manager ya lo puso más lejos, se respeta su valor.
        if entry_price:
            if side == "sell":   # proteger long: trigger por DEBAJO del mercado
                stop_loss = min(stop_loss,
                                entry_price * (1.0 - self._SPOT_STOP_MIN_DISTANCE))
            else:                # cubrir short: trigger por ENCIMA del mercado
                stop_loss = max(stop_loss,
                                entry_price * (1.0 + self._SPOT_STOP_MIN_DISTANCE))

        # Límite del SL: algo peor que el (posiblemente clampado) trigger.
        if side == "sell":
            sl_limit = stop_loss * (1.0 - self._SPOT_STOP_LIMIT_BUFFER)
        else:
            sl_limit = stop_loss * (1.0 + self._SPOT_STOP_LIMIT_BUFFER)

        # Cuantización OBLIGATORIA contra PRICE_FILTER (-1013):
        #   - El buffer 0.999/1.001 introduce decimales arbitrarios sobre el
        #     stop ya redondeado por risk_manager.
        #   - tickSize en Binance Spot Testnet es típicamente 0.01 USDT para
        #     BTC/ETH; pero para otros pares puede ser 0.0001 o menor.
        #   - price_to_precision trunca al múltiplo válido del tickSize.
        sl_q = self._quantize_price(native, stop_loss)
        tp_q = self._quantize_price(native, take_profit)
        sl_limit_q = self._quantize_price(native, sl_limit)

        try:
            # OCO unificado de ccxt para Binance spot: la orden 'limit' es la
            # pierna TP (price=take_profit) y stopLossPrice arma la pierna SL.
            # stopPrice/stopLimitPrice son alias que Binance acepta para el
            # disparo y el límite del leg de stop respectivamente.
            await self._exchange.create_order(
                native, "limit", side, size_base, tp_q,
                {
                    "stopLossPrice": sl_q,        # trigger del leg SL (OCO)
                    "stopPrice": sl_q,            # alias compat Binance
                    "stopLimitPrice": sl_limit_q, # límite del leg SL
                },
            )
        except Exception as oco_err:
            logger.warning(
                "OCO spot no disponible (%s). Fallback a STOP_LOSS_LIMIT + LIMIT.",
                oco_err)
            # SL: STOP_LOSS_LIMIT — trigger=stopPrice, límite=sl_limit.
            # Sin reduceOnly: spot no lo soporta.
            await self._exchange.create_order(
                native, "STOP_LOSS_LIMIT", side, size_base, sl_limit_q,
                {"stopPrice": sl_q},
            )
            # TP: LIMIT simple, también sin reduceOnly.
            await self._exchange.create_order(
                native, "limit", side, size_base, tp_q,
            )

    async def cancel_order(self, asset_string: str,
                           broker_order_id: str) -> bool:
        native = self._native_symbol(asset_string)
        try:
            await self._exchange.cancel_order(broker_order_id, native)
            return True
        except Exception as e:
            logger.warning("cancel_order falló: %s", e)
            return False

    async def fetch_balance(self) -> BalanceData:
        bal = await self._exchange.fetch_balance()
        total = bal.get("total", {}).get(self.quote_currency, 0.0)
        free = bal.get("free", {}).get(self.quote_currency, 0.0)
        return BalanceData(total_quote=float(total),
                            available_quote=float(free),
                            quote_currency=self.quote_currency)

    async def fetch_open_orders(self, asset_string: str) -> Optional[list[dict]]:
        native = self._native_symbol(asset_string)
        try:
            return await self._exchange.fetch_open_orders(native)
        except Exception as e:
            # Fallo de red/exchange: estado desconocido. Devolvemos None para
            # que el reconciliador NO cierre el trade por error.
            logger.warning("Binance fetch_open_orders %s falló: %s",
                            asset_string, e)
            return None

    async def fetch_recent_fill_price(self, asset_string: str,
                                       since_ms: Optional[int] = None,
                                       after_timestamp_ms: Optional[int] = None,
                                       expected_side: Optional[str] = None
                                       ) -> Optional[float]:
        native = self._native_symbol(asset_string)
        try:
            trades = await self._exchange.fetch_my_trades(
                native, since=since_ms, limit=50)
            if not trades:
                return None
            # Filtrado para no confundir el fill de ENTRADA con el de SALIDA:
            #   - after_timestamp_ms: descarta fills anteriores (p.ej. la entrada).
            #   - expected_side: el cierre es el lado OPUESTO al de entrada.
            candidates = trades
            if after_timestamp_ms is not None:
                candidates = [t for t in candidates
                              if (t.get("timestamp") or 0) >= after_timestamp_ms]
            if expected_side is not None:
                want = expected_side.lower()
                candidates = [t for t in candidates
                              if str(t.get("side", "")).lower() == want]
            if not candidates:
                return None
            last = candidates[-1]        # el más reciente que cumple el filtro
            px = last.get("price") or last.get("average")
            return float(px) if px else None
        except Exception as e:
            logger.warning("Binance fetch_recent_fill_price %s falló: %s",
                            asset_string, e)
            return None

    @staticmethod
    def _native_symbol(asset_string: str) -> str:
        if ":" in asset_string:
            return asset_string.split(":", 1)[1]
        return asset_string


# =============================================================================
# AlpacaBrokerAdapter
# =============================================================================

class AlpacaBrokerAdapter(BaseBrokerAdapter):
    """
    Alpaca adapter. Usa alpaca-py si está disponible; fallback a aiohttp+SSE.

    NOTAS:
        - Alpaca soporta paper trading vía base_url separada.
        - Bracket orders son nativas en Alpaca (entry + SL + TP en una sola call).
        - SDK alpaca-py >= 0.30 con TradingClient.submit_order_async no es
          completamente estable; uso una mezcla de async wrappers sobre el
          cliente sync para máxima portabilidad.
    """

    # Key Redis para persistir el último balance conocido. TTL largo (24h)
    # para que sobreviva a reinicios cortos del contenedor sin reseteo a 0.
    _BALANCE_CACHE_REDIS_KEY = "broker:alpaca:last_balance"
    _BALANCE_CACHE_REDIS_TTL = 60 * 60 * 24  # 24h

    def __init__(self, api_key: str, api_secret: str,
                 paper_trading: bool = True,
                 quote_currency: str = "USD",
                 base_url: Optional[str] = None,
                 data_url: Optional[str] = None,
                 max_reconnect_backoff: float = 30.0,
                 redis_client: Optional[Any] = None):
        super().__init__(quote_currency=quote_currency)
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper_trading = paper_trading
        self.base_url = base_url or (
            "https://paper-api.alpaca.markets" if paper_trading
            else "https://api.alpaca.markets"
        )
        self.data_url = data_url or "wss://stream.data.alpaca.markets/v2/iex"
        self.max_reconnect_backoff = max_reconnect_backoff
        self._trading_client = None
        self._stream_client = None
        self._ws_task: Optional[asyncio.Task] = None
        self._subscribed: set[str] = set()

        # ---- Balance cache: fallback atómico ante fallos transitorios ----
        # Memoria es el primer nivel (sin latencia); Redis es persistencia
        # cross-restart si el cliente fue inyectado.
        self._redis = redis_client
        self._balance_cache: Optional[BalanceData] = None
        self._balance_cache_ts: float = 0.0
        self._balance_cache_lock = asyncio.Lock()

    async def connect(self) -> None:
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.live.stock import StockDataStream
        except ImportError as e:
            raise RuntimeError(
                "Falta paquete 'alpaca-py'. pip install alpaca-py>=0.30.0"
            ) from e

        # Trading client (sync, lo envolvemos en run_in_executor)
        self._trading_client = TradingClient(
            api_key=self.api_key, secret_key=self.api_secret,
            paper=self.paper_trading, url_override=self.base_url,
        )
        # Stream client (async nativo de alpaca-py)
        self._stream_client = StockDataStream(
            api_key=self.api_key, secret_key=self.api_secret,
            url_override=self.data_url,
        )
        logger.info("Alpaca connected (paper=%s)", self.paper_trading)

    async def close(self) -> None:
        self._stop_event.set()
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._stream_client:
            try:
                await self._stream_client.close()
            except Exception:
                pass

    async def subscribe_ticker(self, asset_string: str,
                                 callback: TickerCallback) -> None:
        self._ticker_callbacks[asset_string] = callback
        native = self._native_symbol(asset_string)
        self._subscribed.add(native)

        async def _on_quote(quote):
            cb = self._ticker_callbacks.get(asset_string)
            if cb is None:
                return
            ticker = TickerData(
                asset_string=asset_string,
                bid=float(quote.bid_price), ask=float(quote.ask_price),
                last=0.5 * (float(quote.bid_price) + float(quote.ask_price)),
                timestamp_ns=int(quote.timestamp.timestamp() * 1e9)
                              if hasattr(quote.timestamp, "timestamp")
                              else time.time_ns(),
                bid_size=float(getattr(quote, "bid_size", 0)),
                ask_size=float(getattr(quote, "ask_size", 0)),
            )
            await cb(ticker)

        self._stream_client.subscribe_quotes(_on_quote, native)

        # Arrancar el WS task si aún no está corriendo
        if self._ws_task is None or self._ws_task.done():
            self._ws_task = asyncio.create_task(
                self._run_ws_with_reconnect(), name="alpaca:ws")

    async def unsubscribe_ticker(self, asset_string: str) -> None:
        native = self._native_symbol(asset_string)
        self._subscribed.discard(native)
        try:
            self._stream_client.unsubscribe_quotes(native)
        except Exception as e:
            logger.warning("Alpaca unsubscribe falló: %s", e)
        self._ticker_callbacks.pop(asset_string, None)

    async def _run_ws_with_reconnect(self):
        """
        alpaca-py maneja reconnect internamente; nosotros añadimos un wrapper
        que sobrevive a fallos de inicialización del WS y a credenciales
        rotas. AuthenticationError es terminal: salimos del loop en lugar de
        backoffear contra un endpoint que NUNCA aceptará la conexión.

        `_run_forever` es la API async interna del StockDataStream. Si una
        versión futura de alpaca-py renombra el método, fallback explícito a
        `run()` (síncrono) ejecutado en un executor para no bloquear el loop.
        """
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                runner = getattr(self._stream_client, "_run_forever", None)
                if runner is not None:
                    await runner()
                else:
                    # Fallback portable para versiones que retiren la API privada.
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self._stream_client.run)
                backoff = 1.0
            except asyncio.CancelledError:
                break
            except Exception as e:
                exc_name = type(e).__name__
                # 401/403 vía SDK suelen surgir como AuthenticationError /
                # APIError con status_code conocido. No tiene sentido reintentar.
                status = getattr(e, "status_code", None)
                if exc_name in ("AuthenticationError", "Forbidden") or status in (401, 403):
                    logger.error(
                        "Alpaca WS TERMINAL (%s, status=%s): %s. Abortando "
                        "reconnect; revisar API keys y permisos del stream.",
                        exc_name, status, e)
                    return
                logger.error("Alpaca WS error: %s. Backoff %.1fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.max_reconnect_backoff)

    async def submit_market_order(self, asset_string: str, side: str,
                                    size_base: float,
                                    client_order_id: str) -> OrderResult:
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
        except ImportError as e:
            return OrderResult(success=False, broker_order_id=None,
                                client_order_id=client_order_id,
                                error=f"alpaca-py missing: {e}")

        native = self._native_symbol(asset_string)
        side_enum = OrderSide.BUY if side == "buy" else OrderSide.SELL

        # Mismo guard NaN/Inf que en submit_bracket_orders: un upstream
        # corrupto (Kalman degenerado, division por cero) puede llegar con
        # NaN y alpaca-py serializa "qty: NaN" que el endpoint rechaza con
        # un 422 críptico. Abortamos limpio.
        try:
            qty_f = float(size_base)
        except (TypeError, ValueError) as e:
            return OrderResult(success=False, broker_order_id=None,
                                client_order_id=client_order_id,
                                error=f"qty no numérica: {e}")
        if not math.isfinite(qty_f) or qty_f <= 0:
            return OrderResult(success=False, broker_order_id=None,
                                client_order_id=client_order_id,
                                error=f"qty no finita o ≤0: {qty_f}")

        req = MarketOrderRequest(
            symbol=native,
            qty=qty_f,                      # Python float puro (no np.float64)
            side=side_enum,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        loop = asyncio.get_event_loop()
        try:
            order = await loop.run_in_executor(
                None, self._trading_client.submit_order, req)
            return OrderResult(
                success=True, broker_order_id=str(order.id),
                client_order_id=client_order_id,
            )
        except Exception as e:
            # Idéntico tratamiento del 422 al de submit_bracket_orders: el
            # APIError trae el detalle en .message; texto plano para Telegram.
            err_msg = getattr(e, "message", None) or str(e)
            logger.exception("Alpaca submit_market_order falló: %s", err_msg)
            return OrderResult(success=False, broker_order_id=None,
                                client_order_id=client_order_id, error=err_msg)

    async def submit_bracket_orders(self, asset_string: str, side: str,
                                      size_base: float, stop_loss: float,
                                      take_profit: float,
                                      client_order_id: str,
                                      base_price: Optional[float] = None
                                      ) -> OrderResult:
        """
        Bracket order nativa para Alpaca — ambos lados usan OrderClass.BRACKET.

        Validación de brackets en Alpaca (CONSCIENTE del lado):
          - LONG  (parent BUY):  take_profit.limit_price >= base + 0.01 y
            stop_loss.stop_price <= base - 0.01   → limit_price > stop_price.
          - SHORT (parent SELL): take_profit.limit_price <= base - 0.01 y
            stop_loss.stop_price >= base + 0.01   → limit_price < stop_price.
          En ambos lados asignamos cada precio a su campo conceptual real
          (TP→limit_price, SL→stop_price); el guardrail dimensional y los clamps
          de abajo garantizan el orden numérico que cada lado exige.

          NOTA: una versión previa invertía los slots en el short (sl_px→TP,
          tp_px→SL) creyendo que el validador era agnóstico al lado. Era falso:
          esa inversión rompía la validación cruzada del SELL y causaba el 422.
          Eliminada — el mapeo es directo en ambos lados.

        Ventaja sobre el bypass anterior (órdenes independientes):
          Al ser una orden atómica, Alpaca no dispara la detección de wash
          trade (403 "opposite side market/stop order exists").
        """
        try:
            from alpaca.trading.requests import (
                MarketOrderRequest, TakeProfitRequest, StopLossRequest,
            )
            from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
        except ImportError as e:
            return OrderResult(success=False, broker_order_id=None,
                                client_order_id=client_order_id,
                                error=f"alpaca-py missing: {e}")

        native = self._native_symbol(asset_string)
        is_short = side == "sell"

        # Guard NaN/Inf: si la estrategia/risk manager pasa un float no finito
        # (nan/inf) por un bug aguas arriba, round(.., 2) lo propaga, max() es
        # no determinista contra nan y se acaba enviando un payload corrupto a
        # Alpaca → 422 críptico. Abortamos LIMPIO antes de tocar precios.
        for _label, _val in (("take_profit", take_profit),
                              ("stop_loss", stop_loss),
                              ("base_price", base_price)):
            if _val is not None and not math.isfinite(float(_val)):
                logger.warning(
                    "[%s] %s no finito (%s); orden descartada antes del envío.",
                    native, _label, _val)
                return OrderResult(
                    success=False, broker_order_id=None,
                    client_order_id=client_order_id,
                    error=f"{_label} not finite: {_val}")

        # Clamp y redondeo a 2 decimales (tick mínimo equity Alpaca).
        tp_px = round(max(float(take_profit), 0.01), 2)
        sl_px = round(max(float(stop_loss), 0.01), 2)

        # Guardrail dimensional: SHORT → SL > TP; LONG → TP > SL.
        # Si los precios llegan invertidos por bug aguas arriba, los corregimos.
        if is_short and tp_px >= sl_px:
            logger.warning(
                "[%s] SHORT con TP>=SL (tp=%.2f sl=%.2f). Swap defensivo.",
                asset_string, tp_px, sl_px)
            tp_px, sl_px = min(tp_px, sl_px), max(tp_px, sl_px)
        if not is_short and tp_px <= sl_px:
            logger.warning(
                "[%s] LONG con TP<=SL (tp=%.2f sl=%.2f). Swap defensivo.",
                asset_string, tp_px, sl_px)
            tp_px, sl_px = max(tp_px, sl_px), min(tp_px, sl_px)

        # ===== Lock-in defensivo de precios contra el 422 de microestructura =====
        # El validador de Alpaca exige, por lado:
        #   LONG  (parent BUY):  tp.limit_price >= base + 0.01  y  sl.stop_price <= base - 0.01
        #   SHORT (parent SELL): tp.limit_price <= base - 0.01  y  sl.stop_price >= base + 0.01
        # El umbral de 0.01 USD se cruza trivialmente con el slippage de
        # tránsito en activos de alto valor nominal (NVDA, SMH) durante la
        # apertura de NY: el server_base se desplaza varios USD antes del fill
        # y un TP o SL pegado a la entrada queda inválido → 422. Hasta ahora
        # solo blindábamos el TP; el SL quedaba expuesto al mismo gap.
        #
        # Aplicamos un buffer SIMÉTRICO de 4.00 USD a AMBAS piernas y AMBOS
        # lados, anclado en base_price. Eso garantiza que limit_price y
        # stop_price queden siempre del lado correcto de server_base ± 0.01
        # incluso ante rallies o caídas de varios USD entre la señal y el fill.
        # Piso defensivo 0.01 en las piernas "hacia abajo" (sl_long, tp_short)
        # para no generar precios <= 0 en activos baratos; si el clamp toca el
        # piso (buffer no cabe en el rango de precio), lo registramos como
        # WARNING aparte: el bracket pasa el validador pero queda económicamente
        # nominal y la operación opera ese síntoma como mal calibrado para el
        # subyacente.
        if base_price is not None:
            _BRACKET_BUFFER_USD = 4.00
            base_f = float(base_price)
            if not is_short:
                # LONG: tp >= base + 4.00 ; sl <= base - 4.00 (piso 0.01).
                tp_floor = round(base_f + _BRACKET_BUFFER_USD, 2)
                sl_ceiling = round(max(base_f - _BRACKET_BUFFER_USD, 0.01), 2)
                if tp_px < tp_floor:
                    logger.warning(
                        "[%s] BUY TP=%.2f bajo base+%.2f=%.2f (base=%.2f). Clamp "
                        "a %.2f (buffer Alpaca = %.2f USD).",
                        native, tp_px, _BRACKET_BUFFER_USD, tp_floor, base_f,
                        tp_floor, _BRACKET_BUFFER_USD)
                    tp_px = tp_floor
                if sl_px > sl_ceiling:
                    logger.warning(
                        "[%s] BUY SL=%.2f sobre base-%.2f=%.2f (base=%.2f). Clamp "
                        "a %.2f (buffer Alpaca = %.2f USD).",
                        native, sl_px, _BRACKET_BUFFER_USD, sl_ceiling, base_f,
                        sl_ceiling, _BRACKET_BUFFER_USD)
                    sl_px = sl_ceiling
                if sl_ceiling <= 0.01:
                    logger.warning(
                        "[%s] BUY: SL clampado al PISO 0.01 — buffer %.2f USD no "
                        "cabe en base=%.2f; bracket queda con stop nominal.",
                        native, _BRACKET_BUFFER_USD, base_f)
            else:
                # SHORT: tp <= base - 4.00 (piso 0.01) ; sl >= base + 4.00.
                # Mapeo directo (sin inversión) → ver req de abajo.
                tp_ceiling = round(max(base_f - _BRACKET_BUFFER_USD, 0.01), 2)
                sl_floor = round(base_f + _BRACKET_BUFFER_USD, 2)
                if tp_px > tp_ceiling:
                    logger.warning(
                        "[%s] SELL TP=%.2f sobre base-%.2f=%.2f (base=%.2f). Clamp "
                        "a %.2f (buffer Alpaca = %.2f USD).",
                        native, tp_px, _BRACKET_BUFFER_USD, tp_ceiling, base_f,
                        tp_ceiling, _BRACKET_BUFFER_USD)
                    tp_px = tp_ceiling
                if sl_px < sl_floor:
                    logger.warning(
                        "[%s] SELL SL=%.2f bajo base+%.2f=%.2f (base=%.2f). Clamp "
                        "a %.2f (buffer Alpaca = %.2f USD).",
                        native, sl_px, _BRACKET_BUFFER_USD, sl_floor, base_f,
                        sl_floor, _BRACKET_BUFFER_USD)
                    sl_px = sl_floor
                if tp_ceiling <= 0.01:
                    logger.warning(
                        "[%s] SELL: TP clampado al PISO 0.01 — buffer %.2f USD no "
                        "cabe en base=%.2f; bracket queda con TP nominal.",
                        native, _BRACKET_BUFFER_USD, base_f)

        loop = asyncio.get_event_loop()

        if not is_short:
            # LONG: TP > SL; asignación directa, validador satisfecho.
            # Trunca a acciones enteras: Alpaca NO admite fracciones en órdenes
            # bracket (order_class=BRACKET), igual que en el short. Sin esto, un
            # size_base con decimales rebota en el validador con 422.
            long_qty = int(math.floor(size_base))
            if long_qty != size_base:
                logger.info(
                    "[Alpaca] Qty truncada a entero para cumplir con las "
                    "restricciones de brackets fraccionarios.")
            if long_qty <= 0:
                logger.warning(
                    "[%s] LONG con qty truncada a %d (size_base=%s); "
                    "orden descartada para no enviar tallas vacías.",
                    native, long_qty, size_base)
                return OrderResult(
                    success=False, broker_order_id=None,
                    client_order_id=client_order_id,
                    error=f"long qty truncated to {long_qty} "
                          f"(size_base={size_base}); skipping empty/fractional long")
            req = MarketOrderRequest(
                symbol=native,
                qty=long_qty,
                side=OrderSide.BUY,
                # DAY (no GTC): consistente con el short y con la regla de Alpaca
                # de fraccionarias; el bracket viaja ya con qty entera tras el
                # floor, evitando el 422 de 'fractional orders must be DAY'.
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=tp_px),
                stop_loss=StopLossRequest(stop_price=sl_px),
                client_order_id=client_order_id,
            )
        else:
            # SHORT: Alpaca rechaza qty fraccionaria en cortos.
            short_qty = int(math.floor(size_base))
            if short_qty <= 0:
                logger.warning(
                    "[%s] SHORT con qty truncada a %d (size_base=%s); "
                    "orden descartada para no enviar tallas vacías.",
                    native, short_qty, size_base)
                return OrderResult(
                    success=False, broker_order_id=None,
                    client_order_id=client_order_id,
                    error=f"short qty truncated to {short_qty} "
                          f"(size_base={size_base}); skipping empty/fractional short")

            # Mapeo conceptual directo (SIN inversión). El validador de Alpaca
            # es consciente del lado: para un parent SELL exige
            #   take_profit.limit_price <= base - 0.01  (salida con ganancia, por
            #     DEBAJO de la entrada)  → aquí tp_px
            #   stop_loss.stop_price    >= base + 0.01  (protección, por ENCIMA)
            #     → aquí sl_px
            # es decir limit_price(tp_px) < stop_price(sl_px), exactamente el
            # orden numérico que asegura el guardrail dimensional de arriba. La
            # "inversión defensiva" previa (sl_px→TP, tp_px→SL) asumía un
            # validador agnóstico al lado: rompía esta validación cruzada tras
            # los redondeos y disparaba el 422 en los shorts.
            req = MarketOrderRequest(
                symbol=native,
                qty=short_qty,
                side=OrderSide.SELL,
                # DAY (no GTC) por consistencia con el long y la regla de
                # fraccionarias de Alpaca. El short ya es qty entera (floor),
                # pero DAY es válido para brackets y evita divergencia de TIF.
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=tp_px),
                stop_loss=StopLossRequest(stop_price=sl_px),
                client_order_id=client_order_id,
            )

        try:
            order = await loop.run_in_executor(
                None, self._trading_client.submit_order, req)
            return OrderResult(
                success=True, broker_order_id=str(order.id),
                client_order_id=client_order_id,
            )
        except Exception as e:
            # El APIError de Alpaca trae el texto crudo del 422 en `.message`
            # ('take_profit.limit_price must be >= base_price + 0.01'); si el
            # atributo no existe (otra excepción), caemos a str(e). El mensaje
            # contiene llaves/puntos/comparadores: bot_core lo despacha a
            # Telegram en texto plano (parse_mode=None) para no romper el parser.
            err_msg = getattr(e, "message", None) or str(e)
            logger.exception(
                "Alpaca bracket order falló (%s %s qty=%s tp=%.2f sl=%.2f): %s",
                side, native, size_base, tp_px, sl_px, err_msg)
            return OrderResult(success=False, broker_order_id=None,
                                client_order_id=client_order_id, error=err_msg)

    async def cancel_order(self, asset_string: str,
                           broker_order_id: str) -> bool:
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None, self._trading_client.cancel_order_by_id, broker_order_id)
            return True
        except Exception as e:
            logger.warning("Alpaca cancel falló: %s", e)
            return False

    async def _persist_balance_cache(self, bal: BalanceData) -> None:
        """Guarda el balance en memoria y, si hay Redis, también ahí."""
        async with self._balance_cache_lock:
            self._balance_cache = bal
            self._balance_cache_ts = time.time()
        if self._redis is not None:
            try:
                payload = json.dumps({
                    **asdict(bal),
                    "cached_at": time.time(),
                })
                await self._redis.set(
                    self._BALANCE_CACHE_REDIS_KEY, payload,
                    ex=self._BALANCE_CACHE_REDIS_TTL,
                )
            except Exception as e:
                # Persistir el cache nunca debe romper la ejecución.
                logger.debug("No se pudo persistir balance en Redis: %s", e)

    async def _load_balance_cache(self) -> Optional[BalanceData]:
        """Devuelve el último balance conocido (memoria → Redis)."""
        async with self._balance_cache_lock:
            if self._balance_cache is not None:
                return self._balance_cache
        if self._redis is not None:
            try:
                raw = await self._redis.get(self._BALANCE_CACHE_REDIS_KEY)
                if raw:
                    data = json.loads(raw)
                    bal = BalanceData(
                        total_quote=float(data["total_quote"]),
                        available_quote=float(data["available_quote"]),
                        quote_currency=data.get(
                            "quote_currency", self.quote_currency),
                    )
                    async with self._balance_cache_lock:
                        self._balance_cache = bal
                        self._balance_cache_ts = float(
                            data.get("cached_at", time.time()))
                    return bal
            except Exception as e:
                logger.debug("No se pudo leer balance cacheado de Redis: %s", e)
        return None

    async def fetch_balance(self) -> BalanceData:
        """
        Devuelve el balance de la cuenta Alpaca. Ante fallos transitorios
        (red, 5xx, timeouts) devuelve el ÚLTIMO BALANCE CONOCIDO en cache —
        nunca 0.0 — para evitar que el RiskManager interprete una
        desconexión como pérdida total del capital y dispare un falso
        bloqueo por drawdown diario del 100%.
        """
        loop = asyncio.get_event_loop()
        try:
            account = await loop.run_in_executor(
                None, self._trading_client.get_account)
            equity = float(account.equity)
            cash = float(account.cash)
            bal = BalanceData(total_quote=equity, available_quote=cash,
                              quote_currency=self.quote_currency)
            await self._persist_balance_cache(bal)
            return bal
        except Exception as e:
            transient = _is_transient_balance_error(e)
            cached = await self._load_balance_cache()
            if cached is not None:
                age = time.time() - self._balance_cache_ts
                logger.warning(
                    "Alpaca fetch_balance falló (%s: %s). Devolviendo balance "
                    "cacheado (age=%.0fs, equity=%.2f %s). NO se reporta 0 "
                    "para no contaminar el RiskManager.",
                    type(e).__name__, e, age, cached.total_quote,
                    cached.quote_currency,
                )
                return cached
            if transient:
                # Sin cache previo y fallo transitorio: devolvemos NaN-safe
                # marker (equity negativa) que el llamador debe interpretar
                # como "indeterminado" — NUNCA 0.0 para no triggear DD.
                logger.error(
                    "Alpaca fetch_balance falló transitoriamente y NO hay "
                    "cache previa; devolviendo balance indeterminado para "
                    "no bloquear por falso drawdown: %s", e)
                return BalanceData(
                    total_quote=float("nan"),
                    available_quote=float("nan"),
                    quote_currency=self.quote_currency,
                )
            # Error no transitorio (credenciales, cuenta cerrada, etc.):
            # propagamos como BalanceData con 0.0 sólo si NO es transitorio
            # y NO hay cache. Esto preserva la semántica anterior para
            # fallos genuinos de cuenta.
            logger.exception(
                "Alpaca fetch_balance falló con error NO transitorio y sin "
                "cache previa: %s", e)
            return BalanceData(total_quote=0.0, available_quote=0.0,
                                quote_currency=self.quote_currency)

    @staticmethod
    def _native_symbol(asset_string: str) -> str:
        if ":" in asset_string:
            return asset_string.split(":", 1)[1]
        return asset_string


# =============================================================================
# Factory helper
# =============================================================================

def create_broker_adapter(asset_class: str, config: dict) -> BaseBrokerAdapter:
    """
    Factory: dado el asset_class string y config, devuelve el adapter correcto.

    config debe contener al menos:
        - api_key, api_secret
        - paper_trading (bool)
        - quote_currency (opcional)
        - redis_client (opcional, sólo equity): habilita cache persistente
          de balance ante fallos transitorios.
    """
    asset_class = asset_class.lower()
    if asset_class == "crypto":
        return BinanceBrokerAdapter(
            api_key=config["api_key"],
            api_secret=config["api_secret"],
            paper_trading=config.get("paper_trading", True),
            quote_currency=config.get("quote_currency", "USDT"),
        )
    elif asset_class == "equity":
        return AlpacaBrokerAdapter(
            api_key=config["api_key"],
            api_secret=config["api_secret"],
            paper_trading=config.get("paper_trading", True),
            quote_currency=config.get("quote_currency", "USD"),
            redis_client=config.get("redis_client"),
        )
    else:
        raise ValueError(f"Unknown asset_class: {asset_class!r}")
