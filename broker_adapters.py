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
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


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
                                      client_order_id: str) -> OrderResult:
        """
        Envia entry + SL + TP. Cada broker usa su mejor primitiva:
            - Binance: order de entrada market, luego OCO con SL/TP.
            - Alpaca: bracket order nativa.
        """

    @abc.abstractmethod
    async def cancel_order(self, asset_string: str,
                           broker_order_id: str) -> bool: ...

    @abc.abstractmethod
    async def fetch_balance(self) -> BalanceData: ...

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
        """WS loop con reconnect exponencial."""
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
                logger.error("Binance WS error %s: %s. Backoff %.1fs",
                              asset_string, e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.max_reconnect_backoff)

    async def submit_market_order(self, asset_string: str, side: str,
                                    size_base: float,
                                    client_order_id: str) -> OrderResult:
        native = self._native_symbol(asset_string)
        try:
            order = await self._exchange.create_order(
                native, "market", side, size_base, None,
                {"clientOrderId": client_order_id},
            )
            return OrderResult(
                success=True, broker_order_id=order.get("id"),
                client_order_id=client_order_id,
                fills=order.get("trades", []),
            )
        except Exception as e:
            logger.exception("Binance submit_market_order falló: %s", e)
            return OrderResult(success=False, broker_order_id=None,
                               client_order_id=client_order_id, error=str(e))

    async def submit_bracket_orders(self, asset_string: str, side: str,
                                      size_base: float, stop_loss: float,
                                      take_profit: float,
                                      client_order_id: str) -> OrderResult:
        """
        Binance NO tiene bracket nativa para spot. Estrategia:
            1. Market entry.
            2. OCO con SL/TP en lado opuesto.
        Si OCO falla, cerramos posición.
        """
        entry_result = await self.submit_market_order(
            asset_string, side, size_base, client_order_id)
        if not entry_result.success:
            return entry_result

        native = self._native_symbol(asset_string)
        opposite = "sell" if side == "buy" else "buy"
        try:
            # Algunos exchanges spot soportan OCO via create_order tipo oco
            # En ccxt para Binance spot: create_order(..., 'oco', ...)
            # Si tu cuenta/símbolo no soporta OCO, usar dos órdenes separadas.
            await self._exchange.create_order(
                native, "stop_market", opposite, size_base, None,
                {"stopPrice": stop_loss, "reduceOnly": True},
            )
            await self._exchange.create_order(
                native, "limit", opposite, size_base, take_profit,
                {"reduceOnly": True},
            )
            return entry_result
        except Exception as e:
            logger.error("Binance SL/TP falló: %s. Cerrando posición.", e)
            try:
                await self._exchange.create_order(
                    native, "market", opposite, size_base)
            except Exception as close_err:
                logger.exception("CRITICAL: no se pudo cerrar tras SL/TP fail: %s",
                                  close_err)
            entry_result.error = f"SL/TP failed and position closed: {e}"
            entry_result.success = False
            return entry_result

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

    def __init__(self, api_key: str, api_secret: str,
                 paper_trading: bool = True,
                 quote_currency: str = "USD",
                 base_url: Optional[str] = None,
                 data_url: Optional[str] = None,
                 max_reconnect_backoff: float = 30.0):
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
        """alpaca-py maneja reconnect internamente, pero lo monitoreamos."""
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                await self._stream_client._run_forever()
                backoff = 1.0
            except asyncio.CancelledError:
                break
            except Exception as e:
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
        req = MarketOrderRequest(
            symbol=native,
            qty=size_base,
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
            logger.exception("Alpaca submit_market_order falló: %s", e)
            return OrderResult(success=False, broker_order_id=None,
                                client_order_id=client_order_id, error=str(e))

    async def submit_bracket_orders(self, asset_string: str, side: str,
                                      size_base: float, stop_loss: float,
                                      take_profit: float,
                                      client_order_id: str) -> OrderResult:
        """
        Alpaca soporta bracket nativa via order_class='bracket'.
        Mucho más limpio que Binance.
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
        side_enum = OrderSide.BUY if side == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=native,
            qty=size_base,
            side=side_enum,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)),
            stop_loss=StopLossRequest(stop_price=round(stop_loss, 2)),
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
            logger.exception("Alpaca bracket order falló: %s", e)
            return OrderResult(success=False, broker_order_id=None,
                                client_order_id=client_order_id, error=str(e))

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

    async def fetch_balance(self) -> BalanceData:
        loop = asyncio.get_event_loop()
        try:
            account = await loop.run_in_executor(
                None, self._trading_client.get_account)
            equity = float(account.equity)
            cash = float(account.cash)
            return BalanceData(total_quote=equity, available_quote=cash,
                                quote_currency=self.quote_currency)
        except Exception as e:
            logger.exception("Alpaca fetch_balance falló: %s", e)
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
        )
    else:
        raise ValueError(f"Unknown asset_class: {asset_class!r}")
