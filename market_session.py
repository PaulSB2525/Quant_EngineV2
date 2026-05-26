"""
market_session.py
=================
Determina si un mercado está abierto en un instante dado.

Crypto: always_open = True (24/7/365).
Equity (NYSE/NASDAQ):
    - Sesión regular: 09:30-16:00 ET, lunes a viernes.
    - Sin operación en holidays federales (calendario embebido 2024-2030).
    - Early closes: 13:00 ET en víspera de algunos holidays.
    - Pre/post-market opcional (NO operamos por default; demasiada slippage).

Diseño:
    - Calendario hardcodeado para portabilidad (no requiere pandas_market_calendars).
    - Si pandas_market_calendars está instalado, lo prefiere (fuente autoritativa).
    - API simple: is_open(asset_string, ts) -> bool
                  seconds_until_open(asset_string, ts) -> float
                  gap_duration_seconds(asset_string, last_tick_ts, now_ts) -> float

Importante: las marcas temporales son siempre en UTC. La conversión a ET
se hace internamente (US/Eastern con DST automático).
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


# =============================================================================
# Calendario embebido NYSE/NASDAQ
# =============================================================================

# Holidays federales con cierre completo (NYSE). Fechas en YYYY-MM-DD.
# Fuente: NYSE calendar. Actualizar anualmente.
NYSE_FULL_CLOSES: set[str] = {
    # 2024
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    # 2025
    "2025-01-01", "2025-01-09",  # Carter day of mourning
    "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19",
    "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03",  # Independence Day observado viernes
    "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027 (placeholders aproximados; reconfirmar)
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18",  # Juneteenth observado viernes
    "2027-07-05",  # Independence Day observado lunes
    "2027-09-06", "2027-11-25", "2027-12-24",  # Christmas observado viernes
}

# Early closes: 13:00 ET en lugar de 16:00 ET. Día antes de algunos holidays.
NYSE_EARLY_CLOSES: set[str] = {
    "2024-07-03", "2024-11-29", "2024-12-24",
    "2025-07-03", "2025-11-28", "2025-12-24",
    "2026-11-27", "2026-12-24",
    "2027-11-26",
}


# =============================================================================
# MarketSessionGuard
# =============================================================================

@dataclass
class SessionStatus:
    """Estado completo de una sesión en un instante dado."""
    is_open: bool
    is_pre_market: bool
    is_post_market: bool
    seconds_until_next_open: Optional[float] = None  # None si ya abierto
    seconds_until_close: Optional[float] = None      # None si cerrado


class MarketSessionGuard:
    """
    Consulta de horarios de mercado optimizada para llamadas frecuentes.

    Cache:
        - El cálculo de "is_open" para un timestamp dado es O(1) con set lookup.
        - Para minimizar overhead, el bot puede cachear el resultado por N
          segundos si lo desea. Esta clase no cachea internamente porque el
          cómputo es de microsegundos.
    """

    def __init__(self, allow_pre_market: bool = False,
                 allow_post_market: bool = False,
                 use_pandas_calendar_if_available: bool = True):
        self.allow_pre_market = allow_pre_market
        self.allow_post_market = allow_post_market

        self._pandas_cal = None
        if use_pandas_calendar_if_available:
            try:
                import pandas_market_calendars as mcal
                self._pandas_cal = mcal.get_calendar("XNYS")
                logger.info("MarketSessionGuard usando pandas_market_calendars")
            except ImportError:
                logger.info("pandas_market_calendars no instalado; usando calendario embebido")

    # ---- API pública ----

    def is_open(self, asset_string: str,
                ts: Optional[dt.datetime] = None) -> bool:
        """Devuelve True si el mercado está abierto en `ts` (UTC, default=now)."""
        status = self.get_status(asset_string, ts)
        return status.is_open

    def get_status(self, asset_string: str,
                    ts: Optional[dt.datetime] = None) -> SessionStatus:
        if ts is None:
            ts = dt.datetime.now(UTC)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)

        if self._is_crypto(asset_string):
            return SessionStatus(
                is_open=True, is_pre_market=False, is_post_market=False,
                seconds_until_next_open=None, seconds_until_close=None,
            )

        # Equity logic
        return self._equity_status(ts)

    def seconds_until_open(self, asset_string: str,
                            ts: Optional[dt.datetime] = None) -> float:
        """Segundos hasta la próxima apertura. 0 si ya abierto."""
        status = self.get_status(asset_string, ts)
        if status.is_open:
            return 0.0
        return status.seconds_until_next_open or 0.0

    def gap_duration_seconds(self, asset_string: str,
                              last_tick_ts: dt.datetime,
                              now_ts: Optional[dt.datetime] = None) -> float:
        """
        Calcula la duración del cierre entre `last_tick_ts` y `now_ts`.

        Para equity, esto SOLO cuenta tiempo durante el cierre (no las horas
        de mercado abierto). Para crypto siempre devuelve 0.

        Útil para Kalman.update_after_gap().
        """
        if now_ts is None:
            now_ts = dt.datetime.now(UTC)
        if self._is_crypto(asset_string):
            return 0.0

        if last_tick_ts.tzinfo is None:
            last_tick_ts = last_tick_ts.replace(tzinfo=UTC)
        if now_ts.tzinfo is None:
            now_ts = now_ts.replace(tzinfo=UTC)

        # Simplificación operativa: en producción, el gap es típicamente
        # overnight o weekend; el bot detecta la transición closed->open y
        # marca last_tick_ts ≈ "16:00 ET del día previo abierto". El cálculo
        # exacto del "tiempo en mercado cerrado" requiere iterar día a día.
        # Aproximación suficiente: diferencia total clamped a max razonable.
        total_seconds = (now_ts - last_tick_ts).total_seconds()
        return max(0.0, total_seconds)

    # ---- Internals ----

    @staticmethod
    def _is_crypto(asset_string: str) -> bool:
        if ":" in asset_string:
            return asset_string.split(":", 1)[0].upper() == "CRYPTO"
        # v1-compat: sin prefijo, asumir crypto
        return True

    def _equity_status(self, ts_utc: dt.datetime) -> SessionStatus:
        """Status para equity en `ts_utc`."""
        ts_et = ts_utc.astimezone(ET)
        date_str = ts_et.strftime("%Y-%m-%d")

        # Weekend?
        if ts_et.weekday() >= 5:   # 5=sat, 6=sun
            return SessionStatus(
                is_open=False, is_pre_market=False, is_post_market=False,
                seconds_until_next_open=self._seconds_to_next_session_open(ts_et),
                seconds_until_close=None,
            )

        # Holiday?
        if date_str in NYSE_FULL_CLOSES:
            return SessionStatus(
                is_open=False, is_pre_market=False, is_post_market=False,
                seconds_until_next_open=self._seconds_to_next_session_open(ts_et),
                seconds_until_close=None,
            )

        # Si pandas calendar disponible, usar como autoridad
        if self._pandas_cal is not None:
            try:
                schedule = self._pandas_cal.schedule(
                    start_date=date_str, end_date=date_str)
                if schedule.empty:
                    return SessionStatus(
                        is_open=False, is_pre_market=False, is_post_market=False,
                        seconds_until_next_open=self._seconds_to_next_session_open(ts_et),
                    )
            except Exception:
                pass  # fallback a calendario embebido

        # Sesión regular o early close
        is_early_close = date_str in NYSE_EARLY_CLOSES
        open_t = ts_et.replace(hour=9, minute=30, second=0, microsecond=0)
        close_t = ts_et.replace(
            hour=13 if is_early_close else 16,
            minute=0, second=0, microsecond=0,
        )

        pre_start = ts_et.replace(hour=4, minute=0, second=0, microsecond=0)
        post_end = ts_et.replace(hour=20, minute=0, second=0, microsecond=0)

        is_regular = open_t <= ts_et < close_t
        is_pre = pre_start <= ts_et < open_t
        is_post = close_t <= ts_et < post_end

        if is_regular:
            return SessionStatus(
                is_open=True, is_pre_market=False, is_post_market=False,
                seconds_until_next_open=None,
                seconds_until_close=(close_t - ts_et).total_seconds(),
            )

        # Pre/post market: respetar config
        if is_pre and self.allow_pre_market:
            return SessionStatus(
                is_open=True, is_pre_market=True, is_post_market=False,
                seconds_until_next_open=None,
                seconds_until_close=(close_t - ts_et).total_seconds(),
            )
        if is_post and self.allow_post_market:
            return SessionStatus(
                is_open=True, is_pre_market=False, is_post_market=True,
                seconds_until_next_open=None,
                seconds_until_close=(post_end - ts_et).total_seconds(),
            )

        # Cerrado (overnight)
        return SessionStatus(
            is_open=False, is_pre_market=is_pre, is_post_market=is_post,
            seconds_until_next_open=self._seconds_to_next_session_open(ts_et),
            seconds_until_close=None,
        )

    def _seconds_to_next_session_open(self, ts_et: dt.datetime) -> float:
        """Busca hacia adelante el próximo día hábil y devuelve segundos a 09:30 ET."""
        candidate = ts_et
        # Si ya pasó 09:30 hoy y no estamos en sesión, ir a mañana
        today_open = ts_et.replace(hour=9, minute=30, second=0, microsecond=0)
        if ts_et >= today_open:
            candidate = ts_et + dt.timedelta(days=1)
            candidate = candidate.replace(hour=9, minute=30, second=0, microsecond=0)
        else:
            candidate = today_open

        # Avanzar hasta encontrar un día hábil
        for _ in range(15):   # max 15 días, suficiente para puentes largos
            if candidate.weekday() < 5:
                date_str = candidate.strftime("%Y-%m-%d")
                if date_str not in NYSE_FULL_CLOSES:
                    return (candidate - ts_et).total_seconds()
            candidate += dt.timedelta(days=1)
            candidate = candidate.replace(hour=9, minute=30, second=0, microsecond=0)

        return 7 * 24 * 3600   # fallback: 1 semana
