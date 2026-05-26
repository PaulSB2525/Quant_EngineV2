"""
risk_manager.py  (v2)
=====================
Motor de riesgo multi-asset con namespaces diferenciados para Crypto y Equity.

Cambios v1 → v2:
    1. Asset-class split en RiskConfig:
        * CryptoRiskConfig:  vol cap 250%, TCA en bps, maker/taker fees.
        * EquityRiskConfig:  vol cap 60%, TCA flat-fee + SEC/FINRA + half-spread,
                             volume-based slippage (Almgren-Chriss simplificado).
    2. Cross-asset correlation penalty:
        * El RiskManager mantiene una matriz de correlación rolling 1h.
        * Si |ρ(NVDA, BTC)| > 0.7 → Kelly se multiplica por (1 − excess_corr).
        * Si |ρ| > 0.90 → hard cutoff (factor exposure cluster shock).
    3. Coherencia dt-spread:
        * Rechaza señales cuya OU half_life es incoherente con el horizon
          de holding implícito por las protecciones SL/TP.
    4. Pair-trading sizing:
        * evaluate_pair_trade() para señales generadas por engine_math v2
          sobre spreads cointegrados.

Lo que se mantiene de v1:
    - Day-lock vía Redis (SETNX, TTL 30h, idempotente al reinicio).
    - Kelly smoothing EMA per-symbol.
    - SL/TP basados en ATR enviados al exchange.
    - Retorno explícito de RiskDecision con verdict enumerado.
    - Nunca raise por condiciones esperadas.

Referencias adicionales:
    - Almgren, R., & Chriss, N. (2001). "Optimal Execution of Portfolio Transactions."
    - Cartea, Jaimungal, Penalva (2015). Algorithmic and High-Frequency Trading.
    - SEC Rule §31 (Section 31 Fee), FINRA TAF rates.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

import redis.asyncio as aioredis


# =============================================================================
# Asset class
# =============================================================================

class AssetClass(str, Enum):
    CRYPTO = "crypto"
    EQUITY = "equity"


def parse_asset_class(symbol: str) -> tuple[AssetClass, str]:
    """
    Parsea un símbolo uniforme y devuelve (asset_class, native_symbol).

    Convención:
        CRYPTO:BTC/USDT  → (CRYPTO, "BTC/USDT")
        EQUITY:AAPL      → (EQUITY, "AAPL")
        BTC/USDT         → (CRYPTO, "BTC/USDT")  # fallback v1-compat

    El "native_symbol" es lo que se pasa al broker adapter correspondiente.
    """
    if ":" in symbol:
        prefix, native = symbol.split(":", 1)
        prefix = prefix.upper().strip()
        if prefix == "CRYPTO":
            return AssetClass.CRYPTO, native.strip()
        if prefix == "EQUITY":
            return AssetClass.EQUITY, native.strip()
        raise ValueError(f"Prefijo desconocido en '{symbol}'. Use CRYPTO: o EQUITY:")
    # v1-compat: sin prefijo, asumir crypto
    return AssetClass.CRYPTO, symbol.strip()


# =============================================================================
# Verdicts
# =============================================================================

class RiskVerdict(str, Enum):
    APPROVED = "approved"
    REJECT_TCA = "reject_tca"
    REJECT_DRAWDOWN = "reject_drawdown"
    REJECT_KELLY_NEGATIVE = "reject_kelly_negative"
    REJECT_EXPOSURE = "reject_exposure"
    REJECT_VOLATILITY_REGIME = "reject_volatility_regime"
    REJECT_CORRELATION_CLUSTER = "reject_correlation_cluster"
    REJECT_HALF_LIFE_INCOHERENT = "reject_half_life_incoherent"
    REJECT_PAIR_NOT_OPERABLE = "reject_pair_not_operable"
    REJECT_MARKET_CLOSED = "reject_market_closed"


@dataclass(frozen=True)
class RiskDecision:
    verdict: RiskVerdict
    size_quote: float
    stop_loss_price: float
    take_profit_price: float
    reason: str
    metrics: dict = field(default_factory=dict)
    # Pair-trading: tamaños por leg, mismo signo = direcciones opuestas
    leg_a_size_quote: float = 0.0
    leg_b_size_quote: float = 0.0

    @property
    def approved(self) -> bool:
        return self.verdict == RiskVerdict.APPROVED


# =============================================================================
# Trading costs por asset class
# =============================================================================

@dataclass
class CryptoTradingCosts:
    """Costos en bps (1 bp = 0.01%). Defaults para Binance spot."""
    maker_fee_bps: float = 1.0
    taker_fee_bps: float = 5.0
    expected_spread_bps: float = 2.0
    slippage_bps: float = 3.0

    def round_trip_bps(self, use_taker: bool = False) -> float:
        fee = self.taker_fee_bps if use_taker else self.maker_fee_bps
        return 2.0 * (fee + self.expected_spread_bps + self.slippage_bps)


@dataclass
class EquityTradingCosts:
    """
    Costos para equity zero-commission (Alpaca/IBKR retail).

    Componentes (rates vigentes a 2026):
        - SEC §31 fee: sale-only. $20.60 per $1M notional efectivo desde
          2026-04-04 (SEC Fee Rate Advisory FY2026). Cero entre 2025-05-14
          y 2026-04-03 (SEC FY2025 ya recaudada). La tasa se reajusta anual.
        - FINRA TAF: sale-only. $0.000195/share efectivo desde 2026-01-01,
          cap $9.79/trade (FINRA Rule, Schedule A Section 1).
        - Half-spread: bid-ask es el costo dominante en retail.
        - Slippage Almgren-Chriss: linear impact ~ k·(volume/ADV)·σ.

    Notar: SEC y FINRA aplican SOLO en la salida (sale leg). Por simplicidad
    los aplicamos como "half-cost" en el round-trip total. Es ligeramente
    conservador (sobreestima costos en compras-y-venta).

    IMPORTANTE: Las tasas son ajustadas periódicamente. Revisar antes de
    pasar a producción:
        - https://www.sec.gov/rules-regulations/fee-rate-advisories
        - https://www.finra.org/rules-guidance/guidance/trading-activity-fee
    """
    sec_fee_per_million: float = 20.60        # USD per $1M notional (FY2026)
    finra_taf_per_share: float = 0.000195     # USD per share (Jan 2026+)
    finra_taf_cap: float = 9.79               # USD cap per trade (Jan 2026+)
    expected_half_spread_bps: float = 5.0
    almgren_linear_impact_coef: float = 0.1
    expected_volume_pct_of_adv: float = 0.001

    def round_trip_bps(self, notional_usd: float, avg_share_price: float,
                       reference_vol_annualized: float) -> float:
        """
        Devuelve el costo round-trip en bps para una operación de
        `notional_usd` de tamaño.

        Fórmula:
            half_spread × 2  (ida y vuelta)
          + SEC_fee_bps      (solo venta)
          + FINRA_TAF_bps    (solo venta)
          + slippage_AC      (modelo de impacto lineal)
        """
        if notional_usd <= 0 or avg_share_price <= 0:
            return 0.0

        # Half-spread round-trip
        spread_cost_bps = 2.0 * self.expected_half_spread_bps

        # SEC fee: solo venta. notional/1M × 27.80, convertido a bps
        sec_fee_usd = (notional_usd / 1_000_000.0) * self.sec_fee_per_million
        sec_fee_bps = (sec_fee_usd / notional_usd) * 10_000.0

        # FINRA TAF: solo venta, $0.000166/share con cap
        shares = notional_usd / avg_share_price
        taf_usd = min(shares * self.finra_taf_per_share, self.finra_taf_cap)
        taf_bps = (taf_usd / notional_usd) * 10_000.0

        # Slippage Almgren-Chriss simplificado:
        # impact_bps ≈ k × (volume_pct_of_ADV) × σ_per_period (en bps)
        # Para tick-second: σ_per_sec = σ_annual / √(31.5M)
        sigma_per_sec_bps = (reference_vol_annualized /
                             math.sqrt(252 * 6.5 * 3600)) * 10_000.0
        slippage_bps_one_way = (self.almgren_linear_impact_coef
                                * self.expected_volume_pct_of_adv
                                * sigma_per_sec_bps * 100.0)
        slippage_round_trip = 2.0 * slippage_bps_one_way

        return spread_cost_bps + sec_fee_bps + taf_bps + slippage_round_trip


# =============================================================================
# Risk Config split por asset class
# =============================================================================

@dataclass
class CryptoRiskConfig:
    """
    Risk parameters específicos a crypto. Vol cap alto (mercado 24/7
    con high-vol regimes legítimos).
    """
    max_acceptable_vol_annualized: float = 2.50    # 250%
    max_position_pct_of_equity: float = 0.30
    per_trade_stop_loss_atr_mult: float = 2.5
    per_trade_take_profit_atr_mult: float = 4.0
    tca_safety_margin: float = 1.5
    # Coherencia entre half_life y horizon implícito (en segundos)
    min_acceptable_half_life_sec: float = 30.0     # < 30s: scalping puro, rechazar
    max_acceptable_half_life_sec: float = 6 * 3600  # > 6h: muy lento, capital atado


@dataclass
class EquityRiskConfig:
    """
    Risk parameters específicos a equity. Vol cap bajo: >60% anualizada
    en equity indica earnings manipulation, halt, o evento corporativo.
    """
    max_acceptable_vol_annualized: float = 0.60    # 60%
    max_position_pct_of_equity: float = 0.20       # más conservador que crypto
    per_trade_stop_loss_atr_mult: float = 3.0      # equity tiene gaps; SL más amplio
    per_trade_take_profit_atr_mult: float = 5.0
    tca_safety_margin: float = 1.5
    min_acceptable_half_life_sec: float = 5 * 60   # 5 min mínimo
    max_acceptable_half_life_sec: float = 5 * 24 * 3600  # 5 días (holding largo OK)
    # Equity-specific risk filters
    max_pct_of_adv: float = 0.005        # 0.5% del ADV — sobre eso, alta slippage
    block_if_earnings_within_hours: int = 24  # Si earnings call <24h, rechazar


@dataclass
class RiskConfig:
    """
    Configuración top-level. Parámetros realmente compartidos viven aquí;
    los específicos viven dentro de los namespaces crypto/equity.
    """
    # Compartidos
    kelly_fraction: float = 0.25
    kelly_max_leverage: float = 1.0
    kelly_smoothing_alpha: float = 0.3
    daily_drawdown_limit_pct: float = 0.02

    # Correlation cluster control (cross-asset)
    correlation_warning_threshold: float = 0.70   # multiplicador kicks in
    correlation_hard_cutoff: float = 0.90         # hard reject
    correlation_lookback_seconds: int = 3600      # ventana 1h

    # Namespaces por asset class
    crypto: CryptoRiskConfig = field(default_factory=CryptoRiskConfig)
    equity: EquityRiskConfig = field(default_factory=EquityRiskConfig)

    def get_class_config(self, asset_class: AssetClass):
        return self.crypto if asset_class == AssetClass.CRYPTO else self.equity


# =============================================================================
# RiskManager
# =============================================================================

class RiskManager:
    """
    Stateful, asíncrono, persistido en Redis. Métodos clave:
        - evaluate(...): single-asset, signature v1-compat extendida.
        - evaluate_pair_trade(...): para señales pair-trading con CointegrationParams.
        - record_price_observation(...): alimenta la matriz de correlación.
    """

    def __init__(self, redis: aioredis.Redis, config: Optional[RiskConfig] = None):
        self.redis = redis
        self.cfg = config or RiskConfig()

    # ---------- Helpers de día ----------

    @staticmethod
    def _today_utc() -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    async def get_or_set_day_start_equity(self, current_equity: float) -> float:
        key = f"risk:daily_equity_start:{self._today_utc()}"
        existing = await self.redis.get(key)
        if existing is not None:
            return float(existing)
        await self.redis.set(key, current_equity, nx=True, ex=60 * 60 * 30)
        val = await self.redis.get(key)
        return float(val)

    async def is_day_locked(self) -> bool:
        return bool(await self.redis.get(f"risk:day_locked:{self._today_utc()}"))

    async def lock_day(self, reason: str):
        await self.redis.set(
            f"risk:day_locked:{self._today_utc()}",
            json.dumps({"locked_at": time.time(), "reason": reason}),
            ex=60 * 60 * 30,
        )

    async def check_daily_drawdown(self, current_equity: float) -> Optional[RiskDecision]:
        if await self.is_day_locked():
            return RiskDecision(
                verdict=RiskVerdict.REJECT_DRAWDOWN,
                size_quote=0.0, stop_loss_price=0.0, take_profit_price=0.0,
                reason="Día bloqueado: drawdown del 2% ya activado.",
            )
        start = await self.get_or_set_day_start_equity(current_equity)
        if start <= 0:
            return None
        dd = (start - current_equity) / start
        if dd >= self.cfg.daily_drawdown_limit_pct:
            await self.lock_day(
                f"DD diario {dd*100:.2f}% ≥ {self.cfg.daily_drawdown_limit_pct*100:.2f}%"
            )
            return RiskDecision(
                verdict=RiskVerdict.REJECT_DRAWDOWN,
                size_quote=0.0, stop_loss_price=0.0, take_profit_price=0.0,
                reason=f"Drawdown diario {dd*100:.2f}% supera límite "
                       f"{self.cfg.daily_drawdown_limit_pct*100:.2f}%. Día bloqueado.",
                metrics={"daily_drawdown": dd, "start_equity": start,
                         "current_equity": current_equity},
            )
        return None

    # ---------- Kelly ----------

    @staticmethod
    def kelly_fraction_from_edge(win_prob: float, win_loss_ratio: float) -> float:
        if win_loss_ratio <= 0:
            return 0.0
        return win_prob - (1.0 - win_prob) / win_loss_ratio

    @staticmethod
    def kelly_continuous(expected_return: float, variance: float) -> float:
        if variance <= 1e-12:
            return 0.0
        return expected_return / variance

    async def _smooth_kelly(self, symbol: str, raw_kelly: float) -> float:
        key = f"risk:kelly_smoothed:{symbol}"
        prev = await self.redis.get(key)
        if prev is None:
            smoothed = raw_kelly
        else:
            a = self.cfg.kelly_smoothing_alpha
            smoothed = a * raw_kelly + (1.0 - a) * float(prev)
        await self.redis.set(key, smoothed, ex=60 * 60 * 24 * 7)
        return smoothed

    # ---------- Correlation tracking ----------

    async def record_price_observation(self, symbol: str, log_return: float):
        """
        Alimenta la ventana rolling para correlación. Llamar desde bot_core
        cada vez que se computa un retorno.

        Usa Redis sorted sets (timestamp como score) para tener TTL natural.
        """
        key = f"risk:returns:{symbol}"
        now = time.time()
        # Score = timestamp; member = "ts:value" para evitar colisiones
        await self.redis.zadd(key, {f"{now}:{log_return:.10e}": now})
        # Limpiar ventana
        cutoff = now - self.cfg.correlation_lookback_seconds
        await self.redis.zremrangebyscore(key, "-inf", cutoff)
        # TTL del key entero por si el símbolo deja de operarse
        await self.redis.expire(key, self.cfg.correlation_lookback_seconds * 2)

    async def get_correlation(self, symbol_a: str, symbol_b: str) -> Optional[float]:
        """
        Correlación de Pearson sobre los retornos rolling de ambos símbolos.
        Devuelve None si no hay suficientes muestras solapadas.

        Implementación: align temporal por timestamp, calcular Pearson.
        """
        now = time.time()
        cutoff = now - self.cfg.correlation_lookback_seconds

        # Pull de retornos de ambos símbolos en la ventana
        raw_a = await self.redis.zrangebyscore(
            f"risk:returns:{symbol_a}", cutoff, now, withscores=True)
        raw_b = await self.redis.zrangebyscore(
            f"risk:returns:{symbol_b}", cutoff, now, withscores=True)

        if len(raw_a) < 30 or len(raw_b) < 30:
            return None

        # Parsear: cada "member" es "ts:value"
        def parse(rows):
            ts_list = []; val_list = []
            for member, score in rows:
                try:
                    _, val = member.rsplit(":", 1)
                    val_list.append(float(val))
                    ts_list.append(float(score))
                except (ValueError, IndexError):
                    continue
            return ts_list, val_list

        ts_a, val_a = parse(raw_a)
        ts_b, val_b = parse(raw_b)

        if len(val_a) < 30 or len(val_b) < 30:
            return None

        # Align: bucketear ambos por segundos enteros y tomar último valor
        # de cada bucket. Es aproximado pero robusto sin numpy aquí.
        def bucket(ts_list, val_list):
            d = {}
            for t, v in zip(ts_list, val_list):
                d[int(t)] = v
            return d
        d_a = bucket(ts_a, val_a)
        d_b = bucket(ts_b, val_b)
        common = sorted(set(d_a.keys()) & set(d_b.keys()))
        if len(common) < 30:
            return None

        a_arr = [d_a[t] for t in common]
        b_arr = [d_b[t] for t in common]

        # Pearson manual (evitamos importar numpy aquí para mantener cold-start rápido)
        n = len(a_arr)
        mean_a = sum(a_arr) / n
        mean_b = sum(b_arr) / n
        num = sum((a - mean_a) * (b - mean_b) for a, b in zip(a_arr, b_arr))
        var_a = sum((a - mean_a) ** 2 for a in a_arr)
        var_b = sum((b - mean_b) ** 2 for b in b_arr)
        if var_a < 1e-15 or var_b < 1e-15:
            return None
        return num / math.sqrt(var_a * var_b)

    async def correlation_penalty(self, symbol: str,
                                  other_symbols: list[str]) -> tuple[float, dict]:
        """
        Devuelve (penalty_multiplier, info_dict).

        penalty_multiplier ∈ [0.0, 1.0]:
            - 1.0  → sin penalización
            - 0.0  → hard reject
            - intermedio → degrada Kelly proporcionalmente

        Lógica:
            |ρ_max| ≤ correlation_warning_threshold   → 1.0
            |ρ_max| ≥ correlation_hard_cutoff         → 0.0
            entre los dos                              → interp lineal
        """
        max_abs_corr = 0.0
        worst_pair = None
        details = {}
        for other in other_symbols:
            if other == symbol:
                continue
            corr = await self.get_correlation(symbol, other)
            if corr is None:
                continue
            details[other] = corr
            if abs(corr) > abs(max_abs_corr):
                max_abs_corr = corr
                worst_pair = other

        warn = self.cfg.correlation_warning_threshold
        hard = self.cfg.correlation_hard_cutoff

        if abs(max_abs_corr) < warn:
            mult = 1.0
        elif abs(max_abs_corr) >= hard:
            mult = 0.0
        else:
            # Lineal: en warn → 1.0, en hard → 0.0
            mult = 1.0 - (abs(max_abs_corr) - warn) / (hard - warn)

        return mult, {
            "max_abs_corr": max_abs_corr,
            "worst_pair": worst_pair,
            "all_correlations": details,
            "penalty_multiplier": mult,
        }

    # ---------- TCA ----------

    def tca_check_crypto(self, expected_alpha_bps: float,
                         costs: CryptoTradingCosts,
                         use_taker: bool = False) -> tuple[bool, float, float]:
        threshold = costs.round_trip_bps(use_taker=use_taker) \
                    * self.cfg.crypto.tca_safety_margin
        return (expected_alpha_bps >= threshold, expected_alpha_bps, threshold)

    def tca_check_equity(self, expected_alpha_bps: float,
                         costs: EquityTradingCosts,
                         notional_usd: float,
                         avg_share_price: float,
                         reference_vol_annualized: float
                         ) -> tuple[bool, float, float]:
        rt_bps = costs.round_trip_bps(notional_usd, avg_share_price,
                                       reference_vol_annualized)
        threshold = rt_bps * self.cfg.equity.tca_safety_margin
        return (expected_alpha_bps >= threshold, expected_alpha_bps, threshold)

    # ---------- Half-life coherencia ----------

    def _check_half_life_coherent(self, half_life_sec: float,
                                   asset_class: AssetClass) -> Optional[str]:
        """
        Devuelve un mensaje si la half-life es incoherente con el horizon
        de holding implícito, None si está OK.
        """
        cls_cfg = self.cfg.get_class_config(asset_class)
        if half_life_sec < cls_cfg.min_acceptable_half_life_sec:
            return (f"Half-life {half_life_sec:.1f}s < mínimo "
                    f"{cls_cfg.min_acceptable_half_life_sec:.1f}s para "
                    f"{asset_class.value}. Es scalping puro / probable noise.")
        if half_life_sec > cls_cfg.max_acceptable_half_life_sec:
            return (f"Half-life {half_life_sec:.1f}s > máximo "
                    f"{cls_cfg.max_acceptable_half_life_sec:.1f}s para "
                    f"{asset_class.value}. Capital atado demasiado tiempo.")
        return None

    # ---------- Evaluación single-asset ----------

    async def evaluate(
        self,
        *,
        symbol: str,                       # "CRYPTO:BTC/USDT" o "EQUITY:AAPL"
        current_equity: float,
        current_price: float,
        expected_alpha_bps: float,
        win_prob: float,
        win_loss_ratio: float,
        garch_variance_per_period: float,
        garch_periods_per_year: int,
        atr: float,
        side: str,
        ou_half_life_sec: float,
        # Crypto-specific
        crypto_costs: Optional[CryptoTradingCosts] = None,
        use_taker: bool = False,
        # Equity-specific
        equity_costs: Optional[EquityTradingCosts] = None,
        # Cross-asset awareness
        other_open_symbols: Optional[list[str]] = None,
        market_is_open: bool = True,
    ) -> RiskDecision:
        """
        Aplica todas las barreras y devuelve el sizing final.

        Orden de evaluación:
            0. Mercado abierto? (relevante para equity)
            1. Day-lock / drawdown
            2. Régimen de volatilidad (cap diferenciado por asset class)
            3. Half-life coherente con horizon
            4. TCA (fórmula diferenciada)
            5. Correlation penalty (cross-asset)
            6. Kelly (smoothed + capped por class config)
            7. SL/TP basados en ATR
        """
        # ---- 0. Mercado abierto ----
        if not market_is_open:
            return RiskDecision(
                verdict=RiskVerdict.REJECT_MARKET_CLOSED,
                size_quote=0.0, stop_loss_price=0.0, take_profit_price=0.0,
                reason=f"Mercado cerrado para {symbol}. Esperando reapertura.",
            )

        # ---- 1. Drawdown ----
        dd_check = await self.check_daily_drawdown(current_equity)
        if dd_check is not None:
            return dd_check

        # ---- 2. Asset class y configs ----
        asset_class, native_symbol = parse_asset_class(symbol)
        class_cfg = self.cfg.get_class_config(asset_class)

        # ---- 3. Régimen de volatilidad ----
        vol_annualized = math.sqrt(max(garch_variance_per_period, 0.0)
                                    * garch_periods_per_year)
        if vol_annualized > class_cfg.max_acceptable_vol_annualized:
            return RiskDecision(
                verdict=RiskVerdict.REJECT_VOLATILITY_REGIME,
                size_quote=0.0, stop_loss_price=0.0, take_profit_price=0.0,
                reason=(f"Vol anualizada {vol_annualized*100:.1f}% > límite "
                        f"{class_cfg.max_acceptable_vol_annualized*100:.1f}% "
                        f"para {asset_class.value}."),
                metrics={"vol_annualized": vol_annualized, "asset_class": asset_class.value},
            )

        # ---- 4. Half-life coherente ----
        hl_msg = self._check_half_life_coherent(ou_half_life_sec, asset_class)
        if hl_msg is not None:
            return RiskDecision(
                verdict=RiskVerdict.REJECT_HALF_LIFE_INCOHERENT,
                size_quote=0.0, stop_loss_price=0.0, take_profit_price=0.0,
                reason=hl_msg,
                metrics={"ou_half_life_sec": ou_half_life_sec},
            )

        # ---- 5. TCA (diferenciado) ----
        if asset_class == AssetClass.CRYPTO:
            costs_c = crypto_costs or CryptoTradingCosts()
            passes_tca, alpha_bps, thresh_bps = self.tca_check_crypto(
                expected_alpha_bps, costs_c, use_taker=use_taker)
        else:
            costs_e = equity_costs or EquityTradingCosts()
            # Estimación preliminar del notional para TCA
            preliminary_notional = self.cfg.kelly_fraction * current_equity
            passes_tca, alpha_bps, thresh_bps = self.tca_check_equity(
                expected_alpha_bps, costs_e, preliminary_notional,
                current_price, vol_annualized)

        if not passes_tca:
            return RiskDecision(
                verdict=RiskVerdict.REJECT_TCA,
                size_quote=0.0, stop_loss_price=0.0, take_profit_price=0.0,
                reason=f"TCA falló ({asset_class.value}): alpha={alpha_bps:.2f}bps "
                       f"< umbral={thresh_bps:.2f}bps.",
                metrics={"alpha_bps": alpha_bps, "threshold_bps": thresh_bps,
                         "asset_class": asset_class.value},
            )

        # ---- 6. Correlation penalty ----
        corr_mult = 1.0
        corr_info = {}
        if other_open_symbols:
            corr_mult, corr_info = await self.correlation_penalty(
                symbol, other_open_symbols)
            if corr_mult <= 0.0:
                return RiskDecision(
                    verdict=RiskVerdict.REJECT_CORRELATION_CLUSTER,
                    size_quote=0.0, stop_loss_price=0.0, take_profit_price=0.0,
                    reason=(f"Correlación cluster: |ρ| con {corr_info.get('worst_pair')} "
                            f"= {corr_info.get('max_abs_corr'):.3f} ≥ "
                            f"hard cutoff {self.cfg.correlation_hard_cutoff:.2f}."),
                    metrics={"correlation": corr_info},
                )

        # ---- 7. Kelly ----
        kelly_discrete = self.kelly_fraction_from_edge(win_prob, win_loss_ratio)
        alpha_decimal = expected_alpha_bps / 10_000.0
        kelly_continuous = self.kelly_continuous(alpha_decimal,
                                                  garch_variance_per_period)
        kelly_raw = min(kelly_discrete, kelly_continuous)

        if kelly_raw <= 0:
            return RiskDecision(
                verdict=RiskVerdict.REJECT_KELLY_NEGATIVE,
                size_quote=0.0, stop_loss_price=0.0, take_profit_price=0.0,
                reason=f"Kelly raw ≤ 0: discrete={kelly_discrete:.4f}, "
                       f"continuous={kelly_continuous:.4f}",
                metrics={"kelly_discrete": kelly_discrete,
                         "kelly_continuous": kelly_continuous},
            )

        kelly_smoothed = await self._smooth_kelly(symbol, kelly_raw)
        kelly_final = min(
            kelly_smoothed * self.cfg.kelly_fraction * corr_mult,
            self.cfg.kelly_max_leverage,
            class_cfg.max_position_pct_of_equity,
        )

        if kelly_final <= 0:
            return RiskDecision(
                verdict=RiskVerdict.REJECT_KELLY_NEGATIVE,
                size_quote=0.0, stop_loss_price=0.0, take_profit_price=0.0,
                reason="Kelly suavizado terminó ≤0 tras tope/penalización.",
            )

        size_quote = kelly_final * current_equity

        # ---- 8. SL/TP por ATR (con multiplicadores por asset class) ----
        # Coherencia dimensional: ATR ya viene en unidades de precio del
        # subyacente. Forzamos no-negatividad por si el estimador de ATR
        # entrega valores degenerados durante gaps o cold-start.
        sl_dist = max(float(atr), 0.0) * class_cfg.per_trade_stop_loss_atr_mult
        tp_dist = max(float(atr), 0.0) * class_cfg.per_trade_take_profit_atr_mult
        if side == "long":
            sl = current_price - sl_dist
            tp = current_price + tp_dist
        elif side == "short":
            sl = current_price + sl_dist
            tp = current_price - tp_dist
        else:
            raise ValueError(f"side debe ser 'long' o 'short', recibido {side!r}")

        # Sanitización defensiva antes de devolver al broker adapter:
        #   1. Floor positivo: equities y crypto cotizan > 0; un TP negativo
        #      en un short con ATR enorme reventaría la validación de Alpaca.
        #   2. Redondeo a 2 decimales (tick size estándar equity > $1).
        #   3. Invariante de orientación: long ⇒ TP > entry > SL; short ⇒
        #      SL > entry > TP. Si la sanitización rompió la invariante,
        #      revertir al snap mínimo (1 tick de separación).
        MIN_PRICE = 0.01
        MIN_GAP = 0.01
        sl = round(max(sl, MIN_PRICE), 2)
        tp = round(max(tp, MIN_PRICE), 2)

        if side == "long" and tp <= sl:
            tp = round(sl + MIN_GAP, 2)
        elif side == "short" and tp >= sl:
            # En short, TP debe ser estrictamente menor que SL. Si la
            # sanitización los aplastó al floor, separar artificialmente
            # SL hacia arriba para preservar la geometría.
            sl = round(max(sl, tp + MIN_GAP), 2)
            # Y si el floor también empujó tp hacia arriba del precio,
            # asegurar tp < current_price para que no sea inejecutable.
            if tp >= current_price:
                tp = round(max(current_price - MIN_GAP, MIN_PRICE), 2)
                sl = round(max(sl, tp + MIN_GAP), 2)

        return RiskDecision(
            verdict=RiskVerdict.APPROVED,
            size_quote=size_quote,
            stop_loss_price=sl, take_profit_price=tp,
            reason=(
                f"APROBADO [{asset_class.value}]. Kelly raw={kelly_raw:.4f}, "
                f"smoothed={kelly_smoothed:.4f}, corr_mult={corr_mult:.3f}, "
                f"final={kelly_final:.4f}. Alpha={alpha_bps:.2f}bps "
                f"(umbral {thresh_bps:.2f}bps). Vol anualizada={vol_annualized*100:.1f}%."
            ),
            metrics={
                "asset_class": asset_class.value,
                "kelly_raw": kelly_raw,
                "kelly_smoothed": kelly_smoothed,
                "kelly_final": kelly_final,
                "correlation_mult": corr_mult,
                "correlation_info": corr_info,
                "alpha_bps": alpha_bps,
                "threshold_bps": thresh_bps,
                "vol_annualized": vol_annualized,
                "ou_half_life_sec": ou_half_life_sec,
                "sl_distance": sl_dist,
                "tp_distance": tp_dist,
            },
        )

    # ---------- Pair-trade evaluation ----------

    async def evaluate_pair_trade(
        self,
        *,
        symbol_a: str,
        symbol_b: str,
        current_equity: float,
        price_a: float,
        price_b: float,
        beta_hedge: float,
        spread_zscore: float,
        spread_half_life_sec: float,
        spread_garch_variance: float,
        spread_garch_periods_per_year: int,
        atr_spread: float,
        cointegration_verdict: str,    # de CointegrationVerdict
        # Costos por leg
        crypto_costs: Optional[CryptoTradingCosts] = None,
        equity_costs: Optional[EquityTradingCosts] = None,
        market_is_open_a: bool = True,
        market_is_open_b: bool = True,
    ) -> RiskDecision:
        """
        Sizing para pair trade. Determina:
            - side: 'long_spread' si z << 0 (long A, short B), 'short_spread' si z >> 0
            - leg sizes: ajustados por β para neutralidad de hedge
            - barreras aplicadas al spread como si fuera un activo univariante
        """
        # ---- 0. Ambos mercados abiertos ----
        if not (market_is_open_a and market_is_open_b):
            closed = []
            if not market_is_open_a: closed.append(symbol_a)
            if not market_is_open_b: closed.append(symbol_b)
            return RiskDecision(
                verdict=RiskVerdict.REJECT_MARKET_CLOSED,
                size_quote=0.0, stop_loss_price=0.0, take_profit_price=0.0,
                reason=f"Mercado(s) cerrado(s) para pair: {closed}",
            )

        # ---- 1. Drawdown ----
        dd_check = await self.check_daily_drawdown(current_equity)
        if dd_check is not None:
            return dd_check

        # ---- 2. Cointegración operable ----
        if cointegration_verdict != "cointegrated":
            return RiskDecision(
                verdict=RiskVerdict.REJECT_PAIR_NOT_OPERABLE,
                size_quote=0.0, stop_loss_price=0.0, take_profit_price=0.0,
                reason=f"Cointegración no operable: verdict={cointegration_verdict}",
            )

        # ---- 3. Half-life sobre el spread ----
        # Tomamos el namespace del leg más restrictivo (equity si involucrado)
        ac_a, _ = parse_asset_class(symbol_a)
        ac_b, _ = parse_asset_class(symbol_b)
        restrictive_class = AssetClass.EQUITY if AssetClass.EQUITY in (ac_a, ac_b) \
                            else AssetClass.CRYPTO
        hl_msg = self._check_half_life_coherent(spread_half_life_sec, restrictive_class)
        if hl_msg is not None:
            return RiskDecision(
                verdict=RiskVerdict.REJECT_HALF_LIFE_INCOHERENT,
                size_quote=0.0, stop_loss_price=0.0, take_profit_price=0.0,
                reason=f"[PAIR] {hl_msg}",
                metrics={"spread_half_life_sec": spread_half_life_sec},
            )

        # ---- 4. Vol del spread ----
        spread_vol_ann = math.sqrt(max(spread_garch_variance, 0.0)
                                    * spread_garch_periods_per_year)
        cls_cfg = self.cfg.get_class_config(restrictive_class)
        if spread_vol_ann > cls_cfg.max_acceptable_vol_annualized:
            return RiskDecision(
                verdict=RiskVerdict.REJECT_VOLATILITY_REGIME,
                size_quote=0.0, stop_loss_price=0.0, take_profit_price=0.0,
                reason=f"[PAIR] Spread vol {spread_vol_ann*100:.1f}% "
                       f"> límite {cls_cfg.max_acceptable_vol_annualized*100:.1f}%",
            )

        # ---- 5. Determinar dirección ----
        # z << 0: spread está bajo → long A, short B (long_spread)
        # z >> 0: spread está alto → short A, long B (short_spread)
        if abs(spread_zscore) < 2.0:
            return RiskDecision(
                verdict=RiskVerdict.REJECT_KELLY_NEGATIVE,
                size_quote=0.0, stop_loss_price=0.0, take_profit_price=0.0,
                reason=f"|z|={abs(spread_zscore):.3f} < 2.0; señal insuficiente.",
            )
        long_spread = spread_zscore < 0
        pair_side = "long_spread" if long_spread else "short_spread"

        # ---- 6. Sizing (Kelly continuo sobre el spread) ----
        # alpha esperado del trade = 0.5 × |z| × stationary_std_del_spread
        # donde stationary_std = σ_spread_por_periodo × √(holding_periods)
        # holding_periods ≈ half_life / dt
        # En esta llamada, garch_variance es por-periodo y periods_per_year nos
        # da el dt implícito: dt = 1/periods_per_year (años). half_life está
        # en segundos, así que convertimos a "periods":
        dt_seconds = (365 * 24 * 3600) / spread_garch_periods_per_year
        holding_periods = max(spread_half_life_sec / max(dt_seconds, 1e-9), 1.0)
        stationary_std = math.sqrt(max(spread_garch_variance, 0.0)
                                    * holding_periods)
        alpha_decimal = 0.5 * abs(spread_zscore) * stationary_std

        # Kelly continuo sobre el spread: usar la varianza *acumulada en el
        # holding period*, no por-segundo, para consistencia dimensional.
        holding_variance = spread_garch_variance * holding_periods
        kelly_continuous = self.kelly_continuous(alpha_decimal, holding_variance)
        if kelly_continuous <= 0:
            return RiskDecision(
                verdict=RiskVerdict.REJECT_KELLY_NEGATIVE,
                size_quote=0.0, stop_loss_price=0.0, take_profit_price=0.0,
                reason="Kelly continuo ≤ 0 sobre spread.",
            )

        kelly_smoothed = await self._smooth_kelly(f"PAIR:{symbol_a}/{symbol_b}",
                                                   kelly_continuous)
        kelly_final = min(
            kelly_smoothed * self.cfg.kelly_fraction,
            self.cfg.kelly_max_leverage,
            cls_cfg.max_position_pct_of_equity,
        )
        gross_size_quote = kelly_final * current_equity

        # Leg sizes: A recibe el size completo; B recibe size · β (en valor)
        # Si β = 1.3 y gross = $1000, A = $1000, B = $1300 (en dirección opuesta)
        leg_a = gross_size_quote
        leg_b = gross_size_quote * abs(beta_hedge)

        # ---- 7. TCA sobre el ROUND TRIP DEL SPREAD ----
        # Cada leg paga sus propios costos. Sumamos.
        total_cost_bps = 0.0
        if ac_a == AssetClass.CRYPTO:
            cc = crypto_costs or CryptoTradingCosts()
            total_cost_bps += cc.round_trip_bps(use_taker=False)
        else:
            ec = equity_costs or EquityTradingCosts()
            total_cost_bps += ec.round_trip_bps(leg_a, price_a, spread_vol_ann)
        if ac_b == AssetClass.CRYPTO:
            cc = crypto_costs or CryptoTradingCosts()
            total_cost_bps += cc.round_trip_bps(use_taker=False)
        else:
            ec = equity_costs or EquityTradingCosts()
            total_cost_bps += ec.round_trip_bps(leg_b, price_b, spread_vol_ann)

        # Alpha esperado del trade en bps: usar stationary_std del holding period
        expected_alpha_bps = alpha_decimal * 10_000.0
        threshold = total_cost_bps * cls_cfg.tca_safety_margin
        if expected_alpha_bps < threshold:
            return RiskDecision(
                verdict=RiskVerdict.REJECT_TCA,
                size_quote=0.0, stop_loss_price=0.0, take_profit_price=0.0,
                reason=f"[PAIR] TCA: alpha={expected_alpha_bps:.2f}bps "
                       f"< umbral={threshold:.2f}bps (suma 2 legs).",
                metrics={"alpha_bps": expected_alpha_bps, "threshold_bps": threshold},
            )

        # ---- 8. SL/TP sobre el spread (en log-units) ----
        sl_dist = atr_spread * cls_cfg.per_trade_stop_loss_atr_mult
        tp_dist = atr_spread * cls_cfg.per_trade_take_profit_atr_mult

        return RiskDecision(
            verdict=RiskVerdict.APPROVED,
            size_quote=gross_size_quote,
            leg_a_size_quote=leg_a,
            leg_b_size_quote=leg_b,
            stop_loss_price=sl_dist,    # interpretado en spread-units
            take_profit_price=tp_dist,  # idem
            reason=(
                f"APROBADO [PAIR {pair_side}]. β={beta_hedge:.4f}, z={spread_zscore:.3f}, "
                f"half_life={spread_half_life_sec:.1f}s. "
                f"Kelly final={kelly_final:.4f}. Leg A=${leg_a:,.2f}, "
                f"Leg B=${leg_b:,.2f}. Alpha={expected_alpha_bps:.2f}bps "
                f"(umbral {threshold:.2f}bps)."
            ),
            metrics={
                "pair_side": pair_side,
                "beta_hedge": beta_hedge,
                "spread_zscore": spread_zscore,
                "spread_half_life_sec": spread_half_life_sec,
                "kelly_final": kelly_final,
                "leg_a_size_quote": leg_a,
                "leg_b_size_quote": leg_b,
                "alpha_bps": expected_alpha_bps,
                "threshold_bps": threshold,
                "spread_vol_annualized": spread_vol_ann,
                "restrictive_class": restrictive_class.value,
            },
        )

    # ---------- Serialización ----------

    def to_json(self, decision: RiskDecision) -> str:
        d = asdict(decision)
        d["verdict"] = decision.verdict.value
        return json.dumps(d, default=str)
