"""
engine_math.py  (v2)
====================
Núcleo cuantitativo extendido para tracking univariante (v1) y bivariate
cointegrado (v2). Diseñado para soportar:

    - Single-asset OU/Kalman/GARCH (firmas v1, backwards-compatible).
    - Pair-trading: estimación de hedge ratio β por OLS + ADF (Engle-Granger),
      construcción de spread X_t = ln(A) − β·ln(B) − α, y aplicación del
      mismo motor OU/Kalman al spread.
    - Manejo de market closures: gap-update del Kalman con inflación
      explícita de varianza (P y R) sin imputar precios sintéticos.
    - Detección de structural break en el hedge ratio rolling.

Filosofía de extensión v1 → v2:
    1. NO romper firmas v1. El bot existente sigue funcionando.
    2. NO meter NumPy matrices al Kalman escalar; sigue siendo O(1).
    3. NO imputar precios durante un cierre; sí inflar P y R en la reapertura.
    4. Break estructural se expone como flag (verdict), no como excepción.

Referencias v2:
    - Engle, R. F., & Granger, C. W. J. (1987). "Co-integration and error
      correction: representation, estimation, and testing." Econometrica.
    - Gatev, Goetzmann, Rouwenhorst (2006). "Pairs Trading: Performance
      of a Relative-Value Arbitrage Rule." RFS.
    - MacKinnon, J. G. (1996). "Numerical Distribution Functions for Unit
      Root and Cointegration Tests." J. Applied Econometrics.
    - Vidyamurthy (2004). Pairs Trading: Quantitative Methods and Analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize


# =============================================================================
# v1 RETAINED: Ornstein-Uhlenbeck
# =============================================================================

@dataclass(frozen=True)
class OUParams:
    """
    Parámetros de OU:  dX_t = θ·(μ − X_t)·dt + σ·dW_t

    Agnóstica a qué representa X_t: precio, log-precio o spread cointegrado.
    El consumidor (RiskManager) interpreta según contexto.
    """
    theta: float
    mu: float
    sigma: float
    half_life: float
    dt: float

    @property
    def is_operable(self) -> bool:
        return self.theta > 0 and math.isfinite(self.half_life) and self.sigma > 0


def fit_ornstein_uhlenbeck(series: np.ndarray,
                           dt: float = 1.0 / (252 * 24 * 60)) -> OUParams:
    """
    MLE por OLS sobre discretización exacta:
        X_{t+1} = a + b·X_t + ε,  b=exp(-θ·dt), a=μ(1-b), Var(ε)=σ²(1-b²)/(2θ)
    """
    x = np.asarray(series, dtype=np.float64)
    if x.size < 50:
        raise ValueError(f"OU requiere ≥50 observaciones, recibido {x.size}")
    if not np.all(np.isfinite(x)):
        raise ValueError("OU recibió NaN/Inf en la serie")

    x_t = x[:-1]; x_t1 = x[1:]
    n = x_t.size
    sx = x_t.sum(); sy = x_t1.sum()
    sxx = (x_t * x_t).sum(); sxy = (x_t * x_t1).sum()

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        raise ValueError("Serie OU degenerada: varianza ~0")

    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    resid = x_t1 - (a + b * x_t)
    var_resid = (resid * resid).sum() / max(n - 2, 1)

    if b <= 0 or b >= 1:
        return OUParams(
            theta=0.0, mu=float(x.mean()),
            sigma=float(np.sqrt(max(var_resid / dt, 0.0))),
            half_life=math.inf, dt=dt,
        )

    theta = -math.log(b) / dt
    mu = a / (1.0 - b)
    sigma2 = var_resid * (2.0 * theta) / (1.0 - b * b)
    sigma = math.sqrt(max(sigma2, 0.0))
    half_life = math.log(2.0) / theta
    return OUParams(theta=theta, mu=mu, sigma=sigma, half_life=half_life, dt=dt)


def ou_zscore(price: float, params: OUParams) -> float:
    """Z-score bajo X_∞ ~ N(μ, σ²/(2θ)). 0 si no operable."""
    if not params.is_operable:
        return 0.0
    std = params.sigma / math.sqrt(2.0 * params.theta)
    if std < 1e-12:
        return 0.0
    return (price - params.mu) / std


# =============================================================================
# v1 RETAINED + v2 EXTENSION: Kalman Filter escalar
# =============================================================================

@dataclass
class KalmanState:
    """
    Kalman escalar (F=H=1):
        x_t = x_{t-1} + w_t,   w_t ~ N(0, Q)   (estado oculto)
        y_t = x_t + v_t,       v_t ~ N(0, R)   (observación)

    O(1) por update — sin matrices.
    """
    x: float
    P: float
    Q: float
    R: float

    def update(self, y: float) -> Tuple[float, float]:
        """Predict+update estándar."""
        x_prior = self.x
        P_prior = self.P + self.Q
        S = P_prior + self.R
        if S <= 1e-15:
            S = 1e-15
        K = P_prior / S
        innov = y - x_prior
        self.x = x_prior + K * innov
        self.P = (1.0 - K) * P_prior
        return self.x, self.P

    # ----- v2 EXTENSION: gap handler ---------------------------------

    def update_after_gap(self, y: float, gap_seconds: float,
                         gap_volatility_per_sqrt_sec: float) -> Tuple[float, float]:
        """
        Update tras un período sin observaciones (cierre de mercado, halt,
        weekend para equities).

        Lógica matemática:
            1. Durante el gap, el estado oculto difunde libremente sin
               correcciones de observación. Acumulamos esa varianza:
                  ΔP = (gap_vol_per_√s)² · gap_seconds
               Esto es exacto bajo un random walk con varianza Q·Δt.
            2. La PRIMERA observación post-reapertura tiene típicamente
               más ruido de micro-estructura (gap-jump, baja liquidez en
               la apertura). Inflamos R temporalmente:
                  R_inflated = R_base · √(max(gap_seconds, 1))
            3. Update normal con esos parámetros.
            4. Restauramos R_base (solo el primer tick post-gap recibe
               este tratamiento).

        Por qué NO imputamos precios durante el gap:
            - Imputar (e.g., forward-fill) genera retornos sintéticos cero
              que contaminan GARCH y OU.
            - Imputar (e.g., interpolación lineal) introduce información
              futura (look-ahead bias).
            - Solo inflar P+R es la opción matemáticamente consistente con
              "no tenemos información durante este intervalo".

        Por qué inflar R con √Δt y no Δt:
            - Asume que el ruido de micro-estructura post-gap escala como
              la volatilidad acumulada, no como la varianza acumulada.
              Empíricamente más conservador y estable.
        """
        if gap_seconds <= 0:
            return self.update(y)

        gap_variance = (gap_volatility_per_sqrt_sec ** 2) * gap_seconds
        self.P = self.P + gap_variance

        R_base = self.R
        R_inflated = R_base * math.sqrt(max(gap_seconds, 1.0))
        self.R = R_inflated
        try:
            x_post, P_post = self.update(y)
        finally:
            self.R = R_base
        return x_post, P_post


def init_kalman(initial_price: float, obs_variance: float,
                process_variance: float) -> KalmanState:
    return KalmanState(x=initial_price, P=1.0, Q=process_variance, R=obs_variance)


def kalman_smooth_batch(prices: np.ndarray, Q: float, R: float) -> np.ndarray:
    n = prices.size
    out = np.empty(n, dtype=np.float64)
    state = KalmanState(x=float(prices[0]), P=1.0, Q=Q, R=R)
    out[0] = state.x
    for i in range(1, n):
        state.update(float(prices[i]))
        out[i] = state.x
    return out


# =============================================================================
# v1 RETAINED: GARCH(1,1)
# =============================================================================

@dataclass(frozen=True)
class GARCHParams:
    mu: float
    omega: float
    alpha: float
    beta: float
    loglik: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def unconditional_variance(self) -> float:
        p = self.persistence
        if p >= 1.0:
            return math.inf
        return self.omega / (1.0 - p)


def _garch_recursion(returns: np.ndarray, mu: float, omega: float,
                     alpha: float, beta: float) -> np.ndarray:
    n = returns.size
    eps = returns - mu
    sigma2 = np.empty(n, dtype=np.float64)
    persistence = alpha + beta
    if persistence < 1.0:
        sigma2[0] = omega / (1.0 - persistence)
    else:
        sigma2[0] = float(np.var(eps)) + 1e-8
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] * eps[t - 1] + beta * sigma2[t - 1]
    return sigma2


def _garch_neg_loglik(params: np.ndarray, returns: np.ndarray) -> float:
    mu, omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
        return 1e10
    sigma2 = _garch_recursion(returns, mu, omega, alpha, beta)
    eps = returns - mu
    log_term = np.log(sigma2 + 1e-300)
    quad_term = (eps * eps) / (sigma2 + 1e-300)
    nll = 0.5 * np.sum(math.log(2.0 * math.pi) + log_term + quad_term)
    if not np.isfinite(nll):
        return 1e10
    return float(nll)


def fit_garch(returns: np.ndarray) -> GARCHParams:
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 200:
        raise ValueError(f"GARCH requiere ≥200 retornos, recibido {r.size}")
    if not np.all(np.isfinite(r)):
        raise ValueError("GARCH recibió NaN/Inf")
    var_r = float(np.var(r))
    x0 = np.array([float(np.mean(r)), 0.05 * var_r, 0.08, 0.90])
    bounds = [(-0.1, 0.1), (1e-12, 10.0 * var_r), (1e-6, 0.5), (1e-6, 0.999)]
    result = minimize(_garch_neg_loglik, x0, args=(r,),
                      method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 500, "ftol": 1e-9})
    mu, omega, alpha, beta = result.x
    return GARCHParams(mu=float(mu), omega=float(omega),
                       alpha=float(alpha), beta=float(beta),
                       loglik=float(-result.fun))


def garch_forecast_variance(params: GARCHParams, last_eps: float,
                            last_sigma2: float, horizon: int = 1) -> np.ndarray:
    sigma2_next = params.omega + params.alpha * last_eps * last_eps \
                  + params.beta * last_sigma2
    V_inf = params.unconditional_variance
    h = np.arange(1, horizon + 1)
    if math.isinf(V_inf):
        return np.full(horizon, sigma2_next, dtype=np.float64)
    return V_inf + (params.persistence ** (h - 1)) * (sigma2_next - V_inf)


# =============================================================================
# v2 NEW: Cointegración Engle-Granger + pair spreads
# =============================================================================

class CointegrationVerdict(str, Enum):
    """
    Resultado del test de cointegración.

    COINTEGRATED      -> spread estacionario al 5%, operable.
    BORDERLINE        -> estacionario al 10% pero no al 5%, usar con cautela.
    NOT_COINTEGRATED  -> raíz unitaria en residuos, NO operar.
    STRUCTURAL_BREAK  -> hedge ratio rolling se rompió; régimen cambió.
    """
    COINTEGRATED = "cointegrated"
    BORDERLINE = "borderline"
    NOT_COINTEGRATED = "not_cointegrated"
    STRUCTURAL_BREAK = "structural_break"


@dataclass(frozen=True)
class CointegrationParams:
    """
    Resultado del ajuste cointegrado de un par (A, B).

    Modelo:   ln(P_A) = α + β·ln(P_B) + ε    (regresión cointegrante)
    Spread:   X_t = ln(P_A_t) − β·ln(P_B_t) − α

    Convención: A = asset menos líquido / con más alpha;
                B = asset hedger (más líquido).

    Atributos
    ---------
    beta : float
        Hedge ratio. Cuántas unidades de B (log) hedgean 1 unidad de A.
    alpha : float
        Intercepto. Refleja la escala de precios.
    adf_statistic : float
        Estadístico ADF sobre residuos. Más negativo = más estacionario.
    adf_pvalue : float
        p-valor aproximado (MacKinnon 1996).
    spread_std : float
        Std del spread in-sample. Para z-scoring rápido sin OU.
    verdict : CointegrationVerdict
        Decisión final.
    beta_rolling_std : float
        Std de β rolling. Alto = par inestable.
    """
    beta: float
    alpha: float
    adf_statistic: float
    adf_pvalue: float
    spread_std: float
    verdict: CointegrationVerdict
    beta_rolling_std: float = 0.0

    @property
    def is_operable(self) -> bool:
        return self.verdict == CointegrationVerdict.COINTEGRATED


def _ols_simple(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """OLS univariada y = α + β·x. Forma cerrada. Devuelve (α, β)."""
    n = x.size
    sx = x.sum(); sy = y.sum()
    sxx = (x * x).sum(); sxy = (x * y).sum()
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return float(y.mean()), 0.0
    beta = (n * sxy - sx * sy) / denom
    alpha = (sy - beta * sx) / n
    return float(alpha), float(beta)


def _adf_pvalue(adf_stat: float) -> float:
    """
    p-valor aproximado vía interpolación lineal sobre tabla MacKinnon (1996),
    caso "no constant, no trend". Aprox. suficiente para decisión operativa;
    para análisis publicable usar statsmodels.
    """
    table = [
        (-3.50, 0.001),
        (-2.58, 0.01),
        (-1.95, 0.05),
        (-1.62, 0.10),
        (-1.28, 0.20),
        (-0.50, 0.50),
        ( 0.50, 0.80),
    ]
    if adf_stat <= table[0][0]:
        return table[0][1]
    if adf_stat >= table[-1][0]:
        return table[-1][1]
    for i in range(len(table) - 1):
        s1, p1 = table[i]
        s2, p2 = table[i + 1]
        if s1 <= adf_stat <= s2:
            w = (adf_stat - s1) / (s2 - s1) if s2 != s1 else 0.0
            return p1 + w * (p2 - p1)
    return 0.5


def _adf_test(residuals: np.ndarray, lags: int = 1) -> Tuple[float, float]:
    """
    Augmented Dickey-Fuller, regresión sin constante ni tendencia:
        Δr_t = γ·r_{t-1} + Σ φ_i·Δr_{t-i} + u_t
    H0: γ = 0 (raíz unitaria),  H1: γ < 0 (estacionario).

    Devuelve (ADF stat, p-value aproximado).
    """
    r = np.asarray(residuals, dtype=np.float64)
    n = r.size
    if n < 20:
        return 0.0, 1.0

    dr = np.diff(r)
    r_lag = r[:-1]

    if lags == 0 or n - 1 - lags < 10:
        X = r_lag.reshape(-1, 1)
        y = dr
    else:
        valid = n - 1 - lags
        X = np.empty((valid, 1 + lags), dtype=np.float64)
        X[:, 0] = r_lag[lags:]
        for k in range(1, lags + 1):
            # dr lag k: shift right by k posiciones
            X[:, k] = dr[lags - k : len(dr) - k]
        y = dr[lags:]

    XtX = X.T @ X
    Xty = X.T @ y
    try:
        coef = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return 0.0, 1.0

    gamma_hat = float(coef[0])
    resid = y - X @ coef
    dof = max(X.shape[0] - X.shape[1], 1)
    sigma2 = float((resid * resid).sum() / dof)

    try:
        cov = sigma2 * np.linalg.inv(XtX)
        se_gamma = float(math.sqrt(max(cov[0, 0], 1e-30)))
    except np.linalg.LinAlgError:
        return 0.0, 1.0

    if se_gamma < 1e-15:
        return 0.0, 1.0

    adf_stat = gamma_hat / se_gamma
    return adf_stat, _adf_pvalue(adf_stat)


def _rolling_beta(log_a: np.ndarray, log_b: np.ndarray, window: int) -> np.ndarray:
    """
    β rolling de ln(A) ~ ln(B) con ventana móvil, vectorizado vía cumsum.

    Para cada ventana terminando en posición t:
        β_t = [n·Σ(ab) − Σa·Σb] / [n·Σ(b²) − (Σb)²]
    """
    n = log_a.size
    if window > n:
        return np.array([], dtype=np.float64)

    out_len = n - window + 1
    out = np.empty(out_len, dtype=np.float64)

    csum_a = np.concatenate([[0.0], np.cumsum(log_a)])
    csum_b = np.concatenate([[0.0], np.cumsum(log_b)])
    csum_bb = np.concatenate([[0.0], np.cumsum(log_b * log_b)])
    csum_ab = np.concatenate([[0.0], np.cumsum(log_a * log_b)])

    for i in range(out_len):
        end = i + window
        sa = csum_a[end] - csum_a[i]
        sb = csum_b[end] - csum_b[i]
        sbb = csum_bb[end] - csum_bb[i]
        sab = csum_ab[end] - csum_ab[i]
        denom = window * sbb - sb * sb
        if abs(denom) < 1e-12:
            out[i] = 0.0
        else:
            out[i] = (window * sab - sa * sb) / denom
    return out


def fit_cointegration_ols(log_prices_a: np.ndarray,
                          log_prices_b: np.ndarray,
                          rolling_window: Optional[int] = None,
                          break_threshold_sigma: float = 2.0) -> CointegrationParams:
    """
    Engle-Granger 2-step:
        1. OLS:  ln(A) = α + β·ln(B) + ε
        2. ADF sobre ε.

    Parámetros
    ----------
    log_prices_a, log_prices_b : np.ndarray
        Log-precios (no precios crudos). ≥100 puntos, misma longitud.
    rolling_window : Optional[int]
        Si se especifica, calcula β rolling para medir estabilidad y
        detectar break estructural. None = no rolling check.
    break_threshold_sigma : float
        Si último β rolling se desvía >Nσ del β medio, marcamos break.

    Notas operativas
    ----------------
    Orden de variables importa:
        - A = asset dependiente (donde quieres alpha)
        - B = asset hedger (más líquido)
    β NO es simétrico ante swap A↔B salvo homocedasticidad perfecta.

    Sugerencias de pares:
        - Crypto:  A=ETH/USDT, B=BTC/USDT
        - Equity:  A=NVDA,    B=SMH (ETF semiconductores)
        - Cross:   A=COIN,    B=BTC/USDT  (CUIDADO: trading hours mismatch)
    """
    la = np.asarray(log_prices_a, dtype=np.float64)
    lb = np.asarray(log_prices_b, dtype=np.float64)

    if la.shape != lb.shape:
        raise ValueError(f"Shapes desiguales: A={la.shape}, B={lb.shape}")
    if la.size < 100:
        raise ValueError(f"Cointegración requiere ≥100 puntos, recibido {la.size}")
    if not (np.all(np.isfinite(la)) and np.all(np.isfinite(lb))):
        raise ValueError("Cointegración recibió NaN/Inf en log-precios")

    # Paso 1: OLS cointegrante
    alpha, beta = _ols_simple(lb, la)

    # Paso 2: spread y ADF
    spread = la - beta * lb - alpha
    spread_std = float(np.std(spread, ddof=1))
    adf_stat, p_value = _adf_test(spread, lags=1)

    # Verdict base
    if p_value < 0.05:
        verdict = CointegrationVerdict.COINTEGRATED
    elif p_value < 0.10:
        verdict = CointegrationVerdict.BORDERLINE
    else:
        verdict = CointegrationVerdict.NOT_COINTEGRATED

    # Rolling β: detección de structural break
    beta_rolling_std = 0.0
    if rolling_window is not None and rolling_window >= 30 and la.size >= 2 * rolling_window:
        betas = _rolling_beta(la, lb, window=rolling_window)
        if betas.size >= 5:
            mean_beta = float(np.mean(betas))
            std_beta = float(np.std(betas, ddof=1))
            beta_rolling_std = std_beta
            if std_beta > 1e-9:
                deviation_sigma = abs(betas[-1] - mean_beta) / std_beta
                if deviation_sigma > break_threshold_sigma:
                    verdict = CointegrationVerdict.STRUCTURAL_BREAK

    return CointegrationParams(
        beta=beta, alpha=alpha,
        adf_statistic=adf_stat, adf_pvalue=p_value,
        spread_std=spread_std, verdict=verdict,
        beta_rolling_std=beta_rolling_std,
    )


def compute_pair_spread(log_prices_a: np.ndarray, log_prices_b: np.ndarray,
                        params: CointegrationParams) -> np.ndarray:
    """
    Spread cointegrado: X_t = ln(A_t) − β·ln(B_t) − α.

    El OU/Kalman se aplica directamente sobre este spread sin modificaciones,
    porque el spread es univariante. Esa es la elegancia del enfoque
    Engle-Granger: reduce el problema bivariate a uno univariante donde
    todas las herramientas v1 funcionan.
    """
    la = np.asarray(log_prices_a, dtype=np.float64)
    lb = np.asarray(log_prices_b, dtype=np.float64)
    if la.shape != lb.shape:
        raise ValueError(f"Shapes desiguales: A={la.shape}, B={lb.shape}")
    return la - params.beta * lb - params.alpha


def pair_spread_zscore(latest_log_a: float, latest_log_b: float,
                       coint: CointegrationParams,
                       ou: Optional[OUParams] = None) -> float:
    """
    Z-score del spread actual. Si se provee `ou` (ajustado sobre la serie
    histórica del spread), usa distribución estacionaria de OU. Sino, usa
    spread_std in-sample.

    Esta firma es la que el RiskManager consume para señales pair-trading.
    """
    current_spread = latest_log_a - coint.beta * latest_log_b - coint.alpha
    if ou is not None and ou.is_operable:
        return ou_zscore(current_spread, ou)
    if coint.spread_std < 1e-12:
        return 0.0
    return current_spread / coint.spread_std   # media in-sample ≈ 0 por OLS


# =============================================================================
# Self-tests (correr con `python engine_math.py`)
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    print("=" * 72)
    print("v1 RETAINED: OU + Kalman + GARCH")
    print("=" * 72)

    # OU
    N = 5000; dt_test = 1.0 / 252
    theta_true, mu_true, sigma_true = 2.0, 100.0, 0.5
    x = np.empty(N); x[0] = 100.0
    for i in range(1, N):
        x[i] = x[i-1] + theta_true * (mu_true - x[i-1]) * dt_test + \
               sigma_true * math.sqrt(dt_test) * rng.standard_normal()
    fit = fit_ornstein_uhlenbeck(x, dt=dt_test)
    print(f"  OU:     θ̂={fit.theta:.3f} (true 2.0), μ̂={fit.mu:.2f}, σ̂={fit.sigma:.3f}")

    # Kalman
    true_state = np.cumsum(rng.standard_normal(1000) * 0.1) + 100
    obs = true_state + rng.standard_normal(1000) * 0.5
    sm = kalman_smooth_batch(obs, Q=0.01, R=0.25)
    rmse_raw = math.sqrt(np.mean((obs - true_state)**2))
    rmse_kf = math.sqrt(np.mean((sm - true_state)**2))
    print(f"  Kalman: RMSE {rmse_raw:.3f} → {rmse_kf:.3f} ({(1-rmse_kf/rmse_raw)*100:.1f}% reducción)")

    # GARCH
    omega_t, alpha_t, beta_t = 1e-5, 0.10, 0.85
    N = 3000
    eps = np.empty(N); sig2 = np.empty(N)
    sig2[0] = omega_t / (1 - alpha_t - beta_t)
    eps[0] = math.sqrt(sig2[0]) * rng.standard_normal()
    for i in range(1, N):
        sig2[i] = omega_t + alpha_t * eps[i-1]**2 + beta_t * sig2[i-1]
        eps[i] = math.sqrt(sig2[i]) * rng.standard_normal()
    g = fit_garch(eps)
    print(f"  GARCH:  α̂={g.alpha:.3f} (true 0.10), β̂={g.beta:.3f} (true 0.85), persistence={g.persistence:.4f}")

    print()
    print("=" * 72)
    print("v2 NEW: Kalman gap-handler (overnight equity close)")
    print("=" * 72)
    state = init_kalman(100.0, obs_variance=0.04, process_variance=1e-3)
    for _ in range(50):
        state.update(100.0 + rng.standard_normal() * 0.2)
    P_pre = state.P
    print(f"  Pre-gap:  x={state.x:.3f}, P={state.P:.5f}")
    state.update_after_gap(102.0, gap_seconds=16*3600,
                           gap_volatility_per_sqrt_sec=0.001)
    print(f"  Post-gap: x={state.x:.3f}, P={state.P:.5f}")
    print(f"  Δ P attributable a 16h gap @ vol 0.001/√s: "
          f"esperado ≈ {(0.001**2)*16*3600:.5f}")

    # Comparativa: gap correctamente manejado vs gap ignorado
    state_naive = init_kalman(100.0, obs_variance=0.04, process_variance=1e-3)
    for _ in range(50):
        state_naive.update(100.0 + rng.standard_normal() * 0.2)
    state_naive.update(102.0)   # tratamiento naive: ignora el gap
    print(f"  Naive (sin gap): x={state_naive.x:.3f} (sobre-confía en pre-gap)")
    print(f"  Con gap:         x={state.x:.3f} (absorbe más la observación 102)")

    print()
    print("=" * 72)
    print("v2 NEW: Cointegración Engle-Granger")
    print("=" * 72)

    n = 500
    # Test 1: par genuinamente cointegrado
    log_b = np.cumsum(rng.standard_normal(n) * 0.01) + math.log(100)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i-1] + rng.standard_normal() * 0.005   # AR(1) estacionario
    log_a = 0.5 + 1.3 * log_b + noise
    c = fit_cointegration_ols(log_a, log_b, rolling_window=60)
    print(f"  Par cointegrado (true β=1.30, α=0.50):")
    print(f"    β̂={c.beta:.4f}, α̂={c.alpha:.4f}")
    print(f"    ADF stat={c.adf_statistic:.3f}, p={c.adf_pvalue:.4f}")
    print(f"    Verdict: {c.verdict.value}")
    print(f"    β rolling std: {c.beta_rolling_std:.4f}")

    # Test 2: dos random walks independientes (NO cointegrados)
    log_x = np.cumsum(rng.standard_normal(n) * 0.01) + math.log(100)
    log_y = np.cumsum(rng.standard_normal(n) * 0.01) + math.log(50)
    c2 = fit_cointegration_ols(log_x, log_y)
    print(f"  Par NO cointegrado:")
    print(f"    ADF stat={c2.adf_statistic:.3f}, p={c2.adf_pvalue:.4f}")
    print(f"    Verdict: {c2.verdict.value}")

    # Test 3: structural break — β shifts a la mitad de la serie
    log_b3 = np.cumsum(rng.standard_normal(n) * 0.01) + math.log(100)
    noise3 = np.zeros(n)
    for i in range(1, n):
        noise3[i] = 0.7 * noise3[i-1] + rng.standard_normal() * 0.005
    log_a3 = np.empty(n)
    log_a3[:n//2] = 0.5 + 1.3 * log_b3[:n//2] + noise3[:n//2]
    log_a3[n//2:] = 0.5 + 2.0 * log_b3[n//2:] + noise3[n//2:]   # β salta
    c3 = fit_cointegration_ols(log_a3, log_b3, rolling_window=60)
    print(f"  Par con structural break (β salta de 1.3 → 2.0):")
    print(f"    Verdict: {c3.verdict.value}")
    print(f"    β rolling std: {c3.beta_rolling_std:.4f} (alto = inestable)")

    # Test 4: spread + OU + z-score
    spread_series = compute_pair_spread(log_a, log_b, c)
    ou_spread = fit_ornstein_uhlenbeck(spread_series, dt=1.0/(252*24*60))
    z_simple = pair_spread_zscore(log_a[-1], log_b[-1], c)
    z_via_ou = pair_spread_zscore(log_a[-1], log_b[-1], c, ou=ou_spread)
    print(f"  Spread analysis del par cointegrado:")
    print(f"    OU sobre el spread: θ̂={ou_spread.theta:.4f}, half_life={ou_spread.half_life:.5f}")
    print(f"    z-score (spread_std):      {z_simple:.3f}")
    print(f"    z-score (OU stationary):   {z_via_ou:.3f}")
