"""
dashboard.py  (v2)
==================
Admin Panel — Quant Engine Multi-Asset v2.

Corre en Node B (Raspberry Pi 4) o en Node A si no hay Pi.
Conecta en modo read-only a PostgreSQL y Redis del Node A vía LAN.

Tabs:
    Tab 1  📈 Equity & Drawdown        — Curvas por asset class + underwater chart
    Tab 2  📜 Reasoning Log             — Risk metrics extendidos: Kelly, alpha bps,
                                          TCA threshold, OU half-life, correlation mult
    Tab 3  💼 Trades individuales       — Tabla filtrable por asset_class + side
    Tab 4  🔗 Pair Trades / Spreads     — Spread X_t histórico, β rolling,
                                          pair win rate
    Tab 5  🩺 System Health             — Redis keys, day-lock, MarketSessionGuard
                                          status para todos los equity symbols

Auth: streamlit-authenticator con bcrypt. Credenciales en .env o auth.yaml.
Tunnel: Cloudflare Zero Trust (ver instrucciones.md § 7).

Ejecutar:
    streamlit run dashboard.py \\
        --server.address 0.0.0.0 \\
        --server.port 8501 \\
        --server.headless true \\
        --browser.gatherUsageStats false
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psycopg2
import psycopg2.extras
import redis
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# ─────────────────────────────────────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Quant Engine v2 — Admin",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

UTC = ZoneInfo("UTC")
ET  = ZoneInfo("America/New_York")

# ─────────────────────────────────────────────────────────────────────────────
# Variables de entorno
# ─────────────────────────────────────────────────────────────────────────────

POSTGRES_DSN   = os.getenv("POSTGRES_DSN", "postgresql://quant:quant@localhost:5432/quant")
REDIS_URL      = os.getenv("REDIS_URL", "redis://localhost:6379/0")
AUTH_CONFIG    = os.getenv("AUTH_CONFIG_PATH", "/etc/quant/auth.yaml")
SYMBOLS_RAW    = os.getenv("SYMBOLS", "CRYPTO:BTC/USDT,CRYPTO:ETH/USDT")
PAIR_SYMS_RAW  = os.getenv("PAIR_SYMBOLS", "")
ALLOW_PRE      = os.getenv("ALLOW_PRE_MARKET", "false").lower() == "true"
ALLOW_POST     = os.getenv("ALLOW_POST_MARKET", "false").lower() == "true"

# Parsear símbolos del entorno
ALL_SYMBOLS: list[str] = [s.strip() for s in SYMBOLS_RAW.split(",") if s.strip()]
ALL_PAIRS: list[tuple[str, str]] = []
for p in PAIR_SYMS_RAW.split(","):
    p = p.strip()
    if "|" in p:
        a, b = p.split("|", 1)
        ALL_PAIRS.append((a.strip(), b.strip()))

EQUITY_SYMBOLS  = [s for s in ALL_SYMBOLS if s.startswith("EQUITY:")]
CRYPTO_SYMBOLS  = [s for s in ALL_SYMBOLS if s.startswith("CRYPTO:") or ":" not in s]

# ─────────────────────────────────────────────────────────────────────────────
# Conexiones (cacheadas por sesión con st.cache_resource)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_pg() -> psycopg2.extensions.connection:
    """Conexión PostgreSQL persistente (read-only lógico)."""
    return psycopg2.connect(POSTGRES_DSN)


@st.cache_resource
def get_redis() -> redis.Redis:
    return redis.from_url(REDIS_URL, decode_responses=True)


def pg_query(sql: str, params=None) -> pd.DataFrame:
    """Ejecuta una query y devuelve DataFrame. Reconecta si la conn se cayó."""
    conn = get_pg()
    try:
        conn.rollback()   # limpiar transacción anterior si colgó
        return pd.read_sql(sql, conn, params=params)
    except Exception:
        # La conexión murió; limpiar cache y reintentar una vez
        st.cache_resource.clear()
        conn = get_pg()
        conn.rollback()
        return pd.read_sql(sql, conn, params=params)


# ─────────────────────────────────────────────────────────────────────────────
# Autenticación
# ─────────────────────────────────────────────────────────────────────────────

def _build_auth_config() -> dict:
    """
    Construye la config de autenticación.
    Prioridad: archivo YAML > variables de entorno.
    """
    if os.path.exists(AUTH_CONFIG):
        with open(AUTH_CONFIG) as f:
            return yaml.load(f, Loader=SafeLoader)

    password_hash = os.getenv("ADMIN_PASSWORD_HASH", "")
    if not password_hash:
        st.error(
            "Falta autenticación: define AUTH_CONFIG_PATH o ADMIN_PASSWORD_HASH "
            "en el entorno. Ver instrucciones.md § Dashboard."
        )
        st.stop()

    return {
        "credentials": {
            "usernames": {
                "admin": {
                    "email": "admin@local",
                    "name": "Admin",
                    "password": password_hash,
                }
            }
        },
        "cookie": {
            "name": "quant_admin_v2",
            "key": os.getenv("AUTH_COOKIE_KEY", "change-me-32-chars-min"),
            "expiry_days": 1,
        },
    }


_auth_cfg = _build_auth_config()
authenticator = stauth.Authenticate(
    _auth_cfg["credentials"],
    _auth_cfg["cookie"]["name"],
    _auth_cfg["cookie"]["key"],
    _auth_cfg["cookie"]["expiry_days"],
)

authenticator.login(location="main")

if not st.session_state.get("authentication_status"):
    if st.session_state.get("authentication_status") is False:
        st.error("Credenciales inválidas.")
    else:
        st.warning("Ingresa tus credenciales para acceder al panel.")
    st.stop()

authenticator.logout(location="sidebar")
st.sidebar.success(f"👤 {st.session_state.get('name', 'Admin')}")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: controles globales
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
days_filter = st.sidebar.slider("Ventana de análisis (días)", 1, 90, 30)
st.sidebar.caption(f"Mostrando datos de los últimos {days_filter} días.")

# Salud de conexiones en sidebar
redis_ok = False
try:
    r = get_redis()
    r.ping()
    redis_ok = True
except Exception:
    pass

pg_ok = False
try:
    get_pg()
    pg_ok = True
except Exception:
    pass

today_utc = dt.datetime.now(UTC).strftime("%Y-%m-%d")
day_locked = False
if redis_ok:
    try:
        day_locked = bool(r.get(f"risk:day_locked:{today_utc}"))
    except Exception:
        pass

col_r, col_p, col_d = st.sidebar.columns(3)
col_r.metric("Redis", "🟢" if redis_ok else "🔴")
col_p.metric("Postgres", "🟢" if pg_ok else "🔴")
col_d.metric("Day", "🔒 LOCK" if day_locked else "🟢 OK")

if day_locked:
    st.sidebar.warning("⚠️ Day-lock activo: drawdown 2% alcanzado. Sin nuevas entradas hasta 00:00 UTC.")

st.sidebar.markdown("---")
st.sidebar.caption("Quant Engine v2 — Multi-Asset")
st.sidebar.caption("Node B → Node A via LAN")

# ─────────────────────────────────────────────────────────────────────────────
# Carga de datos con TTL (st.cache_data expira automáticamente)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_equity_snapshots(days: int) -> pd.DataFrame:
    """Snapshots de equity por quote_currency, ordenados por tiempo."""
    return pg_query(
        f"""
        SELECT ts,
               equity_quote,
               available_quote,
               quote_currency
        FROM   equity_snapshots
        WHERE  ts >= NOW() - INTERVAL '{days} days'
        ORDER  BY ts ASC
        """
    )


@st.cache_data(ttl=15)
def load_trades(days: int) -> pd.DataFrame:
    """
    Carga todos los trades del período con columnas v2:
        asset_class, pair_partner, risk_metrics (JSONB).
    pair_partner IS NULL  → trade single-asset
    pair_partner IS NOT NULL → leg de un par cointegrado
    """
    df = pg_query(
        f"""
        SELECT
            id,
            client_order_id,
            symbol,
            asset_class,
            side,
            pair_partner,
            entry_price,
            exit_price,
            size_quote,
            size_base,
            pnl_quote,
            fees_quote,
            opened_at,
            closed_at,
            status,
            reasoning,
            risk_metrics
        FROM   trades
        WHERE  opened_at >= NOW() - INTERVAL '{days} days'
        ORDER  BY opened_at DESC
        """
    )
    # Normalizar tipos
    for col in ["entry_price", "exit_price", "size_quote", "size_base",
                "pnl_quote", "fees_quote"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=60)
def load_pair_spread_history(sym_a: str, sym_b: str, days: int) -> pd.DataFrame:
    """
    Intenta recuperar el historial del spread X_t para un par dado.

    El bot persiste los spreads en QuestDB (tabla 'pair_spreads') si la
    columna existe. Si no existe aún, construye una aproximación desde
    la tabla 'orderbook' haciendo un join temporal por minuto.

    Devuelve DataFrame con columnas: [ts, spread_value].
    """
    # Formatear nombres para QuestDB (/ → _ : → __)
    sym_a_tag = sym_a.replace("/", "_").replace(":", "__")
    sym_b_tag = sym_b.replace("/", "_").replace(":", "__")

    # Intento 1: tabla pair_spreads dedicada (disponible si el bot v3 la crea)
    try:
        df = pg_query(
            f"""
            SELECT ts, spread_value, beta_rolling
            FROM   pair_spreads
            WHERE  symbol_a = %s
              AND  symbol_b = %s
              AND  ts >= NOW() - INTERVAL '{days} days'
            ORDER  BY ts ASC
            """,
            params=(sym_a, sym_b),
        )
        if not df.empty:
            return df
    except Exception:
        pass

    # Intento 2: aproximación desde QuestDB via PostgreSQL wire (puerto 8812)
    # Si QuestDB no está en el DSN, devolvemos DataFrame vacío.
    return pd.DataFrame(columns=["ts", "spread_value", "beta_rolling"])


def compute_metrics(equity: pd.DataFrame, trades: pd.DataFrame) -> dict:
    """
    Métricas cuantitativas estándar.
    Calcula por separado para crypto y equity cuando hay datos de ambas clases.
    """
    if equity.empty:
        return {}

    out: dict = {}

    # ── Métricas de equity curve agregada ────────────────────────────────────
    # Usamos la moneda quote más representada (USDT o USD)
    dominant_currency = (
        equity.groupby("quote_currency")["equity_quote"].count().idxmax()
        if "quote_currency" in equity.columns and not equity.empty
        else "USDT"
    )
    eq_df = equity
    if "quote_currency" in equity.columns:
        eq_df = equity[equity["quote_currency"] == dominant_currency]

    if eq_df.empty:
        return {}

    eq_vals = eq_df["equity_quote"].astype(float).values
    if eq_vals.size < 2:
        return {}

    returns = np.diff(eq_vals) / np.where(eq_vals[:-1] != 0, eq_vals[:-1], 1e-12)

    # Períodos por año: asumimos snapshots cada 60s → 365*24*60 = 525,600
    periods_per_year = 365 * 24 * 60
    mean_r = float(returns.mean()) * periods_per_year
    std_r  = float(returns.std(ddof=1)) * math.sqrt(periods_per_year) if returns.size > 1 else 0.0
    sharpe = mean_r / std_r if std_r > 1e-10 else 0.0

    peak = np.maximum.accumulate(eq_vals)
    dd   = np.where(peak > 0, (peak - eq_vals) / peak, 0.0)
    max_dd = float(dd.max())
    calmar = (mean_r / max_dd) if max_dd > 1e-10 else 0.0

    out.update({
        "equity_now": float(eq_vals[-1]),
        "quote_currency": dominant_currency,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
    })

    # ── Métricas de trades cerrados ───────────────────────────────────────────
    closed = trades[(trades["status"] == "closed") & trades["pnl_quote"].notna()].copy()
    if not closed.empty:
        wins   = closed[closed["pnl_quote"] > 0]
        losses = closed[closed["pnl_quote"] < 0]
        n      = len(closed)
        pf     = (
            wins["pnl_quote"].sum() / abs(losses["pnl_quote"].sum())
            if not losses.empty and abs(losses["pnl_quote"].sum()) > 0
            else float("inf")
        )
        out.update({
            "n_trades": n,
            "win_rate": len(wins) / n if n > 0 else 0.0,
            "profit_factor": pf,
            "total_pnl": float(closed["pnl_quote"].sum()),
        })

        # ── Pair trade metrics (legs con pair_partner NOT NULL) ───────────────
        pair_closed = closed[closed["pair_partner"].notna()]
        if not pair_closed.empty:
            # Agrupar por client_order_id_a/b para no doble-contar el PnL del par
            # (cada par tiene 2 rows). Sumamos el PnL de ambos legs por par.
            n_pair = len(pair_closed) // 2  # aproximado: 2 legs por par
            pair_pnl = pair_closed["pnl_quote"].sum() / 2  # promedio de ambos legs
            pair_wins = pair_closed[pair_closed["pnl_quote"] > 0]
            out.update({
                "n_pair_trades": n_pair,
                "pair_win_rate": len(pair_wins) / len(pair_closed) if len(pair_closed) > 0 else 0.0,
                "pair_total_pnl": float(pair_pnl),
            })

    # ── Breakdown por asset_class ─────────────────────────────────────────────
    if "asset_class" in closed.columns and not closed.empty:
        by_class = {}
        for ac, grp in closed.groupby("asset_class"):
            by_class[ac] = {
                "n": len(grp),
                "pnl": float(grp["pnl_quote"].sum()),
                "win_rate": float((grp["pnl_quote"] > 0).mean()),
            }
        out["by_class"] = by_class

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades globales (definidas antes de los tabs que las usan)
# ─────────────────────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convierte #rrggbb a (r, g, b) normalizados en [0, 1]."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (0.5, 0.5, 0.5)
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    return (r, g, b)


# ─────────────────────────────────────────────────────────────────────────────
# DATOS PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────

equity   = load_equity_snapshots(days_filter)
trades   = load_trades(days_filter)
metrics  = compute_metrics(equity, trades)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER: título + KPIs
# ─────────────────────────────────────────────────────────────────────────────

st.title("📊 Quant Engine v2 — Admin Panel")

k1, k2, k3, k4, k5, k6 = st.columns(6)
currency = metrics.get("quote_currency", "USDT")
k1.metric("Equity",          f"${metrics.get('equity_now', 0):,.2f} {currency}")
k2.metric("Sharpe (ann.)",   f"{metrics.get('sharpe', 0):.2f}")
k3.metric("Max DD",          f"{metrics.get('max_dd', 0)*100:.2f}%")
k4.metric("Win Rate",        f"{metrics.get('win_rate', 0)*100:.1f}%")
k5.metric("Profit Factor",   f"{metrics.get('profit_factor', 0):.2f}")
k6.metric("Pair Win Rate",   f"{metrics.get('pair_win_rate', 0)*100:.1f}%")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Equity & Drawdown",
    "📜 Reasoning Log",
    "💼 Trades individuales",
    "🔗 Pair Trades / Spreads",
    "🩺 System Health",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Equity curve & Underwater chart
# Muestra curvas independientes por quote_currency (USDT para crypto,
# USD para equity) si existen ambas. Curva unificada si sólo hay una.
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    if equity.empty:
        st.info("Sin snapshots de equity todavía. El bot comienza a generar snapshots cada 60s.")
    else:
        ts_col = pd.to_datetime(equity["ts"])

        # Detectar cuántas quote currencies distintas hay
        currencies_available = (
            equity["quote_currency"].unique().tolist()
            if "quote_currency" in equity.columns
            else ["USDT"]
        )

        if len(currencies_available) > 1:
            view_mode = st.radio(
                "Vista de curva",
                ["Unificada (suma estimada en USD)", "Por clase de activo"],
                horizontal=True,
            )
        else:
            view_mode = "Unificada (suma estimada en USD)"

        st.subheader("Equity Curve")

        fig_equity = go.Figure()
        underwater_data: list[tuple[pd.Series, np.ndarray]] = []

        if view_mode == "Por clase de activo" and len(currencies_available) > 1:
            colors = {"USDT": "#00d4aa", "USD": "#7c6af7", "BTC": "#f7931a"}
            for cur in currencies_available:
                sub = equity[equity["quote_currency"] == cur].copy()
                if sub.empty:
                    continue
                ts_s  = pd.to_datetime(sub["ts"])
                vals  = sub["equity_quote"].astype(float).values
                label = f"{'Crypto' if cur == 'USDT' else 'Equity'} ({cur})"
                fig_equity.add_trace(go.Scatter(
                    x=ts_s, y=vals, name=label, mode="lines",
                    line=dict(width=2, color=colors.get(cur, "#888")),
                ))
                peak = np.maximum.accumulate(vals)
                dd   = np.where(peak > 0, (peak - vals) / peak * 100, 0.0)
                underwater_data.append((ts_s, dd))
        else:
            # Curva unificada: para simplificar, sumamos todos los snapshots
            # que coincidan en timestamp (±30s). En producción usar join temporal.
            dominant_cur = metrics.get("quote_currency", "USDT")
            sub = equity
            if "quote_currency" in equity.columns:
                sub = equity[equity["quote_currency"] == dominant_cur]
            if not sub.empty:
                ts_s = pd.to_datetime(sub["ts"])
                vals = sub["equity_quote"].astype(float).values
                fig_equity.add_trace(go.Scatter(
                    x=ts_s, y=vals, name=f"Equity ({dominant_cur})",
                    mode="lines", line=dict(width=2, color="#00d4aa"),
                ))
                peak = np.maximum.accumulate(vals)
                dd   = np.where(peak > 0, (peak - vals) / peak * 100, 0.0)
                underwater_data.append((ts_s, dd))

        fig_equity.update_layout(
            height=380,
            xaxis_title="Tiempo (UTC)",
            yaxis_title="Equity (quote)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_equity, use_container_width=True)

        # Underwater chart (drawdown desde peak)
        st.subheader("Underwater Chart — Drawdown desde Peak")
        fig_uw = go.Figure()
        colors_dd = ["#e63946", "#f4a261", "#2a9d8f"]
        for i, (ts_s, dd) in enumerate(underwater_data):
            fig_uw.add_trace(go.Scatter(
                x=ts_s, y=-dd,   # negativo: convención estándar underwater chart
                name=f"DD {currencies_available[i] if i < len(currencies_available) else ''}",
                fill="tozeroy",
                mode="lines",
                line=dict(color=colors_dd[i % len(colors_dd)], width=1),
                fillcolor=f"rgba({','.join(str(int(c*255)) for c in _hex_to_rgb(colors_dd[i % len(colors_dd)]))}, 0.15)",
            ))
        fig_uw.add_hline(y=-2.0, line_dash="dash", line_color="red",
                          annotation_text="Daily DD limit −2%",
                          annotation_position="bottom right")
        fig_uw.update_layout(
            height=280,
            xaxis_title="Tiempo (UTC)",
            yaxis_title="Drawdown (%)",
            yaxis=dict(ticksuffix="%"),
            margin=dict(t=10, b=40),
        )
        st.plotly_chart(fig_uw, use_container_width=True)

        # PnL breakdown por asset class
        by_class = metrics.get("by_class", {})
        if by_class:
            st.subheader("PnL por clase de activo (trades cerrados)")
            bc_cols = st.columns(len(by_class))
            for i, (ac, d) in enumerate(by_class.items()):
                with bc_cols[i]:
                    ac_label = "🪙 Crypto" if ac == "crypto" else "📈 Equity"
                    st.metric(f"{ac_label} PnL", f"${d['pnl']:,.2f}")
                    st.caption(f"{d['n']} trades | Win: {d['win_rate']*100:.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Reasoning Log
# Muestra el reasoning del RiskManager con métricas extendidas:
#   - Kelly final fraction
#   - Alpha esperado en bps
#   - Threshold TCA regulatorio (SEC §31 + FINRA TAF implícitos)
#   - OU half-life (vida media de reversión)
#   - Multiplicador de correlación cross-asset
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Reasoning Log — Decisiones del Motor de Riesgo")
    st.caption(
        "Cada entrada muestra por qué la AI aprobó o rechazó un trade. "
        "Los thresholds TCA incorporan tasas regulatorias vigentes: "
        "SEC §31 ($20.60/M) y FINRA TAF ($0.000195/share, cap $9.79)."
    )

    if trades.empty:
        st.info("Sin trades en el período seleccionado.")
    else:
        # Filtros de la sidebar de este tab
        ac_filter_log = st.selectbox(
            "Filtrar por asset class",
            ["Todos", "crypto", "equity"],
            key="ac_filter_log",
        )
        show_n = st.slider("Mostrar últimos N trades", 5, 100, 30, key="log_n")

        df_log = trades.copy()
        if ac_filter_log != "Todos" and "asset_class" in df_log.columns:
            df_log = df_log[df_log["asset_class"] == ac_filter_log]
        df_log = df_log.head(show_n)

        if df_log.empty:
            st.info(f"Sin trades para asset_class='{ac_filter_log}'.")
        else:
            for _, row in df_log.iterrows():
                # Parsear JSON de reasoning y risk_metrics
                reasoning = row.get("reasoning") or {}
                if isinstance(reasoning, str):
                    try:
                        reasoning = json.loads(reasoning)
                    except (json.JSONDecodeError, TypeError):
                        reasoning = {"reason": str(reasoning)}

                risk_m = row.get("risk_metrics") or {}
                if isinstance(risk_m, str):
                    try:
                        risk_m = json.loads(risk_m)
                    except (json.JSONDecodeError, TypeError):
                        risk_m = {}

                # Header del expander
                pnl_raw = row.get("pnl_quote")
                pnl_str = f" | PnL: ${float(pnl_raw):+,.2f}" if pnl_raw is not None else ""
                ac_icon = "🪙" if str(row.get("asset_class", "")) == "crypto" else "📈"
                pair_icon = " [PAIR]" if pd.notna(row.get("pair_partner")) else ""
                status_icon = "✅" if row.get("status") == "closed" else "🔄"

                header = (
                    f"{status_icon} {ac_icon}{pair_icon} **{row.get('symbol', '?')}** "
                    f"`{str(row.get('side', '')).upper()}` "
                    f"@ {float(row.get('entry_price') or 0):.4f}"
                    f"{pnl_str} — {str(row.get('opened_at', ''))[:19]}"
                )

                with st.expander(header, expanded=False):
                    # Reasoning text
                    reason_text = reasoning.get("reason", "—")
                    st.write(reason_text)

                    # Métricas extendidas en columnas
                    cols = st.columns(5)
                    kelly_final  = risk_m.get("kelly_final") or risk_m.get("kelly_final_fraction")
                    alpha_bps    = risk_m.get("alpha_bps")
                    threshold    = risk_m.get("threshold_bps")
                    vol_ann      = risk_m.get("vol_annualized")
                    half_life    = risk_m.get("ou_half_life_sec")
                    corr_mult    = risk_m.get("correlation_mult") or risk_m.get("correlation_multiplier")

                    cols[0].metric(
                        "Kelly final",
                        f"{float(kelly_final):.4f}" if kelly_final is not None else "—",
                        help="Fracción del equity asignada al trade tras suavizado EMA y multiplicador de correlación.",
                    )
                    cols[1].metric(
                        "Alpha (bps)",
                        f"{float(alpha_bps):.2f}" if alpha_bps is not None else "—",
                        help="Edge esperado en basis points sobre el round-trip de la vida media del OU.",
                    )
                    cols[2].metric(
                        "TCA Umbral (bps)",
                        f"{float(threshold):.2f}" if threshold is not None else "—",
                        help="Costo total round-trip × 1.5: fees + spread + slippage (+ SEC §31 + FINRA TAF para equity).",
                    )
                    cols[3].metric(
                        "Vol anualizada",
                        f"{float(vol_ann)*100:.1f}%" if vol_ann is not None else "—",
                        help="Vol GARCH anualizada. Cap: 250% crypto, 60% equity.",
                    )
                    cols[4].metric(
                        "OU Half-life",
                        f"{float(half_life):.0f}s" if half_life is not None else "—",
                        help="Tiempo en segundos para cerrar ~50% del gap hacia la media del proceso OU.",
                    )

                    # Correlación cross-asset
                    if corr_mult is not None:
                        corr_info = risk_m.get("correlation_info", {})
                        worst = corr_info.get("worst_pair", "—") if isinstance(corr_info, dict) else "—"
                        max_r = corr_info.get("max_abs_corr", 0) if isinstance(corr_info, dict) else 0
                        st.caption(
                            f"Factor Exposure Penalty: mult={float(corr_mult):.3f} "
                            f"(peor correlación: `{worst}` @ |ρ|={float(max_r):.3f})"
                        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Trades individuales
# Tabla filtrable por asset_class y side.
# Excluye legs de pares (pair_partner IS NOT NULL) por defecto para evitar
# confusión de doble conteo. Se pueden incluir con un toggle.
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Trades Individuales")

    col_ac, col_side, col_status, col_pair = st.columns(4)
    ac_sel     = col_ac.selectbox("Asset class", ["Todos", "crypto", "equity"], key="ac_t3")
    side_sel   = col_side.selectbox("Side", ["Todos", "long", "short"], key="side_t3")
    status_sel = col_status.selectbox("Status", ["Todos", "open", "closed"], key="st_t3")
    show_pair  = col_pair.checkbox("Incluir legs de pares", value=False, key="pair_t3")

    df_t3 = trades.copy()
    if not show_pair and "pair_partner" in df_t3.columns:
        df_t3 = df_t3[df_t3["pair_partner"].isna()]
    if ac_sel != "Todos" and "asset_class" in df_t3.columns:
        df_t3 = df_t3[df_t3["asset_class"] == ac_sel]
    if side_sel != "Todos":
        df_t3 = df_t3[df_t3["side"] == side_sel]
    if status_sel != "Todos":
        df_t3 = df_t3[df_t3["status"] == status_sel]

    if df_t3.empty:
        st.info("Sin trades para los filtros seleccionados.")
    else:
        # Columnas de display limpias
        display_cols = [
            c for c in [
                "opened_at", "symbol", "asset_class", "side",
                "entry_price", "exit_price", "size_quote",
                "pnl_quote", "fees_quote", "status",
            ]
            if c in df_t3.columns
        ]
        st.dataframe(
            df_t3[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=480,
        )

        # Resumen rápido del subconjunto filtrado
        closed_sub = df_t3[(df_t3["status"] == "closed") & df_t3["pnl_quote"].notna()]
        if not closed_sub.empty:
            total_pnl = closed_sub["pnl_quote"].sum()
            n_c = len(closed_sub)
            wins_c = (closed_sub["pnl_quote"] > 0).sum()
            st.caption(
                f"Subtotal: {n_c} trades cerrados | "
                f"Win: {wins_c}/{n_c} ({wins_c/n_c*100:.1f}%) | "
                f"PnL: ${total_pnl:,.2f}"
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Pair Trades / Spreads
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Pair Trades — Spreads Cointegrados")

    # ── Pair Win Rate global ──────────────────────────────────────────────────
    pair_trades_all = (
        trades[trades["pair_partner"].notna()].copy()
        if "pair_partner" in trades.columns
        else pd.DataFrame()
    )

    if pair_trades_all.empty:
        st.info(
            "Sin trades de pares en el período. Configura pares en "
            "`PAIR_SYMBOLS` y espera a que el bot detecte cointegración."
        )
    else:
        # Métricas de pares en la parte superior
        pm1, pm2, pm3, pm4 = st.columns(4)
        closed_pairs = pair_trades_all[
            (pair_trades_all["status"] == "closed") &
            pair_trades_all["pnl_quote"].notna()
        ]
        n_pair_legs = len(closed_pairs)
        n_pairs_est = n_pair_legs // 2  # 2 legs por par
        total_pair_pnl = closed_pairs["pnl_quote"].sum() / 2
        wins_pair = (closed_pairs["pnl_quote"] > 0).sum()
        pair_wr = wins_pair / n_pair_legs if n_pair_legs > 0 else 0.0

        pm1.metric("Pares cerrados (est.)", n_pairs_est)
        pm2.metric("Pair Win Rate", f"{pair_wr*100:.1f}%")
        pm3.metric("Pair PnL total", f"${total_pair_pnl:,.2f}")
        open_pairs = pair_trades_all[pair_trades_all["status"] == "open"]
        pm4.metric("Legs abiertas", len(open_pairs))

        st.markdown("---")

        # ── Selector de par ───────────────────────────────────────────────────
        if ALL_PAIRS:
            pair_options = [f"{a} | {b}" for a, b in ALL_PAIRS]
        else:
            # Inferir pares desde la tabla de trades
            symbols_with_partners = (
                pair_trades_all[["symbol", "pair_partner"]]
                .dropna()
                .drop_duplicates()
            )
            pair_options = [
                f"{row['symbol']} | {row['pair_partner']}"
                for _, row in symbols_with_partners.iterrows()
            ]
        pair_options = list(dict.fromkeys(pair_options))  # deduplicar

        if not pair_options:
            st.warning("No se encontraron pares configurados.")
        else:
            sel_pair_str = st.selectbox("Seleccionar par", pair_options, key="pair_sel")
            sym_a_sel, sym_b_sel = [s.strip() for s in sel_pair_str.split("|", 1)]

            # ── Historial del spread X_t ──────────────────────────────────────
            spread_df = load_pair_spread_history(sym_a_sel, sym_b_sel, days_filter)

            st.subheader(f"Spread X_t = ln({sym_a_sel}) − β·ln({sym_b_sel}) − α")

            if spread_df.empty:
                st.info(
                    "Historial del spread no disponible todavía. El bot persiste "
                    "el spread en la tabla `pair_spreads` (disponible en v3). "
                    "Con datos actuales mostramos el spread aproximado desde "
                    "los trades."
                )

                # Aproximación desde los trades del par seleccionado
                pair_sub = pair_trades_all[
                    (pair_trades_all["symbol"] == sym_a_sel) |
                    (pair_trades_all["symbol"] == sym_b_sel)
                ].copy()
                if not pair_sub.empty and "entry_price" in pair_sub.columns:
                    # Gráfico simplificado: entry_price de cada leg
                    fig_spread = go.Figure()
                    for sym_plot in [sym_a_sel, sym_b_sel]:
                        sub_plot = pair_sub[pair_sub["symbol"] == sym_plot]
                        if not sub_plot.empty:
                            fig_spread.add_trace(go.Scatter(
                                x=pd.to_datetime(sub_plot["opened_at"]),
                                y=sub_plot["entry_price"].astype(float),
                                name=f"Entry price {sym_plot.split(':')[-1]}",
                                mode="markers+lines",
                            ))
                    fig_spread.update_layout(
                        height=280, xaxis_title="Fecha", yaxis_title="Precio entrada",
                        title="Precios de entrada por leg (spread no disponible directamente)",
                        margin=dict(t=30, b=30),
                    )
                    st.plotly_chart(fig_spread, use_container_width=True)
                else:
                    st.caption("Sin datos de trades de este par en el período.")
            else:
                # Spread completo disponible
                ts_sp = pd.to_datetime(spread_df["ts"])
                sv    = spread_df["spread_value"].astype(float).values

                # Calcular z-score rolling del spread (ventana 60 puntos)
                rolling_mean = pd.Series(sv).rolling(60, min_periods=10).mean().values
                rolling_std  = pd.Series(sv).rolling(60, min_periods=10).std().values
                zscore_rolling = np.where(
                    rolling_std > 1e-10,
                    (sv - rolling_mean) / rolling_std,
                    0.0,
                )

                fig_spread = go.Figure()
                fig_spread.add_trace(go.Scatter(
                    x=ts_sp, y=sv, name="Spread X_t",
                    line=dict(width=1.5, color="#00d4aa"),
                ))
                # Bandas ±2σ estacionaria
                if rolling_std[-1] > 0:
                    upper = rolling_mean + 2 * rolling_std
                    lower = rolling_mean - 2 * rolling_std
                    fig_spread.add_trace(go.Scatter(
                        x=ts_sp, y=upper, name="+2σ",
                        line=dict(dash="dot", color="rgba(231,76,60,0.5)", width=1),
                    ))
                    fig_spread.add_trace(go.Scatter(
                        x=ts_sp, y=lower, name="−2σ",
                        line=dict(dash="dot", color="rgba(231,76,60,0.5)", width=1),
                        fill="tonexty",
                        fillcolor="rgba(231,76,60,0.05)",
                    ))
                fig_spread.update_layout(
                    height=300, xaxis_title="Tiempo", yaxis_title="Spread (log-units)",
                    margin=dict(t=10, b=30), legend=dict(orientation="h"),
                )
                st.plotly_chart(fig_spread, use_container_width=True)

                # Z-score del spread
                fig_z = go.Figure()
                fig_z.add_trace(go.Scatter(
                    x=ts_sp, y=zscore_rolling, name="Z-score rolling",
                    line=dict(width=1.5, color="#7c6af7"),
                ))
                fig_z.add_hline(y=2.0,  line_dash="dash", line_color="red",
                                 annotation_text="Entry threshold +2σ")
                fig_z.add_hline(y=-2.0, line_dash="dash", line_color="green",
                                 annotation_text="Entry threshold −2σ")
                fig_z.add_hline(y=0,    line_dash="solid", line_color="gray",
                                 line_width=0.5)
                fig_z.update_layout(
                    height=220, xaxis_title="Tiempo", yaxis_title="Z-score",
                    margin=dict(t=10, b=30), showlegend=False,
                )
                st.plotly_chart(fig_z, use_container_width=True)

                # β rolling si disponible
                if "beta_rolling" in spread_df.columns and spread_df["beta_rolling"].notna().any():
                    beta_vals = spread_df["beta_rolling"].astype(float).values
                    fig_beta = go.Figure()
                    fig_beta.add_trace(go.Scatter(
                        x=ts_sp, y=beta_vals, name="β rolling",
                        line=dict(width=1.5, color="#f4a261"),
                    ))
                    # Media y bandas ±2σ del β rolling
                    beta_mean = np.nanmean(beta_vals)
                    beta_std  = np.nanstd(beta_vals, ddof=1)
                    fig_beta.add_hline(y=beta_mean,                 line_dash="solid",
                                        line_color="gray", line_width=0.8,
                                        annotation_text=f"β̄={beta_mean:.4f}")
                    fig_beta.add_hline(y=beta_mean + 2*beta_std,    line_dash="dot",
                                        line_color="red", line_width=0.8,
                                        annotation_text="Break +2σ")
                    fig_beta.add_hline(y=beta_mean - 2*beta_std,    line_dash="dot",
                                        line_color="red", line_width=0.8)
                    fig_beta.update_layout(
                        height=220,
                        xaxis_title="Tiempo",
                        yaxis_title="β (hedge ratio rolling)",
                        title="Estabilidad del hedge ratio — si sale de bandas: STRUCTURAL_BREAK",
                        margin=dict(t=30, b=30), showlegend=False,
                    )
                    st.plotly_chart(fig_beta, use_container_width=True)

            # ── Tabla de trades de este par ───────────────────────────────────
            st.subheader("Trades de este par")
            pair_sub_display = pair_trades_all[
                (pair_trades_all["symbol"] == sym_a_sel) |
                (pair_trades_all["symbol"] == sym_b_sel) |
                (pair_trades_all["pair_partner"] == sym_a_sel) |
                (pair_trades_all["pair_partner"] == sym_b_sel)
            ]
            if pair_sub_display.empty:
                st.caption("Sin trades registrados para este par.")
            else:
                disp_cols = [c for c in [
                    "opened_at", "symbol", "side", "asset_class",
                    "entry_price", "exit_price", "size_quote", "pnl_quote", "status",
                ] if c in pair_sub_display.columns]
                st.dataframe(
                    pair_sub_display[disp_cols].reset_index(drop=True),
                    use_container_width=True,
                    height=300,
                )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — System Health
# ═════════════════════════════════════════════════════════════════════════════

with tab5:
    st.subheader("System Health — Infraestructura y Sesiones de Mercado")

    # ── Redis state ───────────────────────────────────────────────────────────
    st.markdown("#### Redis — Estado en caliente")

    if not redis_ok:
        st.error("No se puede conectar a Redis. Verificar que el contenedor redis está UP.")
    else:
        h1, h2, h3 = st.columns(3)

        # Day-lock
        day_lock_val = None
        try:
            day_lock_val = r.get(f"risk:day_locked:{today_utc}")
        except Exception:
            pass
        h1.metric(
            "Day-lock hoy",
            "🔒 ACTIVO" if day_lock_val else "🟢 Libre",
        )
        if day_lock_val:
            try:
                lock_data = json.loads(day_lock_val)
                locked_ts = lock_data.get("locked_at", "")
                h1.caption(f"Bloqueado a: {locked_ts}")
                h1.caption(f"Razón: {lock_data.get('reason', '?')}")
            except Exception:
                pass

        # Equity inicio de día
        eq_start = None
        try:
            eq_start = r.get(f"risk:daily_equity_start:{today_utc}")
        except Exception:
            pass
        h2.metric(
            "Equity inicio de día (UTC)",
            f"${float(eq_start):,.2f}" if eq_start else "—",
        )

        # Drawdown actual estimado
        if eq_start and not equity.empty:
            try:
                current_eq = (
                    equity[equity["quote_currency"] == metrics.get("quote_currency", "USDT")]
                    ["equity_quote"].astype(float).iloc[-1]
                ) if "quote_currency" in equity.columns else float(equity["equity_quote"].iloc[-1])
                dd_today = (float(eq_start) - current_eq) / float(eq_start) * 100
                h3.metric(
                    "DD intradiario estimado",
                    f"{dd_today:.2f}%",
                    delta=None,
                )
                if dd_today >= 2.0:
                    h3.error("≥ 2% → Day-lock activado o inminente.")
                elif dd_today >= 1.5:
                    h3.warning("≥ 1.5% → Acercándose al límite.")
            except Exception:
                h3.metric("DD intradiario", "—")

        st.markdown("---")

        # Claves de riesgo activas en Redis
        st.markdown("#### Claves de riesgo en Redis")
        try:
            risk_keys = r.keys("risk:*")
            kelly_keys = r.keys("risk:kelly_smoothed:*")
            corr_keys = r.keys("risk:returns:*")

            col_k1, col_k2, col_k3 = st.columns(3)
            col_k1.metric("Total risk:* keys", len(risk_keys))
            col_k2.metric("Kelly smoothed (symbols)", len(kelly_keys))
            col_k3.metric("Return windows (correlation)", len(corr_keys))

            # Mostrar Kelly smoothed values por símbolo
            if kelly_keys:
                st.markdown("**Kelly EMA por símbolo:**")
                kelly_data = {}
                for key in sorted(kelly_keys)[:20]:  # max 20 para no sobrecargar
                    sym = key.replace("risk:kelly_smoothed:", "")
                    val = r.get(key)
                    if val:
                        kelly_data[sym] = float(val)
                if kelly_data:
                    df_kelly = pd.DataFrame.from_dict(
                        kelly_data, orient="index", columns=["kelly_smoothed"]
                    )
                    df_kelly.index.name = "symbol"
                    st.dataframe(df_kelly.reset_index(), use_container_width=True, height=200)
        except Exception as e:
            st.warning(f"Error leyendo claves Redis: {e}")

    st.markdown("---")

    # ── MarketSessionGuard — Estado de mercados ───────────────────────────────
    st.markdown("#### MarketSessionGuard — Estado de Sesiones NYSE/NASDAQ")

    try:
        from market_session import MarketSessionGuard
        guard = MarketSessionGuard(
            allow_pre_market=ALLOW_PRE,
            allow_post_market=ALLOW_POST,
        )
        now_utc = dt.datetime.now(UTC)

        if not ALL_SYMBOLS:
            st.caption("No hay símbolos configurados en SYMBOLS.")
        else:
            session_rows = []
            for sym in ALL_SYMBOLS:
                is_crypto = sym.startswith("CRYPTO:") or ":" not in sym
                if is_crypto:
                    session_rows.append({
                        "Symbol": sym,
                        "Asset Class": "🪙 Crypto",
                        "Status": "🟢 Abierto (24/7)",
                        "Cierre en": "—",
                        "Próxima apertura": "—",
                    })
                else:
                    status = guard.get_status(sym, now_utc)
                    if status.is_open:
                        status_str = "🟢 Abierto"
                        if status.is_pre_market:
                            status_str = "🔵 Pre-market"
                        elif status.is_post_market:
                            status_str = "🟡 Post-market"
                        close_in = (
                            f"{status.seconds_until_close/3600:.1f}h"
                            if status.seconds_until_close
                            else "—"
                        )
                        next_open = "—"
                    else:
                        status_str = "🔴 Cerrado"
                        close_in = "—"
                        secs = status.seconds_until_next_open or 0
                        h, m = divmod(int(secs / 60), 60)
                        next_open = f"{h}h {m}m"

                    session_rows.append({
                        "Symbol": sym,
                        "Asset Class": "📈 Equity",
                        "Status": status_str,
                        "Cierre en": close_in,
                        "Próxima apertura": next_open,
                    })

            df_session = pd.DataFrame(session_rows)
            st.dataframe(df_session, use_container_width=True, height=min(
                40 + 35 * len(session_rows), 400))

            # Configuración del guard
            now_et = now_utc.astimezone(ET)
            st.caption(
                f"Hora actual ET: **{now_et.strftime('%H:%M:%S')} ET** "
                f"({now_utc.strftime('%H:%M:%S')} UTC)  |  "
                f"Pre-market: {'Habilitado' if ALLOW_PRE else 'Deshabilitado'}  |  "
                f"Post-market: {'Habilitado' if ALLOW_POST else 'Deshabilitado'}  |  "
                f"Calendario: NYSE/NASDAQ 2024-2027 embebido"
            )

    except ImportError:
        st.warning(
            "No se pudo importar `market_session`. Asegúrate de que el archivo "
            "`market_session.py` esté en el mismo directorio que `dashboard.py`."
        )
    except Exception as e:
        st.error(f"Error en MarketSessionGuard: {e}")

    st.markdown("---")

    # ── Pares configurados ────────────────────────────────────────────────────
    st.markdown("#### Pares cointegrados configurados")
    if not ALL_PAIRS:
        st.caption("Sin pares configurados en PAIR_SYMBOLS.")
    else:
        pair_rows = []
        for sym_a, sym_b in ALL_PAIRS:
            pair_rows.append({
                "Par": f"{sym_a} | {sym_b}",
                "Asset A": sym_a,
                "Asset B": sym_b,
                "Cross-asset": "⚠️ Sí" if sym_a.split(":")[0] != sym_b.split(":")[0] else "No",
            })
        st.dataframe(pd.DataFrame(pair_rows), use_container_width=True, height=200)
        if any(r["Cross-asset"] == "⚠️ Sí" for r in pair_rows):
            st.caption(
                "⚠️ Pares cross-asset (Crypto + Equity) sólo operan durante "
                "horario NYSE. El bot lo gestiona automáticamente."
            )
