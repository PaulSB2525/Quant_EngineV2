"""
llm_bridge.py  (v2)
===================
Cliente agnóstico multi-asset con validación estructurada.

Backends:
    - Gemini 1.5 Flash (Google AI Studio): sentiment analysis rápido.
    - DeepSeek-R1 (deepseek.com): validación profunda de tesis de trading.

Mejoras v1 → v2:
    1. Schemas explícitos:
        * SentimentResult, TradeValidation, PairValidation, EquityValidation.
        * Validación manual (sin importar pydantic; queda más liviano).
        * Rechazo automático si el LLM devuelve JSON malformado o incompleto.
    2. Validación diferenciada por asset class:
        * validate_crypto_thesis(...): thesis técnica, sin contexto fundamental.
        * validate_equity_thesis(...): thesis técnica + EquityFundamentalContext
                                       (Fed rate, earnings, sector beta, 8-K).
        * validate_pair_thesis(...):   thesis sobre par cointegrado, con
                                       CointegrationParams y β rolling.
    3. Token buckets diferenciados:
        * gemini: 15 rpm (sentiment, ligero).
        * deepseek_light: 30 rpm (validation crypto, payload corto).
        * deepseek_heavy: 15 rpm (validation equity, payload con contexto fundamental).
    4. Estrategia de fallo conservadora:
        * Timeout / error / JSON inválido → REJECT (LLM solo veta, nunca aprueba).
    5. Backwards compat:
        * analyze_news_sentiment y validate_trade_thesis (v1) preservados.

Patrón crítico:
    El LLM JAMÁS inicia trades. Su rol es siempre veto-only. Si la red está
    caída, si DeepSeek está saturado, si el JSON viene corrupto — el trade
    se rechaza. Mejor perder oportunidad que ejecutar a ciegas.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


# =============================================================================
# Enums y schemas estructurados
# =============================================================================

class LLMBackend(str, Enum):
    GEMINI_FLASH = "gemini-1.5-flash"
    DEEPSEEK_R1 = "deepseek-reasoner"


class ValidationVerdict(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    INSUFFICIENT_CONTEXT = "insufficient_context"


@dataclass
class LLMResponse:
    """Respuesta cruda del backend; helpers para parseo robusto."""
    text: str
    backend: LLMBackend
    latency_ms: float
    input_tokens_est: int
    output_tokens_est: int
    raw: dict = field(default_factory=dict)

    def parse_json(self) -> Optional[dict]:
        """
        Extrae JSON balanceado. Tolerante a fences markdown y texto envolvente.
        """
        # ```json ... ``` primero
        fence = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```",
                          self.text, re.DOTALL)
        candidate = fence.group(1) if fence else None
        if candidate is None:
            # Primer { ... } balanceado
            start = self.text.find("{")
            if start == -1:
                return None
            depth = 0
            end = -1
            for i in range(start, len(self.text)):
                if self.text[i] == "{":
                    depth += 1
                elif self.text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end == -1:
                return None
            candidate = self.text[start:end]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None


@dataclass
class SentimentResult:
    """Schema para análisis de sentimiento de noticias."""
    sentiment: str             # 'bullish' | 'bearish' | 'neutral'
    confidence: float          # [0, 1]
    rationale: str
    latency_ms: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> Optional["SentimentResult"]:
        try:
            s = str(d.get("sentiment", "neutral")).lower().strip()
            if s not in {"bullish", "bearish", "neutral"}:
                s = "neutral"
            conf = float(d.get("confidence", 0.0))
            conf = max(0.0, min(1.0, conf))
            rationale = str(d.get("rationale", ""))[:500]
            return cls(sentiment=s, confidence=conf, rationale=rationale)
        except (TypeError, ValueError) as e:
            logger.warning("SentimentResult parse falló: %s", e)
            return None


@dataclass
class ValidationResult:
    """Schema para validación de tesis (cripto, equity, pair)."""
    verdict: ValidationVerdict
    confidence: float
    reasoning: str
    red_flags: list[str] = field(default_factory=list)
    recommended_adjustments: str = ""
    latency_ms: float = 0.0
    raw_response_excerpt: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> Optional["ValidationResult"]:
        try:
            v_raw = str(d.get("verdict", "reject")).lower().strip()
            if v_raw == "accept":
                verdict = ValidationVerdict.ACCEPT
            elif v_raw == "insufficient_context":
                verdict = ValidationVerdict.INSUFFICIENT_CONTEXT
            else:
                verdict = ValidationVerdict.REJECT
            conf = float(d.get("confidence", 0.0))
            conf = max(0.0, min(1.0, conf))
            reasoning = str(d.get("reasoning", ""))[:2000]
            red_flags = d.get("red_flags", [])
            if not isinstance(red_flags, list):
                red_flags = []
            red_flags = [str(f)[:200] for f in red_flags][:10]
            adjustments = str(d.get("recommended_adjustments", ""))[:500]
            return cls(
                verdict=verdict, confidence=conf, reasoning=reasoning,
                red_flags=red_flags, recommended_adjustments=adjustments,
            )
        except (TypeError, ValueError) as e:
            logger.warning("ValidationResult parse falló: %s", e)
            return None

    @classmethod
    def conservative_reject(cls, reason: str) -> "ValidationResult":
        """Construye un reject defensivo (timeout, JSON inválido, etc.)."""
        return cls(
            verdict=ValidationVerdict.REJECT,
            confidence=0.0,
            reasoning=reason,
            red_flags=["llm_unavailable_or_malformed"],
        )


# =============================================================================
# Contexto fundamental para equity (opcional, degradación graceful)
# =============================================================================

@dataclass
class EquityFundamentalContext:
    """
    Contexto fundamental opcional para una equity. El bot puebla los campos
    que tenga; nulos significa "no tengo este dato".

    El LLM DEBE manejar nulls explícitamente: si todos los campos son None,
    debería devolver INSUFFICIENT_CONTEXT o evaluar solo los technicals.

    Atributos
    ---------
    fed_funds_rate_pct : Optional[float]
        Tasa actual de Fed funds. Ej: 5.25 para 5.25%.
    is_fomc_week : Optional[bool]
        True si la semana incluye una reunión FOMC.
    hours_to_next_earnings : Optional[float]
        Horas hasta el próximo earnings call. None si >30 días o desconocido.
    sector : Optional[str]
        Ej: "Technology", "Financials", etc.
    sector_beta_rolling_60d : Optional[float]
        Beta de la equity vs SPY (60 días).
    recent_8k_count_30d : Optional[int]
        # de filings 8-K en los últimos 30 días. Alto = corporate events.
    short_interest_pct_of_float : Optional[float]
        % short interest del float. >20% indica squeeze potencial.
    avg_daily_volume_shares : Optional[float]
        ADV en shares para liquidity check.
    is_pre_market : Optional[bool] = False
        True si la sesión es pre-market (mayor spread, menor liquidez).
    is_post_market : Optional[bool] = False
    """
    fed_funds_rate_pct: Optional[float] = None
    is_fomc_week: Optional[bool] = None
    hours_to_next_earnings: Optional[float] = None
    sector: Optional[str] = None
    sector_beta_rolling_60d: Optional[float] = None
    recent_8k_count_30d: Optional[int] = None
    short_interest_pct_of_float: Optional[float] = None
    avg_daily_volume_shares: Optional[float] = None
    is_pre_market: bool = False
    is_post_market: bool = False

    def to_prompt_dict(self) -> dict:
        """
        Serializa solo los campos no-None. Si todo es None, devuelve dict vacío
        para que el LLM sepa que NO hay contexto fundamental.
        """
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None and v is not False}

    @property
    def is_empty(self) -> bool:
        return not self.to_prompt_dict()


# =============================================================================
# Rate limiter (token bucket)
# =============================================================================

class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: float):
        self.rate = rate_per_sec
        self.capacity = capacity
        self.tokens = capacity
        self.last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, n: float = 1.0):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last = now
            if self.tokens >= n:
                self.tokens -= n
                return
            wait_s = (n - self.tokens) / self.rate
        await asyncio.sleep(wait_s)
        await self.acquire(n)


# =============================================================================
# LLMManager
# =============================================================================

class LLMManager:
    """
    Punto único de entrada. Despacha al backend correcto según operación.

    Métodos:
        - analyze_news_sentiment(text)            (v1, Gemini)
        - validate_trade_thesis(thesis_dict)      (v1, DeepSeek light)  [DEPRECATED]
        - validate_crypto_thesis(...)             (v2, DeepSeek light)
        - validate_equity_thesis(...)             (v2, DeepSeek heavy + context)
        - validate_pair_thesis(...)               (v2, DeepSeek heavy)
    """

    GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None,
        timeout_s: float = 30.0,
    ):
        self.gemini_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.deepseek_key = deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
        self._client = httpx.AsyncClient(timeout=timeout_s)

        # Buckets diferenciados
        self._bucket_gemini = TokenBucket(rate_per_sec=15 / 60, capacity=15)
        self._bucket_ds_light = TokenBucket(rate_per_sec=30 / 60, capacity=30)
        self._bucket_ds_heavy = TokenBucket(rate_per_sec=15 / 60, capacity=15)

    async def close(self):
        await self._client.aclose()

    # ---- v1 retained -----------------------------------------------------

    async def analyze_news_sentiment(self,
                                     headline_or_article: str) -> SentimentResult:
        """
        Gemini Flash sentiment analyzer. Devuelve SentimentResult.

        En caso de fallo (parse o network), devuelve sentiment=neutral con
        confidence=0 — NO bullish/bearish con baja confianza, porque eso
        contamina las señales downstream.
        """
        prompt = (
            "You are a financial markets sentiment analyst. Analyze the news "
            "and return ONLY a JSON object with keys:\n"
            "  - 'sentiment': one of 'bullish', 'bearish', 'neutral'\n"
            "  - 'confidence': float in [0, 1]\n"
            "  - 'rationale': one sentence (max 100 words)\n"
            "No extra text, no markdown fences.\n\n"
            f"NEWS:\n{headline_or_article[:3000]}"
        )
        try:
            resp = await self._call_gemini(prompt, temperature=0.1, max_tokens=200)
        except Exception as e:
            logger.warning("Gemini sentiment falló: %s", e)
            return SentimentResult(
                sentiment="neutral", confidence=0.0,
                rationale="llm_error", latency_ms=0.0,
            )

        parsed = resp.parse_json()
        if not parsed:
            logger.warning("Gemini devolvió JSON no parseable: %s",
                           resp.text[:200])
            return SentimentResult(
                sentiment="neutral", confidence=0.0,
                rationale="parse_error", latency_ms=resp.latency_ms,
            )

        result = SentimentResult.from_dict(parsed)
        if result is None:
            return SentimentResult(
                sentiment="neutral", confidence=0.0,
                rationale="schema_mismatch", latency_ms=resp.latency_ms,
            )
        result.latency_ms = resp.latency_ms
        return result

    async def validate_trade_thesis(self, thesis: dict) -> dict:
        """
        DEPRECATED v1 signature. Mantiene compatibilidad con bot v1.
        Internamente delega a validate_crypto_thesis con coerción del dict.

        NUEVO código debe usar validate_crypto_thesis/equity_thesis/pair_thesis
        que tienen tipos explícitos.
        """
        # Detectar si lleva campos equity-like
        if any(k in thesis for k in ("hours_to_next_earnings", "sector_beta",
                                       "fed_funds_rate_pct")):
            ctx = EquityFundamentalContext(
                fed_funds_rate_pct=thesis.get("fed_funds_rate_pct"),
                hours_to_next_earnings=thesis.get("hours_to_next_earnings"),
                sector_beta_rolling_60d=thesis.get("sector_beta"),
            )
            result = await self.validate_equity_thesis(thesis, ctx)
        else:
            result = await self.validate_crypto_thesis(thesis)

        # Coerción a dict v1
        return {
            "verdict": result.verdict.value if result.verdict != ValidationVerdict.INSUFFICIENT_CONTEXT else "reject",
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "red_flags": result.red_flags,
            "recommended_adjustments": result.recommended_adjustments,
            "_latency_ms": result.latency_ms,
        }

    # ---- v2: validation diferenciada ------------------------------------

    async def validate_crypto_thesis(self, thesis: dict) -> ValidationResult:
        """
        Validación de una tesis single-asset crypto.

        Espera en `thesis` (todos opcionales menos symbol y side):
            - symbol, side
            - ou_zscore, ou_half_life_sec
            - garch_vol_annualized
            - kelly_final_fraction
            - expected_alpha_bps, tca_threshold_bps
            - macro_context (string opcional)
        """
        prompt = self._build_crypto_prompt(thesis)
        try:
            resp = await self._call_deepseek(prompt, bucket="light",
                                              temperature=0.2, max_tokens=600)
        except Exception as e:
            logger.warning("DeepSeek crypto-validation falló: %s", e)
            return ValidationResult.conservative_reject(
                f"LLM unreachable: {type(e).__name__}")

        return self._parse_validation(resp)

    async def validate_equity_thesis(
        self,
        thesis: dict,
        context: Optional[EquityFundamentalContext] = None,
    ) -> ValidationResult:
        """
        Validación equity con contexto fundamental opcional.

        Si `context` es None o vacío, el LLM evalúa solo technicals — pero
        debe explicitar en el reasoning que el contexto fundamental faltó.
        """
        ctx = context or EquityFundamentalContext()
        prompt = self._build_equity_prompt(thesis, ctx)
        try:
            resp = await self._call_deepseek(prompt, bucket="heavy",
                                              temperature=0.2, max_tokens=1000)
        except Exception as e:
            logger.warning("DeepSeek equity-validation falló: %s", e)
            return ValidationResult.conservative_reject(
                f"LLM unreachable: {type(e).__name__}")

        return self._parse_validation(resp)

    async def validate_pair_thesis(
        self,
        pair_thesis: dict,
        equity_context_a: Optional[EquityFundamentalContext] = None,
        equity_context_b: Optional[EquityFundamentalContext] = None,
    ) -> ValidationResult:
        """
        Validación de una tesis pair-trading. El payload debe incluir:
            - symbol_a, symbol_b
            - asset_class_a, asset_class_b ('crypto' | 'equity')
            - beta_hedge, alpha_intercept
            - cointegration_verdict, adf_pvalue, beta_rolling_std
            - spread_zscore, spread_half_life_sec
            - spread_garch_vol_annualized
            - kelly_final_fraction
            - expected_alpha_bps, tca_threshold_bps
            - leg_a_size_quote, leg_b_size_quote

        Los contextos equity son opcionales: si alguno de los legs es equity,
        pasar el contexto correspondiente añade calidad a la validación.
        """
        prompt = self._build_pair_prompt(pair_thesis, equity_context_a,
                                           equity_context_b)
        try:
            resp = await self._call_deepseek(prompt, bucket="heavy",
                                              temperature=0.2, max_tokens=1200)
        except Exception as e:
            logger.warning("DeepSeek pair-validation falló: %s", e)
            return ValidationResult.conservative_reject(
                f"LLM unreachable: {type(e).__name__}")

        return self._parse_validation(resp)

    # ---- Prompt builders -----------------------------------------------

    @staticmethod
    def _build_crypto_prompt(thesis: dict) -> str:
        return (
            "You are a senior quantitative trading risk officer reviewing a "
            "CRYPTO trade thesis. Analyze rigorously, looking for:\n"
            "  - Inconsistencies in statistical signals\n"
            "  - Excessive Kelly sizing given volatility regime\n"
            "  - Insufficient edge vs costs\n"
            "  - Half-life longer than typical holding period\n"
            "  - Conflicts between signal direction and macro context\n\n"
            f"THESIS:\n{json.dumps(thesis, indent=2, default=str)}\n\n"
            "Return ONLY a JSON object with these EXACT keys:\n"
            "  - 'verdict': 'accept' | 'reject'\n"
            "  - 'confidence': float in [0, 1]\n"
            "  - 'reasoning': 2-4 sentences explaining the verdict\n"
            "  - 'red_flags': array of short strings (possibly empty)\n"
            "  - 'recommended_adjustments': string (possibly empty)\n"
            "No markdown fences. No extra text."
        )

    @staticmethod
    def _build_equity_prompt(thesis: dict,
                              context: EquityFundamentalContext) -> str:
        ctx_dict = context.to_prompt_dict()
        has_context = bool(ctx_dict)

        context_block = ""
        if has_context:
            context_block = (
                "\n\nFUNDAMENTAL CONTEXT (available):\n"
                f"{json.dumps(ctx_dict, indent=2, default=str)}\n\n"
                "Consider these fundamental factors in your verdict. "
                "Specifically:\n"
                "  - If earnings <24h away → typically reject (event risk)\n"
                "  - If FOMC this week and trade is rate-sensitive → caution\n"
                "  - High short interest >20% → check for squeeze setup conflict\n"
                "  - High recent 8-K count → corporate-event noise risk\n"
                "  - Pre/post market → expect wider spreads, adjust expectations\n"
            )
        else:
            context_block = (
                "\n\nNOTE: No fundamental context was provided. Evaluate ONLY "
                "the technical signals. If you cannot validate without it, "
                "set verdict='insufficient_context' and explain in reasoning.\n"
            )

        return (
            "You are a senior quantitative trading risk officer reviewing an "
            "EQUITY trade thesis. Apply equity-specific scrutiny:\n"
            "  - Equity vol regimes are tighter than crypto (60% cap)\n"
            "  - Half-life of mean-reversion should match holding intent\n"
            "  - Costs include SEC §31 fee, FINRA TAF, half-spread, AC slippage\n"
            "  - Earnings calls, FOMC, 8-K filings can invalidate technicals\n"
            "  - Sector beta exposure must be considered\n"
            f"\nTHESIS:\n{json.dumps(thesis, indent=2, default=str)}"
            f"{context_block}\n"
            "Return ONLY a JSON object with these EXACT keys:\n"
            "  - 'verdict': 'accept' | 'reject' | 'insufficient_context'\n"
            "  - 'confidence': float in [0, 1]\n"
            "  - 'reasoning': 3-5 sentences. If equity context missing, "
            "explicit about that limitation.\n"
            "  - 'red_flags': array of short strings categorized; prefix each "
            "with 'TECH:', 'FUND:', or 'MACRO:'\n"
            "  - 'recommended_adjustments': string\n"
            "No markdown fences. No extra text."
        )

    @staticmethod
    def _build_pair_prompt(
        pair_thesis: dict,
        ctx_a: Optional[EquityFundamentalContext],
        ctx_b: Optional[EquityFundamentalContext],
    ) -> str:
        ctx_block = ""
        ac_a = pair_thesis.get("asset_class_a", "crypto")
        ac_b = pair_thesis.get("asset_class_b", "crypto")
        if ac_a == "equity" and ctx_a is not None and not ctx_a.is_empty:
            ctx_block += ("\n\nLEG A (equity) fundamental context:\n"
                          f"{json.dumps(ctx_a.to_prompt_dict(), indent=2, default=str)}")
        if ac_b == "equity" and ctx_b is not None and not ctx_b.is_empty:
            ctx_block += ("\n\nLEG B (equity) fundamental context:\n"
                          f"{json.dumps(ctx_b.to_prompt_dict(), indent=2, default=str)}")
        if not ctx_block and (ac_a == "equity" or ac_b == "equity"):
            ctx_block = ("\n\nNOTE: Equity leg(s) present but no fundamental "
                         "context provided. Note this limitation in reasoning.")

        return (
            "You are a senior quantitative trading risk officer reviewing a "
            "PAIR-TRADING thesis on a cointegrated spread. Apply pair-specific "
            "scrutiny:\n"
            "  - Verify cointegration is operable (verdict, ADF p-value)\n"
            "  - β rolling stability: high β_rolling_std implies regime "
            "instability\n"
            "  - Half-life of spread must match holding horizon, not too "
            "short (noise) nor too long (capital atado)\n"
            "  - If legs span asset classes (crypto + equity), beware market "
            "hours mismatch creating execution gaps\n"
            "  - Hedge ratio β must be reasonable; |β| > 5 is suspicious\n"
            "  - Two legs = 2× costs vs single-asset. Alpha must justify.\n"
            f"\nPAIR THESIS:\n{json.dumps(pair_thesis, indent=2, default=str)}"
            f"{ctx_block}\n\n"
            "Return ONLY a JSON object with these EXACT keys:\n"
            "  - 'verdict': 'accept' | 'reject' | 'insufficient_context'\n"
            "  - 'confidence': float in [0, 1]\n"
            "  - 'reasoning': 3-5 sentences\n"
            "  - 'red_flags': array of short strings; prefix each with "
            "'COINT:', 'STAT:', 'EXEC:', or 'FUND:'\n"
            "  - 'recommended_adjustments': string\n"
            "No markdown fences. No extra text."
        )

    # ---- Parsing -------------------------------------------------------

    @staticmethod
    def _parse_validation(resp: LLMResponse) -> ValidationResult:
        parsed = resp.parse_json()
        if not parsed:
            logger.warning("DeepSeek devolvió JSON no parseable: %s",
                           resp.text[:200])
            res = ValidationResult.conservative_reject(
                "LLM no devolvió JSON válido; rechazando por seguridad.")
            res.latency_ms = resp.latency_ms
            res.raw_response_excerpt = resp.text[:500]
            return res

        result = ValidationResult.from_dict(parsed)
        if result is None:
            res = ValidationResult.conservative_reject(
                "LLM JSON no cumple schema; rechazando por seguridad.")
            res.latency_ms = resp.latency_ms
            res.raw_response_excerpt = resp.text[:500]
            return res

        result.latency_ms = resp.latency_ms
        result.raw_response_excerpt = resp.text[:500]
        return result

    # ---- HTTP layer -----------------------------------------------------

    async def _call_gemini(self, prompt: str, *,
                            temperature: float = 0.2,
                            max_tokens: int = 512) -> LLMResponse:
        if not self.gemini_key:
            raise RuntimeError("GEMINI_API_KEY no configurada")
        await self._bucket_gemini.acquire()
        url = (self.GEMINI_URL.format(model="gemini-1.5-flash")
               + f"?key={self.gemini_key}")
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        }
        start = time.perf_counter()
        data = await self._request_with_retry("POST", url, json=payload)
        elapsed_ms = (time.perf_counter() - start) * 1000
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            text = ""
            logger.error("Respuesta Gemini malformada: %s", str(data)[:200])
        usage = data.get("usageMetadata", {})
        return LLMResponse(
            text=text, backend=LLMBackend.GEMINI_FLASH,
            latency_ms=elapsed_ms,
            input_tokens_est=int(usage.get("promptTokenCount", len(prompt) // 4)),
            output_tokens_est=int(usage.get("candidatesTokenCount", len(text) // 4)),
            raw=data,
        )

    async def _call_deepseek(self, prompt: str, *,
                              bucket: str = "light",
                              temperature: float = 0.2,
                              max_tokens: int = 1024) -> LLMResponse:
        if not self.deepseek_key:
            raise RuntimeError("DEEPSEEK_API_KEY no configurada")
        bk = self._bucket_ds_heavy if bucket == "heavy" else self._bucket_ds_light
        await bk.acquire()
        payload = {
            "model": "deepseek-reasoner",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.deepseek_key}",
            "Content-Type": "application/json",
        }
        start = time.perf_counter()
        data = await self._request_with_retry("POST", self.DEEPSEEK_URL,
                                                json=payload, headers=headers)
        elapsed_ms = (time.perf_counter() - start) * 1000
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            text = ""
            logger.error("Respuesta DeepSeek malformada: %s", str(data)[:200])
        usage = data.get("usage", {})
        return LLMResponse(
            text=text, backend=LLMBackend.DEEPSEEK_R1,
            latency_ms=elapsed_ms,
            input_tokens_est=int(usage.get("prompt_tokens", len(prompt) // 4)),
            output_tokens_est=int(usage.get("completion_tokens", len(text) // 4)),
            raw=data,
        )

    async def _request_with_retry(self, method: str, url: str, *,
                                    max_attempts: int = 4, **kwargs) -> dict:
        last_exc: Optional[Exception] = None
        for attempt in range(max_attempts):
            try:
                r = await self._client.request(method, url, **kwargs)
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    backoff = (2 ** attempt) + random.uniform(0, 0.5)
                    logger.warning("HTTP %d intento %d; backoff %.2fs",
                                   r.status_code, attempt + 1, backoff)
                    await asyncio.sleep(backoff)
                    continue
                r.raise_for_status()
                return r.json()
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exc = e
                backoff = (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning("LLM call error (%s); reintento en %.2fs",
                               type(e).__name__, backoff)
                await asyncio.sleep(backoff)
        raise RuntimeError(f"LLM call falló tras {max_attempts} intentos: {last_exc}")
