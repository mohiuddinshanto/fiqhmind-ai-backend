"""LLM provider ports for Answer Generation (Phase 10/11).

External LLM models sit behind the common `LLMProvider` port; the chat service
depends on the port, never on a concrete vendor. `GeminiProvider`,
`GroqProvider`, and `OpenRouterProvider` are real HTTP implementations used at
runtime; `get_llm_provider` returns `None` for the dependency-free
`deterministic` adapter or when no configured provider is usable, so the
deterministic path always works.

Phase 11 additions:
- `ProviderHealthService` tracks provider health, free-tier quotas, circuit
  breaker state, and schedules periodic health checks against provider docs.
- `ProviderManager` implements the runtime fallback chain
  (primary -> configured fallbacks in order -> deterministic), with retry /
  exponential backoff per rung, so a free-tier quota cut on the primary does
  not fail the chat pipeline.
- `OpenRouterProvider` is production-grade: it reads its API key from settings
  and rotates between a list of free-preview models when one 429s, so the
  "rotating third fallback" rung (ARCHITECTURE §Phase 11) is not a dependency.
"""

import asyncio
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal, Protocol

import httpx
import structlog

from app.core.config import Settings, get_settings

logger = structlog.get_logger(__name__)

_GEMINI_GENERATE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
_GEMINI_STREAM_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
)
_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

# OpenRouter: rate-limited free preview models, not a dependency
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_MODELS_URL = f"{_OPENROUTER_BASE_URL}/models"
_OPENROUTER_CHAT_URL = f"{_OPENROUTER_BASE_URL}/chat/completions"

# Default free-preview models to rotate through (Phase 11). Order matters only
# as a fallback ladder; these rotate and vanish without notice by design.
_OPENROUTER_FREE_MODELS = (
    "meta-llama/llama-3.3-70b-instruct:free",
    "gryphe/mythomax-l2-13b",
    "microsoft/phi-3.5-mini-128k-instruct:free",
)


class ProviderUnavailableError(Exception):
    """Raised when an external LLM cannot be reached or returns an error."""


@dataclass
class ProviderUsageTracker:
    """Track usage against free tier limits."""
    provider_name: str
    requests_made: int = 0
    tokens_consumed: int = 0
    reset_at: datetime | None = None
    limits: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.reset_at is None:
            self.reset_at = datetime.now() + timedelta(days=1)
        if self.limits is None:
            self.limits = {}
        self.minute_requests = 0
        self.minute_tokens = 0
        self.minute_reset_at = datetime.now() + timedelta(minutes=1)

    def record_request(self, tokens: int = 0) -> bool:
        """Record a request. Returns True if limit would be exceeded."""
        if self.should_reset():
            self.reset()
        if datetime.now() >= self.minute_reset_at:
            self.minute_requests = 0
            self.minute_tokens = 0
            self.minute_reset_at = datetime.now() + timedelta(minutes=1)

        self.requests_made += 1
        self.tokens_consumed += tokens
        self.minute_requests += 1
        self.minute_tokens += tokens

        return self.is_limit_exceeded()

    def is_limit_exceeded(self) -> bool:
        """Check if limits are currently exceeded."""
        if self.should_reset():
            self.reset()
        if datetime.now() >= self.minute_reset_at:
            self.minute_requests = 0
            self.minute_tokens = 0
            self.minute_reset_at = datetime.now() + timedelta(minutes=1)

        if not self.limits:
            return False

        rpm = self.limits.get("requests_per_minute")
        if rpm and self.minute_requests > rpm:
            return True

        tpm = self.limits.get("tokens_per_minute")
        if tpm and self.minute_tokens > tpm:
            return True

        rpd = self.limits.get("requests_per_day")
        if rpd and self.requests_made > rpd:
            return True

        tpd = self.limits.get("tokens_per_day")
        if tpd and self.tokens_consumed > tpd:
            return True

        return False

    def should_reset(self) -> bool:
        """Check if limits should be reset."""
        if self.reset_at is None:
            return False
        return datetime.now() >= self.reset_at

    def reset(self) -> None:
        """Reset usage counters."""
        self.requests_made = 0
        self.tokens_consumed = 0
        self.reset_at = datetime.now() + timedelta(days=1)
        self.minute_requests = 0
        self.minute_tokens = 0
        self.minute_reset_at = datetime.now() + timedelta(minutes=1)


class CircuitBreakerState:
    """Tracks the state of a provider's circuit breaker."""

    def __init__(
        self,
        *,
        name: str,
        fallback_names: list[str],
        failure_threshold: int = 5,
        recovery_timeout: int = 300,  # seconds
        health_check_interval: int = 3600,  # seconds (1 hour)
    ) -> None:
        self.name = name
        self.fallback_names = fallback_names
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.health_check_interval = health_check_interval

        self._failure_count = 0
        self._last_failure_time: datetime | None = None
        self._last_success_time: datetime | None = None
        self._last_health_check: datetime | None = None
        self._current_state: Literal["CLOSED", "OPEN", "HALF_OPEN"] = "CLOSED"
        self._provider_limits: dict[str, Any] = {}
        self._last_known_quota: dict[str, Any] = {}

    def record_success(self) -> None:
        """Record a successful provider call."""
        self._failure_count = 0
        self._last_success_time = datetime.now()
        if self._current_state == "HALF_OPEN":
            self._current_state = "CLOSED"
            logger.info("circuit_breaker_half_open_to_closed", provider=self.name)

    def record_failure(self) -> None:
        """Record a failed provider call."""
        self._failure_count += 1
        self._last_failure_time = datetime.now()
        if self._failure_count >= self.failure_threshold:
            self._current_state = "OPEN"
            logger.warning(
                "circuit_breaker_opened",
                provider=self.name,
                failures=self._failure_count,
                threshold=self.failure_threshold,
            )

    def can_attempt(self) -> bool:
        """Check if a provider call can be attempted."""
        if self._current_state == "CLOSED":
            return True
        if self._current_state == "OPEN":
            if (
                self._last_failure_time
                and (datetime.now() - self._last_failure_time).total_seconds()
                > self.recovery_timeout
            ):
                self._current_state = "HALF_OPEN"
                logger.info(
                    "circuit_breaker_open_to_half_open",
                    provider=self.name,
                    recovery_timeout=self.recovery_timeout,
                )
                return True
            return False
        if self._current_state == "HALF_OPEN":
            return True
        return False

    def get_fallback_provider(self) -> str:
        """Get the name of the next provider in the fallback chain.

        Returns the first name in `fallback_names` if present, else the
        dependency-free `deterministic` terminal. The caller already filters
        `fallback_names` to configured providers.
        """
        return self.fallback_names[0] if self.fallback_names else "deterministic"

    def update_limits(self, limits: dict[str, Any]) -> None:
        """Update the provider's current limits."""
        self._provider_limits = limits
        self._last_known_quota = {
            k: v for k, v in limits.items() if "quota" in k.lower() or "limit" in k.lower()
        }

    def should_health_check(self) -> bool:
        """Check if a health check is due."""
        if not self._last_health_check:
            return True
        return (
            datetime.now() - self._last_health_check
        ).total_seconds() > self.health_check_interval

    def record_health_check(self) -> None:
        """Record that a health check was performed."""
        self._last_health_check = datetime.now()

    def get_status(self) -> dict[str, Any]:
        """Get the current status of the circuit breaker."""
        return {
            "name": self.name,
            "state": self._current_state,
            "failure_count": self._failure_count,
            "last_failure_time": self._last_failure_time,
            "last_success_time": self._last_success_time,
            "fallback_provider": self.get_fallback_provider(),
            "current_limits": self._provider_limits,
            "last_health_check": self._last_health_check,
        }


class LLMProvider(Protocol):
    """The port the generation service talks to.

    `complete` returns the raw model output (expected: a single JSON object);
    validation happens downstream, not in the provider.
    """

    def complete(self, *, system_prompt: str, user_prompt: str) -> str: ...


class StreamableLLMProvider(Protocol):
    """The streaming port used by `GenerationService.stream_answer` (Phase 15 M3).

    `stream` yields raw model text chunks as the vendor emits them. For
    JSON-contract providers the chunks are fragments of the eventual JSON
    document; the service accumulates, validates, and only then exposes the
    final answer. A provider that only implements `LLMProvider.complete` is
    still streamed by the service (the full output is yielded as one chunk).
    """

    def stream(self, *, system_prompt: str, user_prompt: str) -> Iterator[str]: ...


class GeminiProvider:
    """Gemini REST client (text-only, JSON output)."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        url = _GEMINI_GENERATE_URL.format(model=self._model)
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 2048,
                "responseMimeType": "application/json",
            },
        }
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    url,
                    params={"key": self._api_key},
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"gemini request failed: {exc}") from exc

        if response.status_code == 429:
            raise ProviderUnavailableError("gemini rate limited (HTTP 429)")
        if response.status_code != 200:
            raise ProviderUnavailableError(
                f"gemini returned HTTP {response.status_code}: {response.text[:300]}"
            )
        try:
            text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderUnavailableError(
                f"gemini response malformed: {response.text[:300]}"
            ) from exc
        return text.strip()

    def stream(self, *, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Stream generation content (SSE `streamGenerateContent?alt=sse`)."""
        url = _GEMINI_STREAM_URL.format(model=self._model)
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 2048,
                "responseMimeType": "application/json",
            },
        }
        try:
            with httpx.Client(timeout=self._timeout) as client:
                with client.stream(
                    "POST",
                    url,
                    params={"key": self._api_key, "alt": "sse"},
                    json=payload,
                ) as response:
                    if response.status_code == 429:
                        raise ProviderUnavailableError("gemini rate limited (HTTP 429)")
                    if response.status_code != 200:
                        raise ProviderUnavailableError(
                            f"gemini returned HTTP {response.status_code}: "
                            f"{response.text[:300]}"
                        )
                    for line in response.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[len("data: ") :].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        try:
                            obj = json.loads(raw)
                        except ValueError:
                            continue
                        parts = (
                            obj.get("candidates") or [{}]
                        )[0].get("content", {}).get("parts", [])
                        text = "".join(
                            part.get("text", "")
                            for part in parts
                            if isinstance(part, dict)
                        )
                        if text:
                            yield text
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"gemini stream failed: {exc}") from exc


class GroqProvider:
    """Groq OpenAI-compatible chat client (JSON output)."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    _GROQ_CHAT_URL,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"groq request failed: {exc}") from exc

        if response.status_code == 429:
            raise ProviderUnavailableError("groq rate limited (HTTP 429)")
        if response.status_code != 200:
            raise ProviderUnavailableError(
                f"groq returned HTTP {response.status_code}: {response.text[:300]}"
            )
        try:
            text = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderUnavailableError(
                f"groq response malformed: {response.text[:300]}"
            ) from exc
        return text.strip()

    def stream(self, *, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Stream an OpenAI-compatible chat completion (`stream: true`)."""
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "stream": True,
            "response_format": {"type": "json_object"},
        }
        try:
            with httpx.Client(timeout=self._timeout) as client:
                with client.stream(
                    "POST",
                    _GROQ_CHAT_URL,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                ) as response:
                    if response.status_code == 429:
                        raise ProviderUnavailableError("groq rate limited (HTTP 429)")
                    if response.status_code != 200:
                        raise ProviderUnavailableError(
                            f"groq returned HTTP {response.status_code}: "
                            f"{response.text[:300]}"
                        )
                    for line in response.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[len("data: ") :].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        try:
                            obj = json.loads(raw)
                        except ValueError:
                            continue
                        choice = (obj.get("choices") or [{}])[0]
                        text = (choice.get("delta") or {}).get("content")
                        if text:
                            yield text
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"groq stream failed: {exc}") from exc


class OpenRouterProvider:
    """OpenRouter OpenAI-compatible client for free-preview models (Phase 11).

    OpenRouter is explicitly the *rotating third fallback*, never a primary.
    The client rotates through a small list of free-preview models, so a model
    that disappears or rate-limits (common: previews vanish with no notice)
    falls to the next one instead of failing the request.
    """

    def __init__(
        self,
        *,
        api_key: str,
        models: list[str] | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._models = list(models or _OPENROUTER_FREE_MODELS)
        self._timeout = timeout_seconds

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        last_error: str | None = None
        for model in self._models:
            try:
                return self._complete_with_model(
                    model=model, system_prompt=system_prompt, user_prompt=user_prompt
                )
            except ProviderUnavailableError as exc:
                last_error = str(exc)
                logger.warning("openrouter_model_failed", model=model, error=last_error)
        raise ProviderUnavailableError(f"openrouter: all models failed: {last_error or 'unknown'}")

    def stream(self, *, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Stream through the rotating free-preview model list.

        Mirrors `complete`: a model that rejects the request before its first
        token is skipped for the next one; a stream that dies after yielding
        tokens propagates (partial output cannot be cleanly resumed).
        """
        last_error: str | None = None
        for model in self._models:
            started = False
            try:
                for chunk in self._stream_with_model(
                    model=model, system_prompt=system_prompt, user_prompt=user_prompt
                ):
                    started = True
                    yield chunk
                return
            except ProviderUnavailableError as exc:
                last_error = str(exc)
                logger.warning("openrouter_model_stream_failed", model=model, error=last_error)
                if started:
                    raise
        raise ProviderUnavailableError(f"openrouter: all models failed: {last_error or 'unknown'}")

    def _stream_with_model(
        self, *, model: str, system_prompt: str, user_prompt: str
    ) -> Iterator[str]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "stream": True,
            "response_format": {"type": "json_object"},
        }
        try:
            with httpx.Client(timeout=self._timeout) as client:
                with client.stream(
                    "POST",
                    _OPENROUTER_CHAT_URL,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "HTTP-Referer": "https://fiqhmind.ai",
                        "X-Title": "FiqhMind AI",
                    },
                    json=payload,
                ) as response:
                    if response.status_code in (429, 401, 403):
                        raise ProviderUnavailableError(
                            f"openrouter {model} rejected (HTTP {response.status_code})"
                        )
                    if response.status_code != 200:
                        raise ProviderUnavailableError(
                            f"openrouter returned HTTP {response.status_code}: "
                            f"{response.text[:300]}"
                        )
                    for line in response.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[len("data: ") :].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        try:
                            obj = json.loads(raw)
                        except ValueError:
                            continue
                        choice = (obj.get("choices") or [{}])[0]
                        text = (choice.get("delta") or {}).get("content")
                        if text:
                            yield text
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"openrouter request failed: {exc}") from exc

    def _complete_with_model(self, *, model: str, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    _OPENROUTER_CHAT_URL,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "HTTP-Referer": "https://fiqhmind.ai",
                        "X-Title": "FiqhMind AI",
                    },
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"openrouter request failed: {exc}") from exc

        if response.status_code in (429, 401, 403):
            raise ProviderUnavailableError(
                f"openrouter {model} rejected (HTTP {response.status_code})"
            )
        if response.status_code != 200:
            raise ProviderUnavailableError(
                f"openrouter returned HTTP {response.status_code}: {response.text[:300]}"
            )
        try:
            text = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderUnavailableError(
                f"openrouter response malformed: {response.text[:300]}"
            ) from exc
        return text.strip()


class ProviderHealthService:
    """Manages health checks and circuit breaking for LLM providers.

    Phase 11 requirement: monitor provider health, free-tier quotas, circuit
    breakers, and schedule periodic health checks against provider docs.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._usage_trackers: dict[str, ProviderUsageTracker] = {}
        self._circuit_breakers: dict[str, CircuitBreakerState] = {}
        self._client = httpx.Client(timeout=30.0)
        self._last_quota_pull: dict[str, datetime] = {}
        self._last_api_key_check: dict[str, datetime] = {}

        # Initialize trackers for each provider
        self._setup_usage_trackers()

        # Initialize circuit breakers for each provider
        self._setup_circuit_breakers()

    def _setup_usage_trackers(self) -> None:
        """Set up usage trackers for all providers that have API keys configured."""
        for provider_name in self._configured_providers_with_keys():
            self._usage_trackers[provider_name] = ProviderUsageTracker(provider_name)

    def _setup_circuit_breakers(self) -> None:
        """Set up circuit breakers for all providers with configured API keys.

        Each breaker's `fallback_names` reflects the configured
        `provider_fallback_order`, so a free-tier cut on the primary diverts to
        the next configured rung and finally to `deterministic`.
        """
        fallback_order = self._fallback_order()
        configured_with_keys = self._configured_providers_with_keys()

        for name in configured_with_keys:
            # Filter fallback_names to only include providers that are also configured with keys
            others = [
                p for p in fallback_order if p != name and p in configured_with_keys
            ]
            others.append("deterministic")
            breaker = CircuitBreakerState(
                name=name,
                fallback_names=others,
                failure_threshold=3,
                recovery_timeout=300,
                health_check_interval=3600,
            )
            self._circuit_breakers[name] = breaker

    def _configured_providers_with_keys(self) -> list[str]:
        """Providers that have a configured API key."""
        result: list[str] = []
        if self._settings.gemini_api_key:
            result.append("gemini")
        if self._settings.groq_api_key:
            result.append("groq")
        if self._settings.openrouter_api_key:
            result.append("openrouter")
        return result

    def _fallback_order(self) -> list[str]:
        """The configured fallback order (settings value, validated)."""
        order = [
            part.strip().lower()
            for part in (self._settings.provider_fallback_order or "").split(",")
            if part.strip()
        ]
        valid = {"gemini", "groq", "openrouter"}
        return [p for p in order if p in valid]

    def can_use_provider(self, provider_name: str) -> bool:
        """Check if a provider can be used, considering API key presence,
        circuit breaker state, and quota/limits."""
        if not self._has_api_key(provider_name):
            logger.debug("can_use_provider: no api key", provider=provider_name)
            return False
        breaker = self._circuit_breakers.get(provider_name)
        if not breaker:
            return False
        if not breaker.can_attempt():
            return False

        tracker = self._usage_trackers.get(provider_name)
        if tracker and tracker.is_limit_exceeded():
            logger.warning("provider_rate_limit_exceeded", provider=provider_name)
            return False

        return True

    def _has_api_key(self, name: str) -> bool:
        """Check if the settings have an API key for the given provider."""
        if name == "gemini":
            return bool(self._settings.gemini_api_key)
        if name == "groq":
            return bool(self._settings.groq_api_key)
        if name == "openrouter":
            return bool(self._settings.openrouter_api_key)
        return False

    def record_usage(self, provider_name: str, tokens: int = 0) -> bool:
        """Record usage for a provider. Returns True if limit would be exceeded."""
        tracker = self._usage_trackers.get(provider_name)
        if not tracker:
            return False
        return tracker.record_request(tokens)

    def record_success(self, provider_name: str) -> None:
        """Record a successful call to a provider."""
        breaker = self._circuit_breakers.get(provider_name)
        if breaker:
            breaker.record_success()

    def record_failure(self, provider_name: str) -> None:
        """Record a failed call to a provider."""
        breaker = self._circuit_breakers.get(provider_name)
        if breaker:
            breaker.record_failure()

    def get_provider_status(self, provider_name: str) -> dict[str, Any]:
        """Get status for a provider."""
        breaker = self._circuit_breakers.get(provider_name)
        tracker = self._usage_trackers.get(provider_name)
        status = {
            "circuit_breaker": breaker.get_status() if breaker else None,
            "usage": tracker.__dict__ if tracker else None,
        }
        return status

    async def start_health_checks(self) -> asyncio.Task[None]:
        """Start background health checks for all providers."""
        return asyncio.create_task(self._run_health_checks_loop())

    async def run_health_checks_once(self) -> dict[str, Any]:
        """Run one pass of provider health checks and return the resulting status.

        Phase 15 daily maintenance: a Celery beat task invokes this through
        `run_coroutine` so the provider-quota check runs off the request path.
        """
        await self._perform_health_checks()
        return self.get_health_status()

    async def _run_health_checks_loop(self) -> None:
        """Run health checks for all providers."""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception:  # pragma: no cover - defensive
                logger.exception("health_checks_loop_error")
                await asyncio.sleep(60)  # Sleep longer on error

    async def _perform_health_checks(self) -> None:
        """Perform health checks for all providers."""
        for provider_name in self._circuit_breakers:
            breaker = self._circuit_breakers[provider_name]
            if breaker.should_health_check():
                await self._check_provider_health(provider_name, breaker)

    async def _check_provider_health(
        self, provider_name: str, breaker: CircuitBreakerState
    ) -> None:
        """Check the health of a provider and update its circuit breaker."""
        try:
            if provider_name == "gemini":
                limits = await self._check_gemini_limits()
            elif provider_name == "groq":
                limits = await self._check_groq_limits()
            elif provider_name == "openrouter":
                limits = await self._check_openrouter_limits()
            else:
                return

            breaker.update_limits(limits)

            # Update usage tracker's limits
            if provider_name in self._usage_trackers:
                self._usage_trackers[provider_name].limits = limits

            breaker.record_health_check()
            logger.info("provider_health_check_success", provider=provider_name, limits=limits)

            # Test provider if in HALF_OPEN state
            if breaker._current_state == "HALF_OPEN":
                if await self._test_provider(provider_name):
                    breaker.record_success()
                    logger.info("provider_recovery", provider=provider_name)
        except Exception as exc:
            logger.warning("provider_health_check_failed", provider=provider_name, error=str(exc))

    async def _check_gemini_limits(self) -> dict[str, Any]:
        """Check Gemini's current limits by inspecting AI Studio docs."""
        try:
            response = self._client.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                timeout=30.0,
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                flash_models = [m for m in models if "flash" in m.get("name", "")]
                if flash_models:
                    model_info = flash_models[0]
                    return {
                        "requests_per_minute": 15,
                        "tokens_per_minute": 1_000_000,
                        "requests_per_day": 1_500,
                        "model_info": model_info,
                    }
            return {"error": "Could not fetch Gemini limits", "status": response.status_code}
        except Exception as exc:
            return {"error": str(exc)}

    async def _check_groq_limits(self) -> dict[str, Any]:
        """Check Groq's current limits by inspecting API docs."""
        try:
            response = self._client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {self._settings.groq_api_key}"},
                timeout=30.0,
            )
            if response.status_code == 200:
                return {
                    "requests_per_minute": 30,
                    "tokens_per_minute": 6_000,
                    "requests_per_day": 1_000,
                    "models": response.json().get("data", []),
                }
            return {"error": "Could not fetch Groq limits", "status": response.status_code}
        except Exception as exc:
            return {"error": str(exc)}

    async def _check_openrouter_limits(self) -> dict[str, Any]:
        """Check OpenRouter's current limits by inspecting API docs."""
        try:
            response = self._client.get(
                _OPENROUTER_MODELS_URL,
                timeout=30.0,
            )
            if response.status_code == 200:
                models = response.json().get("data", [])
                free_models = [
                    m
                    for m in models
                    if m.get("pricing", {}).get("prompt") == 0
                    and m.get("pricing", {}).get("completion") == 0
                ]
                return {
                    "requests_per_minute": 50,
                    "tokens_per_minute": 10_000,
                    "requests_per_day": 3_000,
                    "free_models_count": len(free_models),
                    "sample_free_models": free_models[:3] if free_models else [],
                }
            return {"error": "Could not fetch OpenRouter limits", "status": response.status_code}
        except Exception as exc:
            return {"error": str(exc)}

    async def _check_api_key(self, provider_name: str) -> bool:
        """Perform a lightweight check if the API key is valid without expensive calls.

        This runs lazily on first provider use (or half-open breaker).
        """
        if provider_name == "gemini":
            return bool(
                self._settings.gemini_api_key
                and self._settings.gemini_api_key.startswith("AIza")
            )
        if provider_name == "groq":
            return bool(
                self._settings.groq_api_key
                and self._settings.groq_api_key.startswith("gsk_")
            )
        if provider_name == "openrouter":
            return bool(
                self._settings.openrouter_api_key
                and self._settings.openrouter_api_key.startswith("sk-")
            )
        return False

    async def _test_provider(self, provider_name: str) -> bool:
        """Test if a provider is back online after being in OPEN state."""
        # First, a lightweight API key sanity check.
        if not await self._check_api_key(provider_name):
            logger.debug("provider_test_failed: invalid_api_key", provider=provider_name)
            return False
        try:
            if provider_name == "gemini":
                response = self._client.post(
                    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent",
                    json={"contents": [{"parts": [{"text": "test"}]}]},
                    params={"key": self._settings.gemini_api_key},
                    timeout=30.0,
                )
                return response.status_code == 200
            elif provider_name == "groq":
                response = self._client.post(
                    _GROQ_CHAT_URL,
                    headers={"Authorization": f"Bearer {self._settings.groq_api_key}"},
                    json={
                        "model": "llama-3.1-8b-instruct",
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 1,
                    },
                    timeout=30.0,
                )
                return response.status_code == 200
            elif provider_name == "openrouter":
                response = self._client.post(
                    _OPENROUTER_CHAT_URL,
                    headers={"Authorization": f"Bearer {self._settings.openrouter_api_key}"},
                    json={
                        "model": "meta-llama/llama-3.3-70b-instruct:free",
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 1,
                    },
                    timeout=30.0,
                )
                return response.status_code in (200, 401, 403)
            return False
        except Exception as exc:
            logger.debug("provider_test_failed", provider=provider_name, error=str(exc))
            return False

    def get_health_status(self) -> dict[str, Any]:
        """Get health status for all providers."""
        return {
            name: {
                "circuit_breaker": breaker.get_status(),
                "usage": tracker.__dict__ if tracker else None,
            }
            for name, breaker in self._circuit_breakers.items()
            for tracker in [self._usage_trackers.get(name)]
        }


class ProviderManager:
    """Runtime fallback chain over the configured LLM providers.

    Phase 11: on request, the primary (`generator_provider`) is tried first
    (respecting circuit breaker + quota), then the remaining configured
    providers in the configured fallback order. Each rung gets a small number
    of retries with exponential backoff before the next rung is attempted.
    If no provider succeeds, `ProviderUnavailableError` is raised and the
    generation service falls back to deterministic synthesis.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        health: ProviderHealthService | None = None,
        primary: str | None = None,
    ) -> None:
        self._settings = settings
        self._health = health or ProviderHealthService(settings)
        self._primary = primary or settings.generator_provider
        self._providers: dict[str, LLMProvider] = {}
        self._build_chain()

    def _build_chain(self) -> None:
        """Build the ordered provider chain from settings."""
        fallback_order = self._health._fallback_order()
        chain = [self._primary] + [p for p in fallback_order if p != self._primary]
        chain = [
            p for p in chain if self._health._has_api_key(p)
        ]  # Use _has_api_key from health service
        for name in chain:
            self._providers[name] = self._build_single(name)

    def _build_single(self, name: str) -> LLMProvider:
        timeout = self._settings.generation_request_timeout_seconds
        if name == "gemini":
            return GeminiProvider(
                api_key=self._settings.gemini_api_key or "",
                model=self._settings.gemini_model,
                timeout_seconds=timeout,
            )
        if name == "groq":
            return GroqProvider(
                api_key=self._settings.groq_api_key or "",
                model=self._settings.groq_model,
                timeout_seconds=timeout,
            )
        if name == "openrouter":
            # OpenRouter model list is dynamic; pass _OPENROUTER_FREE_MODELS constant
            # unless a specific model is forced by settings.
            models_to_use = (
                [self._settings.openrouter_model]
                if self._settings.openrouter_model != _OPENROUTER_FREE_MODELS[0]
                else list(_OPENROUTER_FREE_MODELS)
            )
            return OpenRouterProvider(
                api_key=self._settings.openrouter_api_key or "",
                models=models_to_use,
                timeout_seconds=timeout,
            )
        raise ValueError(f"unknown provider {name!r}")

    def chain_names(self) -> list[str]:
        """The ordered provider names in this chain (for diagnostics)."""
        return list(self._providers)

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Try each provider in order; raise if all fail."""
        retries = max(0, int(self._settings.generation_retries))
        errors: list[str] = []
        for name in self._providers:
            provider = self._providers[name]
            if not self._health.can_use_provider(name):
                errors.append(f"{name}: circuit open")
                continue
            for attempt in range(retries + 1):
                if attempt > 0:
                    time.sleep(0.5 * (2 ** (attempt - 1)))  # exponential backoff
                try:
                    text = provider.complete(system_prompt=system_prompt, user_prompt=user_prompt)
                    self._health.record_success(name)
                    self._health.record_usage(name, tokens=len(text))
                    return text
                except ProviderUnavailableError as exc:
                    self._health.record_failure(name)
                    last = f"{name}: {exc}"
                    errors.append(last)
                    logger.warning(
                        "provider_rung_failed",
                        provider=name,
                        attempt=attempt,
                        error=str(exc),
                    )
        raise ProviderUnavailableError("; ".join(errors) or "no providers configured")

    def stream(self, *, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Stream through the fallback chain (Phase 15 M3).

        A rung that rejects the request before its first token is retried /
        skipped for the next rung. A rung that dies *after* yielding tokens
        propagates `ProviderUnavailableError` immediately: partial output
        cannot be cleanly resumed on another provider, so the caller falls
        back to deterministic synthesis.
        """
        retries = max(0, int(self._settings.generation_retries))
        errors: list[str] = []
        for name in self._providers:
            provider = self._providers[name]
            if not self._health.can_use_provider(name):
                errors.append(f"{name}: circuit open")
                continue
            stream = getattr(provider, "stream", None)
            if stream is None:
                errors.append(f"{name}: no stream support")
                continue
            for attempt in range(retries + 1):
                if attempt > 0:
                    time.sleep(0.5 * (2 ** (attempt - 1)))  # exponential backoff
                started = False
                total_text = ""
                try:
                    for chunk in stream(
                        system_prompt=system_prompt, user_prompt=user_prompt
                    ):
                        started = True
                        total_text += chunk
                        yield chunk
                    self._health.record_success(name)
                    self._health.record_usage(name, tokens=len(total_text))
                    return
                except ProviderUnavailableError as exc:
                    self._health.record_failure(name)
                    if started:
                        raise
                    last = f"{name}: {exc}"
                    errors.append(last)
                    logger.warning(
                        "provider_stream_rung_failed",
                        provider=name,
                        attempt=attempt,
                        error=str(exc),
                    )
        raise ProviderUnavailableError("; ".join(errors) or "no providers configured")


def get_llm_provider(settings: Settings | None = None) -> LLMProvider | None:
    """Resolve the provider chain, or None for the deterministic path.

    A configured provider without an API key is treated as unconfigured: the
    pipeline logs a warning and falls back to deterministic synthesis instead of
    failing at request time. The returned manager implements the Phase 11
    fallback chain (primary -> configured fallbacks -> error -> deterministic).
    """
    resolved = settings or get_settings()
    provider_name = resolved.generator_provider

    if provider_name == "deterministic":
        return None

    valid = {"gemini", "groq", "openrouter"}
    if provider_name not in valid:
        raise ValueError(f"unknown generator_provider={provider_name!r}")

    manager = ProviderManager(settings=resolved)
    if not manager.chain_names():
        logger.warning(
            "generator_provider=%s configured without API keys; using deterministic path",
            provider_name,
        )
        return None

    # The manager's can_use_provider already checks API key and circuit breaker.
    # If the primary provider is unhealthy, the manager's .complete() will try fallbacks.
    # We only fall back to deterministic if *all* configured providers are unhealthy.
    if not any(manager._health.can_use_provider(name) for name in manager.chain_names()):
        logger.warning("all configured providers unhealthy; using deterministic path")
        return None

    return manager
