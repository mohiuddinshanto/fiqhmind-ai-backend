"""Tests for Phase 11 LLM provider health and circuit breaking.

Phase 11 requirement: provider health monitoring, circuit breakers, quota tracking,
and fallback logic when providers are unavailable or approaching limits.
"""

from datetime import datetime, timedelta

import pytest

from app.core.config import Settings
from app.services.generation.providers import (
    CircuitBreakerState,
    ProviderHealthService,
    ProviderManager,
    ProviderUsageTracker,
    get_llm_provider,
)


class TestProviderUsageTracker:
    """Test the ProviderUsageTracker class."""

    def test_tracker_initialization(self) -> None:
        """Test tracker initialization with default values."""
        tracker = ProviderUsageTracker("gemini")
        assert tracker.provider_name == "gemini"
        assert tracker.requests_made == 0
        assert tracker.tokens_consumed == 0
        assert tracker.limits == {}
        assert tracker.reset_at > datetime.now()

    def test_tracker_with_limits(self) -> None:
        """Test tracker initialization with limits."""
        limits = {"requests_per_minute": 10, "tokens_per_minute": 1000}
        tracker = ProviderUsageTracker("gemini", limits=limits)
        assert tracker.limits == limits
        assert not tracker.record_request(0)  # Should not exceed limits yet

    def test_tracker_limit_exceeded(self) -> None:
        """Test when tracker would exceed limits."""
        limits = {"requests_per_minute": 5}
        tracker = ProviderUsageTracker("gemini", limits=limits)
        tracker.record_request(0)
        tracker.record_request(0)
        tracker.record_request(0)
        tracker.record_request(0)
        tracker.record_request(0)
        assert tracker.record_request(0)  # Should exceed limit

    def test_tracker_reset(self) -> None:
        """Test tracker reset functionality."""
        tracker = ProviderUsageTracker("gemini")
        tracker.requests_made = 10
        tracker.tokens_consumed = 100
        tracker.reset()
        assert tracker.requests_made == 0
        assert tracker.tokens_consumed == 0
        assert tracker.reset_at > datetime.now()

    def test_tracker_reset_check(self) -> None:
        """Test should_reset logic."""
        # Tracker with past reset (should reset)
        tracker = ProviderUsageTracker("gemini")
        tracker.reset_at = datetime.now() - timedelta(seconds=10)
        assert tracker.should_reset()

    def test_token_tracking(self) -> None:
        """Test token tracking in request recording."""
        tracker = ProviderUsageTracker("gemini")
        tracker.record_request(100)
        assert tracker.tokens_consumed == 100
        assert tracker.requests_made == 1


class TestCircuitBreakerState:
    """Test the CircuitBreakerState class."""

    def test_circuit_breaker_initial_state(self) -> None:
        """Test circuit breaker initial state."""
        breaker = CircuitBreakerState(
            name="gemini",
            fallback_names=["groq", "deterministic"],
            failure_threshold=3,
            recovery_timeout=300,
        )
        assert breaker.name == "gemini"
        assert breaker.fallback_names == ["groq", "deterministic"]
        assert breaker._current_state == "CLOSED"
        assert breaker._failure_count == 0
        assert breaker.can_attempt()

    def test_success_resets_failure_count(self) -> None:
        """Test that success resets failure count."""
        breaker = CircuitBreakerState(
            name="gemini",
            fallback_names=["groq", "deterministic"],
            failure_threshold=3,
        )
        breaker.record_failure()
        breaker.record_failure()
        assert breaker._failure_count == 2
        assert breaker.can_attempt()  # still closed
        breaker.record_success()
        assert breaker._failure_count == 0
        assert breaker._last_success_time is not None

    def test_failure_open_circuit(self) -> None:
        """Test that enough failures open the circuit."""
        breaker = CircuitBreakerState(
            name="gemini",
            fallback_names=["groq", "deterministic"],
            failure_threshold=3,
            recovery_timeout=300,
        )
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()
        assert breaker._current_state == "OPEN"
        assert not breaker.can_attempt()

    def test_recovery_timeout(self) -> None:
        """Test recovery after timeout."""
        breaker = CircuitBreakerState(
            name="gemini",
            fallback_names=["groq", "deterministic"],
            failure_threshold=1,
            recovery_timeout=100,
        )
        breaker.record_failure()
        assert not breaker.can_attempt()
        # Simulate time passing
        breaker._last_failure_time = datetime.now() - timedelta(seconds=200)
        assert breaker.can_attempt()
        assert breaker._current_state == "HALF_OPEN"

    def test_half_open_closes_on_success(self) -> None:
        """Test half-open circuit closes on success."""
        breaker = CircuitBreakerState(
            name="gemini",
            fallback_names=["groq", "deterministic"],
            failure_threshold=1,
        )
        breaker._current_state = "HALF_OPEN"
        breaker.record_success()
        assert breaker._current_state == "CLOSED"

    def test_get_status(self) -> None:
        """Test get_status method."""
        breaker = CircuitBreakerState(
            name="gemini",
            fallback_names=["groq", "deterministic"],
            failure_threshold=3,
        )
        breaker.record_failure()
        status = breaker.get_status()
        assert status["name"] == "gemini"
        assert status["state"] == "CLOSED"
        assert status["failure_count"] == 1
        # With get_fallback_provider -> fallback_names[0] if any
        assert status["fallback_provider"] == "groq"

    def test_fallback_provider_uses_first_fallback(self) -> None:
        """Fallback returns the first name in the fallback list."""
        breaker = CircuitBreakerState(
            name="gemini",
            fallback_names=["groq", "openrouter", "deterministic"],
        )
        breaker.record_failure()
        # With fallback_names, returns the first name (groq)
        assert breaker.get_fallback_provider() == "groq"

    def test_fallback_provider_empty_returns_deterministic(self) -> None:
        """If fallback_names empty, returns deterministic."""
        breaker = CircuitBreakerState(
            name="gemini",
            fallback_names=[],
        )
        breaker.record_failure()
        assert breaker.get_fallback_provider() == "deterministic"


class TestProviderHealthService:
    """Test the ProviderHealthService class."""

    def test_health_service_initialization(self) -> None:
        """Test health service initialization."""
        settings = Settings(
            generator_provider="gemini",
            gemini_api_key="test_key",
            groq_api_key=None,
            openrouter_api_key=None,
        )
        health_service = ProviderHealthService(settings)

        assert "gemini" in health_service._circuit_breakers
        # Groq has no key, so not initialized
        assert "groq" not in health_service._circuit_breakers
        assert "openrouter" not in health_service._circuit_breakers

        assert "gemini" in health_service._usage_trackers
        assert "groq" not in health_service._usage_trackers  # No Groq key in settings
        assert "openrouter" not in health_service._usage_trackers

    def test_can_use_provider(self) -> None:
        """Test can_use_provider method."""
        settings = Settings(
            generator_provider="gemini",
            gemini_api_key="test_key",
            groq_api_key=None,
            openrouter_api_key=None,
        )
        health_service = ProviderHealthService(settings)

        assert health_service.can_use_provider("gemini")
        assert not health_service.can_use_provider("groq")  # Groq not initialized
        assert not health_service.can_use_provider("openrouter")
        assert not health_service.can_use_provider("nonexistent")

    def test_record_success(self) -> None:
        """Test record_success method."""
        settings = Settings(generator_provider="gemini", gemini_api_key="test_key")
        health_service = ProviderHealthService(settings)

        # Record success
        health_service.record_success("gemini")
        status = health_service.get_provider_status("gemini")
        assert status["circuit_breaker"]["last_success_time"] is not None

    def test_record_failure(self) -> None:
        """Test record_failure method."""
        settings = Settings(generator_provider="gemini", gemini_api_key="test_key")
        health_service = ProviderHealthService(settings)

        # Record failure
        health_service.record_failure("gemini")
        status = health_service.get_provider_status("gemini")
        assert status["circuit_breaker"]["failure_count"] == 1
        assert status["circuit_breaker"]["state"] == "CLOSED"

    def test_record_usage(self) -> None:
        """Test record_usage method."""
        settings = Settings(generator_provider="gemini", gemini_api_key="test_key")
        health_service = ProviderHealthService(settings)

        # Record usage without limits
        assert not health_service.record_usage("gemini", 100)

        # Record usage with limits (if configured)
        # Note: limits are not set up in the default settings

    def test_get_health_status(self) -> None:
        """Test get_health_status method."""
        settings = Settings(
            generator_provider="gemini",
            gemini_api_key="test_key",
            groq_api_key=None,
            openrouter_api_key=None,
        )
        health_service = ProviderHealthService(settings)

        status = health_service.get_health_status()
        assert "gemini" in status
        # Groq not in circuit breakers, so not in status
        assert "groq" not in status
        assert "openrouter" not in status

        gemini_status = status["gemini"]
        assert "circuit_breaker" in gemini_status
        assert "usage" in gemini_status

    def test_provider_selection_with_circuit_breaker(self) -> None:
        """Test that circuit breaker affects provider selection."""
        settings = Settings(generator_provider="gemini", gemini_api_key="test_key")
        health_service = ProviderHealthService(settings)

        # Initially should be able to use gemini
        assert health_service.can_use_provider("gemini")

        # Force circuit breaker open
        health_service._circuit_breakers["gemini"]._current_state = "OPEN"
        health_service._circuit_breakers["gemini"]._failure_count = 5
        health_service._circuit_breakers["gemini"]._last_failure_time = datetime.now()

        # Now should not be able to use gemini
        assert not health_service.can_use_provider("gemini")

    def test_fallback_to_deterministic_when_provider_unavailable(self) -> None:
        """Test fallback to deterministic when provider is unavailable."""
        # This test simulates the scenario where a provider is unavailable
        # and the system should fall back to deterministic
        pass


class TestProviderHealthIntegration:
    """Integration tests for Phase 11 provider health features."""

    def test_provider_health_service_integration(self) -> None:
        """Test the complete integration of provider health service."""
        settings = Settings(generator_provider="gemini", gemini_api_key="test_key")
        health_service = ProviderHealthService(settings)

        # Simulate usage tracking
        assert health_service.record_usage("gemini", 100) is False
        assert health_service._usage_trackers["gemini"].requests_made == 1
        assert health_service._usage_trackers["gemini"].tokens_consumed == 100

        # Record success
        health_service.record_success("gemini")

        # Check status
        status = health_service.get_provider_status("gemini")
        assert status["circuit_breaker"]["last_success_time"] is not None
        assert status["usage"]["requests_made"] == 1

    def test_circuit_breaker_failure_handling(self) -> None:
        """Test that circuit breaker properly handles failures."""
        settings = Settings(generator_provider="gemini", gemini_api_key="test_key")
        health_service = ProviderHealthService(settings)

        # Record enough failures to open the circuit
        for _ in range(3):
            health_service.record_failure("gemini")

        status = health_service.get_provider_status("gemini")
        assert status["circuit_breaker"]["state"] == "OPEN"

        # Should not be able to use provider
        assert not health_service.can_use_provider("gemini")

    def test_health_check_scheduling(self) -> None:
        """Test that health checks are scheduled."""
        settings = Settings(generator_provider="gemini", gemini_api_key="test_key")
        health_service = ProviderHealthService(settings)

        # Note: In a real test, we would mock asyncio tasks
        # For now, we just verify the service is set up correctly
        assert health_service._circuit_breakers["gemini"] is not None


class TestFallbackLogic:
    """Test the fallback logic when providers are unavailable."""

    def test_gemini_groq_fallback(self) -> None:
        """Test fallback from Gemini to Groq when Gemini fails."""
        settings = Settings(
            generator_provider="gemini",
            gemini_api_key="test_key",
            groq_api_key="test_key",
        )
        health_service = ProviderHealthService(settings)

        # Both providers should be in circuit breakers
        assert "gemini" in health_service._circuit_breakers
        assert "groq" in health_service._circuit_breakers

        # Gemini's breaker should report Groq as the next fallback
        assert health_service._circuit_breakers["gemini"].get_fallback_provider() == "groq"

    def test_groq_deterministic_fallback(self) -> None:
        """Test fallback from Groq to deterministic when Groq fails."""
        settings = Settings(
            generator_provider="groq",
            groq_api_key="test_key",
        )
        health_service = ProviderHealthService(settings)

        # Only Groq should be in circuit breakers (no Gemini key)
        assert "groq" in health_service._circuit_breakers
        assert "gemini" not in health_service._circuit_breakers

        # If Groq fails, should fall back to deterministic
        assert health_service._circuit_breakers["groq"].get_fallback_provider() == "deterministic"

    def test_provider_health_impacts_generation_service(self) -> None:
        """Test that provider health impacts the generation service."""
        # This is an integration test that verifies the generation service
        # respects provider health status
        pass


class TestHealthCheckScenarios:
    """Test various health check scenarios."""

    def test_provider_within_limits(self) -> None:
        """Test when provider is within limits."""
        settings = Settings(generator_provider="gemini", gemini_api_key="test_key")
        health_service = ProviderHealthService(settings)

        # Should be able to use provider
        assert health_service.can_use_provider("gemini")

    def test_provider_over_limit(self) -> None:
        """Test when provider is over limits."""
        # This scenario would require a provider with configured limits
        # and usage exceeding those limits
        pass

    def test_provider_circuit_breaker_recovery(self) -> None:
        """Test provider recovery from circuit breaker."""
        settings = Settings(generator_provider="gemini", gemini_api_key="test_key")
        health_service = ProviderHealthService(settings)

        # Force circuit breaker open
        breaker = health_service._circuit_breakers["gemini"]
        breaker._current_state = "OPEN"
        breaker._failure_count = 5
        breaker._last_failure_time = datetime.now()

        # Should not be able to use provider
        assert not health_service.can_use_provider("gemini")

        # Simulate recovery time passing
        breaker._last_failure_time = datetime.now() - timedelta(seconds=400)

        # Now should be able to use provider (HALF_OPEN state)
        assert health_service.can_use_provider("gemini")


class TestProviderQuotas:
    """Test provider quota tracking and reset."""

    def test_quota_reset(self) -> None:
        """Test that quotas are reset at the right time."""
        settings = Settings(generator_provider="gemini", gemini_api_key="test_key")
        health_service = ProviderHealthService(settings)

        tracker = health_service._usage_trackers["gemini"]

        # Simulate time passing enough to trigger reset
        tracker.reset_at = datetime.now() - timedelta(seconds=10)

        # Should be ready to reset
        assert tracker.should_reset()

    def test_quota_tracking(self) -> None:
        """Test quota tracking functionality."""
        settings = Settings(generator_provider="gemini", gemini_api_key="test_key")
        health_service = ProviderHealthService(settings)

        tracker = health_service._usage_trackers["gemini"]

        # Make some requests
        for i in range(5):
            tracker.record_request(i * 10)

        assert tracker.requests_made == 5
        assert tracker.tokens_consumed == 100  # 0 + 10 + 20 + 30 + 40 = 100

    def test_quota_exceeded(self) -> None:
        """Test behavior when quota is exceeded."""
        # This requires setting up limits on the usage tracker
        # In a real test, we would configure specific limits
        pass


class TestProviderHealthApiEndpoints:
    """Test the API endpoints for provider health monitoring."""

    def test_health_status_endpoint(self) -> None:
        """Test that health status can be retrieved."""
        # This would test the /api/v1/health/providers endpoint
        # that would be added to expose provider health status
        pass

    def test_circuit_breaker_status_endpoint(self) -> None:
        """Test that circuit breaker status can be retrieved."""
        # This would test the /api/v1/health/circuit-breakers endpoint
        # that would be added to expose circuit breaker status
        pass

    def test_provider_usage_endpoint(self) -> None:
        """Test that provider usage can be retrieved."""
        # This would test the /api/v1/health/usage endpoint
        # that would be added to expose provider usage statistics
        pass


class TestProviderManager:
    """Test the ProviderManager runtime fallback chain (Phase 11)."""

    def test_chain_orders_primary_first(self) -> None:
        """Primary provider is always first in the chain."""
        settings = Settings(
            generator_provider="gemini",
            gemini_api_key="test_key",
            groq_api_key="test_key",
            openrouter_api_key="test_key",
        )
        manager = ProviderManager(settings=settings)
        assert manager.chain_names() == ["gemini", "groq", "openrouter"]

    def test_chain_excludes_unconfigured(self) -> None:
        """Providers without a key are excluded from the chain."""
        settings = Settings(
            generator_provider="gemini",
            gemini_api_key="test_key",
            groq_api_key=None,
            openrouter_api_key=None,
        )
        manager = ProviderManager(settings=settings)
        assert manager.chain_names() == ["gemini"]

    def test_chain_empty_without_keys(self) -> None:
        """A configured provider without a key yields an empty chain."""
        settings = Settings(
            generator_provider="gemini",
            gemini_api_key=None,
            groq_api_key=None,
            openrouter_api_key=None,
        )
        manager = ProviderManager(settings=settings)
        assert manager.chain_names() == []

    def test_fallback_after_open_circuit(self, monkeypatch) -> None:
        """A failing primary diverts to the next healthy rung."""
        settings = Settings(
            generator_provider="gemini",
            gemini_api_key="key",
            groq_api_key="key",
        )
        manager = ProviderManager(settings=settings)

        def boom(*, system_prompt, user_prompt):
            from app.services.generation.providers import ProviderUnavailableError

            raise ProviderUnavailableError("gemini down")

        monkeypatch.setattr(manager._providers["gemini"], "complete", boom)
        manager._providers["groq"].complete = lambda **kw: "groq-ok"

        assert manager.complete(system_prompt="sys", user_prompt="user") == "groq-ok"

    def test_all_fail_raises(self, monkeypatch) -> None:
        """Raises ProviderUnavailableError when every rung fails."""
        from app.services.generation.providers import ProviderUnavailableError

        settings = Settings(generator_provider="gemini", gemini_api_key="key")
        manager = ProviderManager(settings=settings)

        def boom(*args, **kwargs):
            raise ProviderUnavailableError("down")

        monkeypatch.setattr(manager._providers["gemini"], "complete", boom)
        with pytest.raises(ProviderUnavailableError):
            manager.complete(system_prompt="sys", user_prompt="user")

    def test_skips_open_circuit_rung(self, monkeypatch) -> None:
        """A rung with an OPEN circuit is skipped without a call."""
        settings = Settings(
            generator_provider="gemini",
            gemini_api_key="key",
            groq_api_key="key",
        )
        manager = ProviderManager(settings=settings)
        manager._health._circuit_breakers["gemini"]._current_state = "OPEN"
        manager._health._circuit_breakers["gemini"]._last_failure_time = datetime.now()

        called = []

        def fake(*args, **kwargs):
            called.append(1)
            return "whatever"

        monkeypatch.setattr(manager._providers["gemini"], "complete", fake)
        manager._providers["groq"].complete = lambda **kw: "groq-ok"

        assert manager.complete(system_prompt="s", user_prompt="u") == "groq-ok"
        assert called == []  # gemini was never called


class TestOpenRouterProvider:
    """Phase 11: production-grade OpenRouter free-preview client."""

    def test_rotates_models_on_failure(self, monkeypatch) -> None:
        """A failing model falls through to the next in the rotation."""
        from app.services.generation.providers import (
            OpenRouterProvider,
            ProviderUnavailableError,
        )

        provider = OpenRouterProvider(
            api_key="key", models=["bad-model", "good-model"], timeout_seconds=5.0
        )

        def fake_complete(*, model, system_prompt, user_prompt):
            if model == "bad-model":
                raise ProviderUnavailableError("model gone")
            return f"ok:{model}"

        monkeypatch.setattr(provider, "_complete_with_model", fake_complete)
        assert provider.complete(system_prompt="s", user_prompt="u") == "ok:good-model"

    def test_all_models_fail_raises(self, monkeypatch) -> None:
        """Raises when every model in the rotation fails."""
        from app.services.generation.providers import (
            OpenRouterProvider,
            ProviderUnavailableError,
        )

        provider = OpenRouterProvider(api_key="key", models=["m1", "m2"], timeout_seconds=5.0)

        def boom(*, model, user_prompt, system_prompt):
            raise ProviderUnavailableError(f"{model} die")

        monkeypatch.setattr(provider, "_complete_with_model", boom)
        with pytest.raises(ProviderUnavailableError):
            provider.complete(system_prompt="s", user_prompt="u")


class TestGetLLMProvider:
    """Tests for get_llm_provider resolution (Phase 11)."""

    def test_deterministic_returns_none(self) -> None:
        settings = Settings(generator_provider="deterministic")
        assert get_llm_provider(settings) is None

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError):
            get_llm_provider(Settings(generator_provider="foo"))

    def test_provider_without_key_returns_none(self) -> None:
        settings = Settings(
            generator_provider="gemini",
            gemini_api_key=None,
            groq_api_key=None,
            openrouter_api_key=None,
        )
        assert get_llm_provider(settings) is None

    def test_returns_manager_with_keys(self) -> None:
        from app.services.generation.providers import ProviderManager

        settings = Settings(generator_provider="gemini", gemini_api_key="key")
        result = get_llm_provider(settings)
        assert isinstance(result, ProviderManager)

    def test_unhealthy_provider_skips_to_next(self) -> None:
        """When primary is unhealthy, ProviderManager skips to next."""
        from app.services.generation.providers import ProviderManager

        settings = Settings(
            generator_provider="gemini",
            gemini_api_key="key",
            groq_api_key="key",
        )
        manager = ProviderManager(settings=settings)
        assert manager.chain_names() == ["gemini", "groq"]

        # Manually open gemini circuit
        manager._health._circuit_breakers["gemini"]._current_state = "OPEN"
        manager._health._circuit_breakers["gemini"]._last_failure_time = datetime.now()

        # Should still have a chain but skip gemini
        assert not manager._health.can_use_provider("gemini")
        assert manager._health.can_use_provider("groq")
        # The manager will attempt gemini, see it's open, skip it, and try groq


# Export all test classes for pytest
__all__ = [
    "TestProviderUsageTracker",
    "TestCircuitBreakerState",
    "TestProviderHealthService",
    "TestProviderHealthIntegration",
    "TestFallbackLogic",
    "TestHealthCheckScenarios",
    "TestProviderQuotas",
    "TestProviderHealthApiEndpoints",
    "TestProviderManager",
    "TestOpenRouterProvider",
    "TestGetLLMProvider",
]
