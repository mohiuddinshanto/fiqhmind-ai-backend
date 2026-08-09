"""Phase 10 — Answer Generation.

Evidence-grounded LLM synthesis behind a provider port, with a deterministic
synthesizer as the dependency-free fallback. See ARCHITECTURE §Phase 10.
"""

from app.services.generation.service import GenerationService, get_generator

__all__ = ["GenerationService", "get_generator"]
