"""
Provider adapter contract.

A `Protocol`, not an ABC. Python's structural typing means any class with a
matching `complete()` method satisfies this contract without inheriting
from anything — nothing to import at the adapter definition site, nothing
to remember to subclass. This is more permissive than Java's interface
model in one specific way worth knowing: it's checked by a type checker
(mypy/pyright) at analysis time, not by the runtime at call time, so a
typo'd method name won't raise until you actually call it. Run mypy in CI
if you want the Java-style guarantee back.
"""
from typing import Protocol


class LLMProvider(Protocol):
    """Anything with an async `complete(prompt) -> str` method satisfies this."""

    async def complete(self, prompt: str) -> str: ...
