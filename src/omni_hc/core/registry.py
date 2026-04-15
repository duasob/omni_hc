from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Simple named registry for framework extension points."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, T] = {}

    def register(self, key: str, value: T | None = None):
        if value is not None:
            self._register(key, value)
            return value

        def decorator(obj: T) -> T:
            self._register(key, obj)
            return obj

        return decorator

    def _register(self, key: str, value: T) -> None:
        if key in self._items:
            raise KeyError(f"{self.name} registry already contains '{key}'")
        self._items[key] = value

    def get(self, key: str) -> T:
        try:
            return self._items[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(
                f"Unknown {self.name} entry '{key}'. Available: {available}"
            ) from exc

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def items(self) -> tuple[tuple[str, T], ...]:
        return tuple(sorted(self._items.items()))
