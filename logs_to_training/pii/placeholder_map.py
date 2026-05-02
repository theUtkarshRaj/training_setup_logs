"""Session-scoped placeholder counters for stable redaction."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlaceholderMap:
    """
    Maps normalized sensitive spans to placeholders like <PHONE_1>.

    Counters are per session so the same real value always maps to the same token
    within one export row.
    """

    _counters: dict[str, int] = field(default_factory=dict)
    _value_to_placeholder: dict[str, str] = field(default_factory=dict)

    def placeholder_for(self, category: str, normalized_value: str) -> str:
        key = f"{category}::{normalized_value}"
        if key in self._value_to_placeholder:
            return self._value_to_placeholder[key]
        self._counters[category] = self._counters.get(category, 0) + 1
        token = f"<{category.upper()}_{self._counters[category]}>"
        self._value_to_placeholder[key] = token
        return token
