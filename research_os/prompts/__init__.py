"""Versioned prompt files: research_os/prompts/<name>.v<N>.md. The version id is logged on every LLM call."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_DIR = Path(__file__).parent
_VERSION = re.compile(r"\.v(\d+)\.md$")


@dataclass(frozen=True)
class Prompt:
    """A loaded prompt template."""

    name: str
    version: str
    template: str

    @property
    def id(self) -> str:
        """'<name>.<version>' (logged with each LLM call)."""
        return f"{self.name}.{self.version}"

    def render(self, **kwargs: str) -> str:
        """Fill {placeholders}; KeyError if one is missing."""
        return self.template.format(**kwargs)


def load_prompt(name: str, version: str | None = None, directory: str | None = None) -> Prompt:
    """Load the requested (default: highest) version of a prompt."""
    d = Path(directory) if directory else _DIR
    found = {}
    for f in d.glob(f"{name}.v*.md"):
        m = _VERSION.search(f.name)
        if m and f.name[: -len(m.group(0))] == name:
            found[int(m.group(1))] = f
    if not found:
        raise FileNotFoundError(f"no prompt named {name!r} in {d}")
    if version:
        n = int(version.lstrip("v"))
        if n not in found:
            raise FileNotFoundError(f"prompt {name!r} has no version {version}")
    else:
        n = max(found)
    return Prompt(name=name, version=f"v{n}", template=found[n].read_text())
