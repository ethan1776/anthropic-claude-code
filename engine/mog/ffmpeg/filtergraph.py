"""Build FFmpeg filtergraphs from Python objects instead of hand-written strings.

FFmpeg parses a filtergraph at two levels, each with its own special characters:
  1. inside one option value:  \\  '  :
  2. the graph as a whole:     \\  '  [  ]  ,  ;
A value (say, a file path with a colon or comma in it) must be escaped for level 1 and
then the whole filter description escaped for level 2. Getting this wrong by hand is the
classic FFmpeg bug; here it happens in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _esc_value(v: object) -> str:
    s = f"{v:.6g}" if isinstance(v, float) else str(v)
    for c in ("\\", "'", ":"):
        s = s.replace(c, "\\" + c)
    return s


def _esc_graph(s: str) -> str:
    for c in ("\\", "'", "[", "]", ",", ";"):
        s = s.replace(c, "\\" + c)
    return s


@dataclass
class F:
    """One filter, e.g. F("eq", contrast=1.3, saturation=0)."""

    name: str
    opts: dict[str, object] = field(default_factory=dict)

    def __init__(self, name: str, **opts: object):
        self.name = name
        self.opts = {k: v for k, v in opts.items() if v is not None}

    def render(self) -> str:
        if not self.opts:
            return self.name
        args = ":".join(f"{k}={_esc_value(v)}" for k, v in self.opts.items())
        # Escape only the argument part at graph level; the name and '=' are syntax.
        return f"{self.name}={_esc_graph(args)}"


@dataclass
class Chain:
    inputs: list[str]
    filters: list[F]
    outputs: list[str]

    def render(self) -> str:
        ins = "".join(f"[{i}]" for i in self.inputs)
        outs = "".join(f"[{o}]" for o in self.outputs)
        return ins + ",".join(f.render() for f in self.filters) + outs


class Graph:
    def __init__(self) -> None:
        self.chains: list[Chain] = []
        self._n = 0

    def label(self, prefix: str = "t") -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def add(self, inputs: list[str], filters: list[F], outputs: list[str]) -> None:
        if not filters:
            raise ValueError("a chain needs at least one filter (use F('null') / F('anull') to pass through)")
        self.chains.append(Chain(inputs, filters, outputs))

    def render(self) -> str:
        return ";".join(c.render() for c in self.chains)
