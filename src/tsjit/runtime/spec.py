"""Zero-bypass positive-allowlist StreamSpec and TweakPatch validation.

Enforces strict causality (`strict_causality=True`, `lookahead=0`), finite
bounded parameters, explicit field allowlists (rejecting unknown keys), and a
positive-allowlist recursive-descent parser for declarative gate expressions
(zero regex blacklists, zero `eval`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar

DEFAULT_MAX_CHANNELS = 65536

ALLOWED_REVISION_MODES = frozenset({"pwl", "soft_v2", "hybrid"})

ALLOWED_SPEC_FIELDS = frozenset(
    {
        "name",
        "channels",
        "horizon",
        "window",
        "strict_causality",
        "lookahead",
        "ewma_gamma",
        "ridge_lambda",
        "surprise_threshold",
        "evalue_threshold",
        "max_age",
        "alpha",
        "aci_gamma",
        "lambda_age",
        "revision_mode",
        "gate_expr",
    }
)

ALLOWED_TWEAK_FIELDS = frozenset(
    {
        "surprise_threshold",
        "evalue_threshold",
        "max_age",
        "alpha",
        "aci_gamma",
        "lambda_age",
        "revision_mode",
        "gate_expr",
    }
)

ALLOWED_EXPR_IDENTS = frozenset(
    {
        "abs_sg",
        "e_value",
        "e_cusum",
        "e_ville",
        "e_sr",
        "age",
        "max_age",
        "surprise_threshold",
        "evalue_threshold",
    }
)

ALLOWED_NAME_FIRST = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
ALLOWED_NAME_REST = ALLOWED_NAME_FIRST | frozenset("0123456789_.-")


class SpecValidationError(ValueError):
    """Raised when a StreamSpec or TweakPatch violates strict positive-allowlist rules."""


# ---------- Positive-Allowlist Gate Expression Parser & Evaluator ----------

_TOK_IDENT = "IDENT"
_TOK_NUM = "NUM"
_TOK_CMP = "CMP"
_TOK_AND = "AND"
_TOK_OR = "OR"
_TOK_NOT = "NOT"
_TOK_LP = "LP"
_TOK_RP = "RP"
_TOK_EOF = "EOF"

_CMP_OPS = frozenset({"<", "<=", ">", ">=", "==", "!="})
_MAX_EXPR_LEN = 256
_MAX_EXPR_DEPTH = 16


def _tokenize_gate_expr(expr: str) -> list[tuple[str, Any]]:
    if not isinstance(expr, str):
        raise SpecValidationError("gate_expr must be a string or None")
    if len(expr) == 0 or len(expr) > _MAX_EXPR_LEN:
        raise SpecValidationError(f"gate_expr length must be in 1..{_MAX_EXPR_LEN}")

    tokens: list[tuple[str, Any]] = []
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        if ch in " \t\n\r":
            i += 1
            continue
        if ch == "(":
            tokens.append((_TOK_LP, "("))
            i += 1
            continue
        if ch == ")":
            tokens.append((_TOK_RP, ")"))
            i += 1
            continue
        if ch in "<>=!":
            if i + 1 < n and expr[i : i + 2] in _CMP_OPS:
                tokens.append((_TOK_CMP, expr[i : i + 2]))
                i += 2
                continue
            if ch in "<>":
                tokens.append((_TOK_CMP, ch))
                i += 1
                continue
            raise SpecValidationError(f"Disallowed operator at position {i} in gate_expr")
        if ch in "0123456789" or (ch == "." and i + 1 < n and expr[i + 1] in "0123456789"):
            j = i
            has_dot = False
            has_exp = False
            while j < n:
                c = expr[j]
                if c in "0123456789":
                    j += 1
                elif c == "." and not has_dot and not has_exp:
                    has_dot = True
                    j += 1
                elif c in "eE" and not has_exp:
                    has_exp = True
                    j += 1
                    if j < n and expr[j] in "+-":
                        j += 1
                else:
                    break
            raw = expr[i:j]
            try:
                val = float(raw)
            except ValueError as exc:
                raise SpecValidationError(f"Invalid numeric literal {raw!r} in gate_expr") from exc
            if not math.isfinite(val):
                raise SpecValidationError(f"Non-finite numeric literal {raw!r} in gate_expr")
            tokens.append((_TOK_NUM, val))
            i = j
            continue
        if ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_":
            j = i
            while (
                j < n
                and expr[j] in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
            ):
                j += 1
            word = expr[i:j]
            if word == "and":
                tokens.append((_TOK_AND, word))
            elif word == "or":
                tokens.append((_TOK_OR, word))
            elif word == "not":
                tokens.append((_TOK_NOT, word))
            elif word in ALLOWED_EXPR_IDENTS:
                tokens.append((_TOK_IDENT, word))
            else:
                raise SpecValidationError(
                    f"Disallowed identifier {word!r} in gate_expr; "
                    f"allowed={sorted(ALLOWED_EXPR_IDENTS)}"
                )
            i = j
            continue
        raise SpecValidationError(f"Disallowed character {ch!r} at position {i} in gate_expr")

    tokens.append((_TOK_EOF, None))
    return tokens


class GateExpression:
    """Compiled positive-allowlist boolean expression over causal telemetry scalars."""

    def __init__(self, source: str, ast_node: tuple[Any, ...]) -> None:
        self.source = source
        self._node = ast_node

    @classmethod
    def parse(cls, expr: str) -> GateExpression:
        tokens = _tokenize_gate_expr(expr)
        pos = 0

        def peek() -> tuple[str, Any]:
            return tokens[pos]

        def consume(expected: str | None = None) -> tuple[str, Any]:
            nonlocal pos
            tok = tokens[pos]
            if expected is not None and tok[0] != expected:
                raise SpecValidationError(
                    f"Expected {expected} in gate_expr, got {tok[0]} ({tok[1]!r})"
                )
            pos += 1
            return tok

        def parse_or(depth: int) -> tuple[Any, ...]:
            if depth > _MAX_EXPR_DEPTH:
                raise SpecValidationError("gate_expr exceeds maximum nesting depth")
            left = parse_and(depth)
            while peek()[0] == _TOK_OR:
                consume(_TOK_OR)
                right = parse_and(depth)
                left = ("OR", left, right)
            return left

        def parse_and(depth: int) -> tuple[Any, ...]:
            if depth > _MAX_EXPR_DEPTH:
                raise SpecValidationError("gate_expr exceeds maximum nesting depth")
            left = parse_not(depth)
            while peek()[0] == _TOK_AND:
                consume(_TOK_AND)
                right = parse_not(depth)
                left = ("AND", left, right)
            return left

        def parse_not(depth: int) -> tuple[Any, ...]:
            if depth > _MAX_EXPR_DEPTH:
                raise SpecValidationError("gate_expr exceeds maximum nesting depth")
            if peek()[0] == _TOK_NOT:
                consume(_TOK_NOT)
                return ("NOT", parse_not(depth + 1))
            return parse_atom(depth)

        def parse_atom(depth: int) -> tuple[Any, ...]:
            if depth > _MAX_EXPR_DEPTH:
                raise SpecValidationError("gate_expr exceeds maximum nesting depth")
            tok_t, _ = peek()
            if tok_t == _TOK_LP:
                consume(_TOK_LP)
                node = parse_or(depth + 1)
                consume(_TOK_RP)
                return node
            left_val = parse_value()
            op_tok = consume(_TOK_CMP)
            right_val = parse_value()
            return ("CMP", op_tok[1], left_val, right_val)

        def parse_value() -> tuple[Any, ...]:
            tok_t, tok_v = peek()
            if tok_t == _TOK_IDENT:
                consume(_TOK_IDENT)
                return ("IDENT", tok_v)
            if tok_t == _TOK_NUM:
                consume(_TOK_NUM)
                return ("NUM", tok_v)
            raise SpecValidationError(
                f"Expected numeric literal or allowed identifier in gate_expr, got {tok_v!r}"
            )

        root = parse_or(0)
        if peek()[0] != _TOK_EOF:
            raise SpecValidationError(f"Unexpected trailing token {peek()[1]!r} in gate_expr")
        return cls(expr, root)

    def evaluate(self, env: Mapping[str, float]) -> bool:
        def eval_val(node: tuple[Any, ...]) -> float:
            kind, val = node
            if kind == "NUM":
                return float(val)
            if kind == "IDENT":
                # Preserve +Inf / -Inf so infinite surprise or e-value exceeds finite thresholds;
                # NaN is handled safely in comparison ops below.
                return float(env[val])
            raise SpecValidationError(f"Invalid value AST node {kind!r}")

        def eval_bool(node: tuple[Any, ...]) -> bool:
            kind = node[0]
            if kind == "OR":
                return eval_bool(node[1]) or eval_bool(node[2])
            if kind == "AND":
                return eval_bool(node[1]) and eval_bool(node[2])
            if kind == "NOT":
                return not eval_bool(node[1])
            if kind == "CMP":
                op = node[1]
                lv = eval_val(node[2])
                rv = eval_val(node[3])
                if math.isnan(lv) or math.isnan(rv):
                    return False
                if op == "<":
                    return lv < rv
                if op == "<=":
                    return lv <= rv
                if op == ">":
                    return lv > rv
                if op == ">=":
                    return lv >= rv
                if op == "==":
                    return lv == rv
                if op == "!=":
                    return lv != rv
            raise SpecValidationError(f"Invalid bool AST node {kind!r}")

        return eval_bool(self._node)


# ---------- Primitive Strict Validators ----------


def _req_int(name: str, val: Any, lo: int, hi: int) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise SpecValidationError(f"{name} must be an integer (got {type(val).__name__})")
    if val < lo or val > hi:
        raise SpecValidationError(f"{name}={val} out of allowed bounds [{lo}, {hi}]")
    return val


def _req_finite_float(
    name: str,
    val: Any,
    lo: float,
    hi: float,
    *,
    lo_inclusive: bool = True,
    hi_inclusive: bool = True,
) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise SpecValidationError(f"{name} must be a real number (got {type(val).__name__})")
    f = float(val)
    if not math.isfinite(f):
        raise SpecValidationError(f"{name} must be finite (got {val!r})")
    if (f < lo) or (not lo_inclusive and f == lo):
        op = ">=" if lo_inclusive else ">"
        raise SpecValidationError(f"{name}={f} must satisfy {op} {lo}")
    if (f > hi) or (not hi_inclusive and f == hi):
        op = "<=" if hi_inclusive else "<"
        raise SpecValidationError(f"{name}={f} must satisfy {op} {hi}")
    return f


def _req_name(val: Any) -> str:
    if not isinstance(val, str) or not (1 <= len(val) <= 64):
        raise SpecValidationError("name must be a string of length 1..64")
    if val[0] not in ALLOWED_NAME_FIRST or any(c not in ALLOWED_NAME_REST for c in val[1:]):
        raise SpecValidationError(
            f"name {val!r} violates positive allowlist [A-Za-z][A-Za-z0-9_.-]{{0,63}}"
        )
    return val


def _req_channels(val: Any, max_channels: int) -> tuple[int, ...]:
    if not isinstance(val, (tuple, list)) or len(val) == 0:
        raise SpecValidationError("channels must be a non-empty tuple/list of channel indices")
    if len(val) > max_channels:
        raise SpecValidationError(f"channels length {len(val)} exceeds max_channels={max_channels}")
    seen: set[int] = set()
    out: list[int] = []
    for idx, ch in enumerate(val):
        c = _req_int(f"channels[{idx}]", ch, 0, max_channels - 1)
        if c in seen:
            raise SpecValidationError(f"duplicate channel index {c} in channels")
        seen.add(c)
        out.append(c)
    return tuple(out)


@dataclass(frozen=True)
class StreamSpec:
    """Strictly validated causal streaming specification."""

    max_channels: ClassVar[int] = DEFAULT_MAX_CHANNELS

    name: str = "default"
    channels: tuple[int, ...] = (0,)
    horizon: int = 1
    window: int = 64
    strict_causality: bool = True
    lookahead: int = 0
    ewma_gamma: float = 0.95
    ridge_lambda: float = 1e-2
    surprise_threshold: float = 2.5
    evalue_threshold: float = 20.0
    max_age: int = 16
    alpha: float = 0.1
    aci_gamma: float = 0.05
    lambda_age: float = 0.15
    revision_mode: str = "pwl"
    gate_expr: str | None = None

    def __post_init__(self) -> None:
        raw = {
            "name": self.name,
            "channels": self.channels,
            "horizon": self.horizon,
            "window": self.window,
            "strict_causality": self.strict_causality,
            "lookahead": self.lookahead,
            "ewma_gamma": self.ewma_gamma,
            "ridge_lambda": self.ridge_lambda,
            "surprise_threshold": self.surprise_threshold,
            "evalue_threshold": self.evalue_threshold,
            "max_age": self.max_age,
            "alpha": self.alpha,
            "aci_gamma": self.aci_gamma,
            "lambda_age": self.lambda_age,
            "revision_mode": self.revision_mode,
            "gate_expr": self.gate_expr,
        }
        normalized = _validate_fields(raw, max_channels=self.max_channels)
        for field_name, field_val in normalized.items():
            object.__setattr__(self, field_name, field_val)

    def compiled_gate_expr(self) -> GateExpression | None:
        if self.gate_expr is None:
            return None
        return GateExpression.parse(self.gate_expr)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "channels": list(self.channels),
            "horizon": self.horizon,
            "window": self.window,
            "strict_causality": self.strict_causality,
            "lookahead": self.lookahead,
            "ewma_gamma": self.ewma_gamma,
            "ridge_lambda": self.ridge_lambda,
            "surprise_threshold": self.surprise_threshold,
            "evalue_threshold": self.evalue_threshold,
            "max_age": self.max_age,
            "alpha": self.alpha,
            "aci_gamma": self.aci_gamma,
            "lambda_age": self.lambda_age,
            "revision_mode": self.revision_mode,
            "gate_expr": self.gate_expr,
        }


@dataclass(frozen=True)
class TweakPatch:
    """Validated, immutable live-tweak patch over strictly allowed non-structural fields."""

    updates: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.updates, Mapping) or len(self.updates) == 0:
            raise SpecValidationError("TweakPatch.updates must be a non-empty mapping")
        raw = dict(self.updates)
        unknown = set(raw.keys()) - ALLOWED_TWEAK_FIELDS
        if unknown:
            raise SpecValidationError(
                f"Disallowed or unknown tweak fields: {sorted(unknown)}; "
                f"allowed={sorted(ALLOWED_TWEAK_FIELDS)}"
            )
        base = {
            "name": "default",
            "channels": (0,),
            "horizon": 1,
            "window": 64,
            "strict_causality": True,
            "lookahead": 0,
            "ewma_gamma": 0.95,
            "ridge_lambda": 1e-2,
            "surprise_threshold": 2.5,
            "evalue_threshold": 20.0,
            "max_age": 16,
            "alpha": 0.1,
            "aci_gamma": 0.05,
            "lambda_age": 0.15,
            "revision_mode": "pwl",
            "gate_expr": None,
        }
        base.update(raw)
        normalized = _validate_fields(base, max_channels=DEFAULT_MAX_CHANNELS)
        frozen_updates = MappingProxyType({k: normalized[k] for k in raw})
        object.__setattr__(self, "updates", frozen_updates)


def _validate_fields(raw: Mapping[str, Any], *, max_channels: int) -> dict[str, Any]:
    name = _req_name(raw["name"])
    channels = _req_channels(raw["channels"], max_channels)
    horizon = _req_int("horizon", raw["horizon"], 1, 1024)
    window = _req_int("window", raw["window"], 2, 1_048_576)

    sc = raw["strict_causality"]
    if sc is not True:
        raise SpecValidationError("strict_causality must be True (non-causal mode is forbidden)")
    lookahead = _req_int("lookahead", raw["lookahead"], 0, 0)

    ewma_gamma = _req_finite_float(
        "ewma_gamma", raw["ewma_gamma"], 0.0, 1.0, lo_inclusive=False, hi_inclusive=True
    )
    ridge_lambda = _req_finite_float("ridge_lambda", raw["ridge_lambda"], 1e-12, 1e6)
    surprise_threshold = _req_finite_float(
        "surprise_threshold", raw["surprise_threshold"], 0.0, 1e6, lo_inclusive=False
    )
    evalue_threshold = _req_finite_float(
        "evalue_threshold", raw["evalue_threshold"], 1.0, 1e9, lo_inclusive=False
    )
    max_age = _req_int("max_age", raw["max_age"], 1, 100_000)
    alpha = _req_finite_float("alpha", raw["alpha"], 0.001, 0.5)
    aci_gamma = _req_finite_float("aci_gamma", raw["aci_gamma"], 0.0, 1.0)
    lambda_age = _req_finite_float("lambda_age", raw["lambda_age"], 0.0, 100.0)

    mode = raw["revision_mode"]
    if not isinstance(mode, str) or mode not in ALLOWED_REVISION_MODES:
        raise SpecValidationError(
            f"revision_mode {mode!r} not in allowed set {sorted(ALLOWED_REVISION_MODES)}"
        )

    gate_expr = raw["gate_expr"]
    if gate_expr is not None:
        GateExpression.parse(gate_expr)

    return {
        "name": name,
        "channels": channels,
        "horizon": horizon,
        "window": window,
        "strict_causality": sc,
        "lookahead": lookahead,
        "ewma_gamma": ewma_gamma,
        "ridge_lambda": ridge_lambda,
        "surprise_threshold": surprise_threshold,
        "evalue_threshold": evalue_threshold,
        "max_age": max_age,
        "alpha": alpha,
        "aci_gamma": aci_gamma,
        "lambda_age": lambda_age,
        "revision_mode": mode,
        "gate_expr": gate_expr,
    }


def validate_spec(
    spec_or_dict: StreamSpec | Mapping[str, Any],
    *,
    max_channels: int = DEFAULT_MAX_CHANNELS,
) -> StreamSpec:
    """Validate a StreamSpec or mapping using a zero-bypass positive allowlist."""
    if (
        isinstance(max_channels, bool)
        or not isinstance(max_channels, int)
        or max_channels < 1
        or max_channels > DEFAULT_MAX_CHANNELS
    ):
        raise SpecValidationError(f"max_channels must be an integer in [1, {DEFAULT_MAX_CHANNELS}]")

    if isinstance(spec_or_dict, StreamSpec):
        raw = {
            "name": spec_or_dict.name,
            "channels": spec_or_dict.channels,
            "horizon": spec_or_dict.horizon,
            "window": spec_or_dict.window,
            "strict_causality": spec_or_dict.strict_causality,
            "lookahead": spec_or_dict.lookahead,
            "ewma_gamma": spec_or_dict.ewma_gamma,
            "ridge_lambda": spec_or_dict.ridge_lambda,
            "surprise_threshold": spec_or_dict.surprise_threshold,
            "evalue_threshold": spec_or_dict.evalue_threshold,
            "max_age": spec_or_dict.max_age,
            "alpha": spec_or_dict.alpha,
            "aci_gamma": spec_or_dict.aci_gamma,
            "lambda_age": spec_or_dict.lambda_age,
            "revision_mode": spec_or_dict.revision_mode,
            "gate_expr": spec_or_dict.gate_expr,
        }
        _validate_fields(raw, max_channels=max_channels)
        return spec_or_dict
    elif isinstance(spec_or_dict, Mapping):
        unknown = set(spec_or_dict.keys()) - ALLOWED_SPEC_FIELDS
        if unknown:
            raise SpecValidationError(f"Unknown StreamSpec keys rejected: {sorted(unknown)}")
        defaults: dict[str, Any] = {
            "name": "default",
            "channels": (0,),
            "horizon": 1,
            "window": 64,
            "strict_causality": True,
            "lookahead": 0,
            "ewma_gamma": 0.95,
            "ridge_lambda": 1e-2,
            "surprise_threshold": 2.5,
            "evalue_threshold": 20.0,
            "max_age": 16,
            "alpha": 0.1,
            "aci_gamma": 0.05,
            "lambda_age": 0.15,
            "revision_mode": "pwl",
            "gate_expr": None,
        }
        defaults.update(spec_or_dict)
        raw = defaults
    else:
        raise SpecValidationError(
            f"Expected StreamSpec or mapping, got {type(spec_or_dict).__name__}"
        )

    normalized = _validate_fields(raw, max_channels=max_channels)
    return StreamSpec(**normalized)


def validate_tweak(patch_or_dict: TweakPatch | Mapping[str, Any]) -> TweakPatch:
    """Validate a TweakPatch (rejecting unknown, non-finite, or structural fields)."""
    if isinstance(patch_or_dict, TweakPatch):
        return TweakPatch(updates=dict(patch_or_dict.updates))
    if isinstance(patch_or_dict, Mapping):
        return TweakPatch(updates=dict(patch_or_dict))
    raise SpecValidationError(
        f"Expected TweakPatch or mapping, got {type(patch_or_dict).__name__}"
    )


def apply_tweak(
    spec: StreamSpec,
    patch_or_dict: TweakPatch | Mapping[str, Any],
    *,
    max_channels: int = DEFAULT_MAX_CHANNELS,
) -> StreamSpec:
    """Return a newly validated StreamSpec with patch applied (pure, atomic)."""
    patch = validate_tweak(patch_or_dict)
    merged = spec.to_dict()
    merged.update(patch.updates)
    return validate_spec(merged, max_channels=max_channels)
