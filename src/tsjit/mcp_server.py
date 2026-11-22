"""Stdio JSON-RPC 2.0 Model Context Protocol (MCP) server for tsjit.

Exposes 6 tools over stdio (auto-detecting newline-delimited JSON and
Content-Length framing):
  - tsjit_info
  - tsjit_verify
  - tsjit_benchmark
  - tsjit_validate_spec
  - tsjit_stream_sim
  - tsjit_hub_demo
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping
from typing import Any, BinaryIO

from ._version import __version__

PROTOCOL_VERSION = "2024-11-05"
MAX_FRAME_BYTES = 16 * 1024 * 1024  # 16 MiB safety cap
_MAX_CONTENT_LENGTH = MAX_FRAME_BYTES
_MAX_LINE_BYTES = MAX_FRAME_BYTES + 1024
_MAX_HEADER_LINES = 32
_MAX_SIM_VALUES = 100_000
_MAX_CHANNELS = 65536
_EOF = object()

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "tsjit_info",
        "description": (
            "Return tsjit version, native C11 backend, SIMD ISA, and optional backend status."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "tsjit_verify",
        "description": (
            "Run native C11 build, seqlock contention, causality, and NaN-isolation verification."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"quick": {"type": "boolean", "default": True}},
            "additionalProperties": False,
        },
    },
    {
        "name": "tsjit_benchmark",
        "description": (
            "Run local C11 PWL-ARRIVE, Schur-complement, and multi-agent hub benchmarks."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "steps": {"type": "integer", "minimum": 100, "maximum": 500000, "default": 10000},
                "n_agents": {"type": "integer", "minimum": 1, "maximum": 128, "default": 8},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "tsjit_validate_spec",
        "description": (
            "Validate a StreamSpec dictionary against tsjit's zero-bypass positive allowlist."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "spec": {"type": "object"},
                "max_channels": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": _MAX_CHANNELS,
                    "default": _MAX_CHANNELS,
                },
            },
            "required": ["spec"],
            "additionalProperties": False,
        },
    },
    {
        "name": "tsjit_stream_sim",
        "description": (
            "Simulate a causal streaming series through GapAsyncRuntime with OnlineACI and EValue."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 1,
                    "maxItems": _MAX_SIM_VALUES,
                },
                "spec": {"type": "object"},
            },
            "required": ["values"],
            "additionalProperties": False,
        },
    },
    {
        "name": "tsjit_hub_demo",
        "description": (
            "Run a MultiAgentStreamHub demo verifying NaN isolation, hot-add, and SHA-256 ledger."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "n_steps": {"type": "integer", "minimum": 5, "maximum": 5000, "default": 40},
                "inject_nan_channel": {"type": "boolean", "default": True},
            },
            "additionalProperties": False,
        },
    },
]

_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    t["name"]: t["inputSchema"] for t in TOOL_DEFINITIONS
}


def _clamp_json_float(val: float) -> float:
    if math.isfinite(val):
        return val
    if math.isnan(val):
        return 0.0
    return math.copysign(sys.float_info.max, val)


def _sanitize_json(obj: Any) -> Any:
    """Recursively clamp non-finite floats so RFC 8259 (allow_nan=False) never fails."""
    if isinstance(obj, float):
        return _clamp_json_float(obj)
    if isinstance(obj, Mapping):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    return obj


def _reject_non_rfc_constant(tok: str) -> Any:
    raise ValueError(f"Non-RFC8259 JSON constant rejected: {tok!r}")


def _req_bounded_int(name: str, val: Any, lo: int, hi: int) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(f"{name} must be an integer (got {type(val).__name__})")
    if val < lo or val > hi:
        raise ValueError(f"{name}={val} must be in [{lo}, {hi}]")
    return val


def _call_tool(name: str, arguments: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    if name == "tsjit_info":
        from .cli import collect_info

        return collect_info(), False

    if name == "tsjit_verify":
        from .cli import run_verification

        quick_raw = arguments.get("quick", True)
        if not isinstance(quick_raw, bool):
            return {"error": "quick must be a boolean"}, True
        res = run_verification(quick=quick_raw)
        return res, not bool(res.get("ok", False))

    if name == "tsjit_benchmark":
        from .cli import run_benchmark

        steps = _req_bounded_int("steps", arguments.get("steps", 10_000), 100, 500_000)
        n_agents = _req_bounded_int("n_agents", arguments.get("n_agents", 8), 1, 128)
        return run_benchmark(steps=steps, n_agents=n_agents), False

    if name == "tsjit_validate_spec":
        from .runtime.spec import SpecValidationError, validate_spec

        spec_obj = arguments.get("spec")
        max_channels_raw = arguments.get("max_channels", _MAX_CHANNELS)
        try:
            max_channels = _req_bounded_int("max_channels", max_channels_raw, 1, _MAX_CHANNELS)
            validated = validate_spec(spec_obj, max_channels=max_channels)  # type: ignore[arg-type]
            return {"valid": True, "spec": validated.to_dict()}, False
        except (SpecValidationError, ValueError) as exc:
            return {"valid": False, "error": str(exc)}, True

    if name == "tsjit_stream_sim":
        from .runtime.arrival import GapAsyncRuntime
        from .runtime.spec import SpecValidationError, StreamSpec, validate_spec

        vals = arguments.get("values")
        if not isinstance(vals, list) or len(vals) == 0:
            return {"error": "values must be a non-empty list of numbers"}, True
        if len(vals) > _MAX_SIM_VALUES:
            return {"error": f"values length {len(vals)} exceeds maximum {_MAX_SIM_VALUES}"}, True
        parsed_vals: list[float] = []
        for idx, v in enumerate(vals):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return {"error": f"values[{idx}] must be a number (got {type(v).__name__})"}, True
            fv = float(v)
            if not math.isfinite(fv):
                return {"error": f"values[{idx}] must be finite"}, True
            parsed_vals.append(fv)
        spec_in = arguments.get("spec")
        try:
            spec = validate_spec(spec_in) if spec_in is not None else StreamSpec(name="mcp_sim")
        except SpecValidationError as exc:
            return {"error": str(exc)}, True

        dispatches = 0
        alarms = 0
        last_res = None
        with GapAsyncRuntime(spec=spec) as rt:
            for t, x in enumerate(parsed_vals):
                res = rt.step(t, x)
                last_res = res
                if res.dispatched:
                    dispatches += 1
                if res.alarm:
                    alarms += 1
            telem = _sanitize_json(rt.telemetry())
        return {
            "steps": len(parsed_vals),
            "dispatches": dispatches,
            "alarms": alarms,
            "last": {
                "step": last_res.step,
                "q9": [_clamp_json_float(q) for q in last_res.q9],
                "cal_lo": _clamp_json_float(last_res.cal_lo),
                "cal_hi": _clamp_json_float(last_res.cal_hi),
                "abs_sg": _clamp_json_float(last_res.abs_sg),
                "e_cusum": _clamp_json_float(last_res.e_cusum),
                "mark": last_res.mark_name,
            }
            if last_res is not None
            else None,
            "telemetry": telem,
        }, False

    if name == "tsjit_hub_demo":
        from .runtime.hub import MultiAgentStreamHub
        from .runtime.spec import StreamSpec

        n_steps = _req_bounded_int("n_steps", arguments.get("n_steps", 40), 5, 5000)
        inject_raw = arguments.get("inject_nan_channel", True)
        if not isinstance(inject_raw, bool):
            return {"error": "inject_nan_channel must be a boolean"}, True
        inject_nan = inject_raw
        with MultiAgentStreamHub(3, window=16) as hub:
            a0 = hub.subscribe(StreamSpec(name="agent_clean", channels=(0, 1)))
            a1 = hub.subscribe(StreamSpec(name="agent_all", channels=(0, 1, 2)))
            for t in range(n_steps // 2):
                v2 = float("nan") if inject_nan else (3.0 + 0.1 * t)
                hub.ingest(t, [math.sin(0.2 * t), math.cos(0.2 * t), v2])
            a2 = hub.hot_add_agent(StreamSpec(name="agent_hot", channels=(0, 1)))
            for t in range(n_steps // 2, n_steps):
                v2 = float("nan") if inject_nan else (3.0 + 0.1 * t)
                hub.ingest(t, [math.sin(0.2 * t), math.cos(0.2 * t), v2])
            return {
                "steps": n_steps,
                "agents": {
                    "clean_id": a0.agent_id,
                    "all_id": a1.agent_id,
                    "hot_added_id": a2.agent_id,
                    "shared_group_clean_and_hot": a0.group_id == a2.group_id,
                },
                "swarm_e_value": hub.swarm_e_value(),
                "ledger_verified": hub.verify_ledger(),
                "ledger_entries": len(hub.ledger.entries),
                "ebr": hub.ebr_stats(),
            }, False

    raise KeyError(f"Unknown tool: {name}")


def _validate_tool_schema(tool_name: str, tool_args: Mapping[str, Any]) -> str | None:
    """Enforce required fields and additionalProperties=False against TOOL_DEFINITIONS."""
    schema = _TOOL_SCHEMAS[tool_name]
    allowed = set(schema.get("properties", {}).keys())
    unknown = sorted(set(tool_args.keys()) - allowed)
    if unknown and schema.get("additionalProperties") is False:
        return f"Invalid params: unknown argument(s) {unknown} for tool {tool_name!r}"
    missing = [req for req in schema.get("required", ()) if req not in tool_args]
    if missing:
        return f"Invalid params: missing required argument(s) {missing} for tool {tool_name!r}"
    return None


def _is_valid_rpc_id(val: Any) -> bool:
    if val is None or isinstance(val, str):
        return True
    if isinstance(val, bool):
        return False
    if isinstance(val, int):
        return True
    if isinstance(val, float):
        return math.isfinite(val)
    return False


def handle_request(msg: Any) -> dict[str, Any] | None:
    """Handle one parsed JSON-RPC 2.0 message, returning a response dict or None."""
    if not isinstance(msg, Mapping) or msg.get("jsonrpc") != "2.0":
        raw_id = msg.get("id") if isinstance(msg, Mapping) else None
        return {
            "jsonrpc": "2.0",
            "id": raw_id if _is_valid_rpc_id(raw_id) else None,
            "error": {"code": -32600, "message": "Invalid Request"},
        }

    method = msg.get("method")
    has_id = "id" in msg
    msg_id = msg.get("id")
    if has_id and not _is_valid_rpc_id(msg_id):
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32600,
                "message": "Invalid Request: 'id' must be a string, number, or null",
            },
        }
    if not isinstance(method, str) or not method:
        return {
            "jsonrpc": "2.0",
            "id": msg_id if has_id else None,
            "error": {"code": -32600, "message": "Invalid Request: 'method' must be a string"},
        }

    # Notifications have a valid method, no id, and expect no response
    if not has_id:
        return None

    if "params" not in msg or msg["params"] is None:
        params: Any = {}
    else:
        params = msg["params"]

    if method == "initialize":
        if not isinstance(params, Mapping):
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32602, "message": "Invalid params: expected object"},
            }
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "tsjit-mcp", "version": __version__},
            },
        }

    if method == "ping":
        if not isinstance(params, Mapping):
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32602, "message": "Invalid params: expected object"},
            }
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    if method == "tools/list":
        if not isinstance(params, Mapping):
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32602, "message": "Invalid params: expected object"},
            }
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOL_DEFINITIONS}}

    if method == "tools/call":
        if not isinstance(params, Mapping) or not isinstance(params.get("name"), str):
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32602, "message": "Invalid params: tool 'name' required"},
            }
        tool_name = params["name"]
        if "arguments" not in params or params["arguments"] is None:
            tool_args: Any = {}
        else:
            tool_args = params["arguments"]
        if not isinstance(tool_args, Mapping):
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32602,
                    "message": "Invalid params: 'arguments' must be an object",
                },
            }
        if tool_name not in _TOOL_SCHEMAS:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }
        schema_err = _validate_tool_schema(tool_name, tool_args)
        if schema_err is not None:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32602, "message": schema_err},
            }
        try:
            payload, is_err = _call_tool(tool_name, tool_args)
        except KeyError:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({"error": str(exc)}, allow_nan=False),
                        }
                    ],
                    "isError": True,
                },
            }
        safe_payload = _sanitize_json(payload)
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(safe_payload, sort_keys=True, allow_nan=False),
                    }
                ],
                "isError": is_err,
            },
        }

    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


class _FrameError(ValueError):
    """Framing or JSON parse error carrying response framing mode and abort flag."""

    def __init__(self, message: str, *, framed: bool, fatal: bool = False) -> None:
        super().__init__(message)
        self.framed = framed
        self.fatal = fatal


def _is_framed_header_line(stripped: bytes) -> bool:
    """Enter framed mode ONLY on explicit Content-Length or Content-Type headers."""
    if b":" not in stripped:
        return False
    header_name = stripped.split(b":", 1)[0].strip().lower()
    return header_name in (b"content-length", b"content-type")


def _read_capped_line(stream: BinaryIO, *, framed: bool) -> bytes:
    line = stream.readline(_MAX_LINE_BYTES + 1)
    if len(line) > _MAX_LINE_BYTES:
        raise _FrameError(
            f"Line length exceeds maximum frame limit ({MAX_FRAME_BYTES} bytes)",
            framed=framed,
            fatal=True,
        )
    return line


def _read_exact(stream: BinaryIO, length: int) -> bytes | None:
    """Read exactly `length` bytes even across partial/chunked pipe or socket reads."""
    if length == 0:
        return b""
    chunks: list[bytes] = []
    remaining = length
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


def _read_message(stream: BinaryIO) -> tuple[Any, bool]:
    """Read one JSON-RPC message; returns (parsed_obj_or_EOF, use_content_length_framing)."""
    while True:
        line = _read_capped_line(stream, framed=False)
        if not line:
            return _EOF, False
        stripped = line.strip()
        if not stripped:
            continue
        if _is_framed_header_line(stripped):
            headers: dict[bytes, bytes] = {}
            cur = stripped
            n_headers = 0
            while cur:
                n_headers += 1
                if n_headers > _MAX_HEADER_LINES:
                    raise _FrameError("Too many header lines", framed=True, fatal=True)
                if b":" not in cur:
                    raise _FrameError("Malformed header line", framed=True, fatal=True)
                k, v = cur.split(b":", 1)
                hk = k.strip().lower()
                if hk == b"content-length" and b"content-length" in headers:
                    raise _FrameError("Duplicate Content-Length header", framed=True, fatal=True)
                headers[hk] = v.strip()
                hline = _read_capped_line(stream, framed=True)
                if not hline:
                    return _EOF, True
                if hline in (b"\r\n", b"\n"):
                    break
                cur = hline.strip()
            cl_raw = headers.get(b"content-length")
            if cl_raw is None:
                raise _FrameError("Missing Content-Length header", framed=True, fatal=True)
            if not cl_raw.isdigit():
                raise _FrameError(f"Invalid Content-Length: {cl_raw!r}", framed=True, fatal=True)
            # Check digit count BEFORE int() so >4300-digit headers cannot bypass _FrameError
            # via Python's str-to-int ValueError and leak smuggled body bytes!
            if len(cl_raw) > 10:
                raise _FrameError(
                    f"Content-Length exceeds maximum {MAX_FRAME_BYTES}",
                    framed=True,
                    fatal=True,
                )
            length = int(cl_raw)
            if length < 0 or length > MAX_FRAME_BYTES:
                raise _FrameError(
                    f"Content-Length {length} exceeds maximum {MAX_FRAME_BYTES}",
                    framed=True,
                    fatal=True,
                )
            body = _read_exact(stream, length)
            if body is None:
                return _EOF, True
            try:
                parsed = json.loads(
                    body.decode("utf-8"),
                    parse_constant=_reject_non_rfc_constant,
                )
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                raise _FrameError(f"Parse error: {exc}", framed=True, fatal=False) from exc
            return parsed, True

        try:
            parsed = json.loads(
                stripped.decode("utf-8"),
                parse_constant=_reject_non_rfc_constant,
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            raise _FrameError(f"Parse error: {exc}", framed=False, fatal=False) from exc
        return parsed, False


def _write_message(stream: BinaryIO, obj: Mapping[str, Any], framed: bool) -> None:
    safe_obj = _sanitize_json(obj)
    data = json.dumps(safe_obj, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if framed:
        header = f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
        stream.write(header + data)
    else:
        stream.write(data + b"\n")
    stream.flush()


def serve(stdin: BinaryIO, stdout: BinaryIO) -> int:
    """Serve JSON-RPC 2.0 MCP requests from stdin to stdout until EOF or fatal frame error."""
    while True:
        try:
            msg, framed = _read_message(stdin)
        except _FrameError as exc:
            err = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": str(exc) or "Parse error"},
            }
            _write_message(stdout, err, exc.framed)
            if exc.fatal:
                # Abort stream loop cleanly so unread oversized/desynced bytes can NEVER
                # be smuggled as subsequent JSON-RPC messages!
                return 0
            continue
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            err = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
            _write_message(stdout, err, False)
            continue
        if msg is _EOF:
            return 0
        resp = handle_request(msg)
        if resp is not None:
            _write_message(stdout, resp, framed)


def serve_stdio() -> int:
    return serve(sys.stdin.buffer, sys.stdout.buffer)
