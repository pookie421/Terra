"""Three-tier Ollama tool agent with a deterministic live-scan fallback."""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import httpx

from backend.api.scan_service import GlobalScanService


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "scan_global_location",
            "description": "Run a live global landslide scan with the requested scenario perturbations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location_query": {"type": "string"},
                    "rain_delta_mm": {"type": "number", "default": 0.0},
                    "storm_duration_hr": {"type": "number", "default": 24.0},
                    "pga_override_g": {"type": "number", "default": 0.0},
                    "snowmelt_temp_surge_c": {"type": "number", "default": 0.0},
                    "burn_severity_override": {"type": "number", "default": 0.0},
                },
                "required": ["location_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_hexagon_causal_physics",
            "description": "Inspect the active scan's physics and causal breakdown for one H3 cell.",
            "parameters": {"type": "object", "properties": {"h3_index": {"type": "string"}}, "required": ["h3_index"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trace_runout_and_evacuation",
            "description": "Trace a provisional Voellmy runout path and list mapped exposure intersections.",
            "parameters": {"type": "object", "properties": {"h3_index": {"type": "string"}}, "required": ["h3_index"]},
        },
    },
]

_NUMBER = r"[-+]?\d+(?:\.\d+)?"
_NUMERIC_TOKEN = re.compile(r"(?<![\w])" + _NUMBER + r"(?:[eE][-+]?\d+)?")
_XML_TAG = re.compile(r"<\s*/?\s*(?:ui_action|tool_call)\b[^>]*>", re.IGNORECASE)
_ANY_TAG = re.compile(r"<\s*/?[A-Za-z][^>]*>")


def parse_scenario_request(
    message: str, current_query: str = "", sliders: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Extract only explicit perturbations; Mw is never converted into PGA."""
    text = str(message or "")
    lowered = text.lower()
    values: Dict[str, Any] = {
        "rain_delta_mm": 0.0,
        "storm_duration_hr": 24.0,
        "pga_override_g": 0.0,
        "snowmelt_temp_surge_c": 0.0,
        "burn_severity_override": 0.0,
    }
    explicit = set()
    rain_matches = re.findall("(" + _NUMBER + r")\s*mm\b", lowered)
    if rain_matches:
        values["rain_delta_mm"] = max(0.0, float(rain_matches[-1]))
        explicit.add("rain_delta_mm")
    pga_matches = re.findall("(" + _NUMBER + r")\s*g\b", lowered)
    if pga_matches:
        values["pga_override_g"] = max(0.0, float(pga_matches[-1]))
        explicit.add("pga_override_g")
    temp_matches = re.findall("(" + _NUMBER + r")\s*(?:°\s*c|degrees?\s*celsius|celsius)\b", lowered)
    if temp_matches:
        values["snowmelt_temp_surge_c"] = max(0.0, float(temp_matches[-1]))
        explicit.add("snowmelt_temp_surge_c")
    burn_matches = re.findall("(" + _NUMBER + r")\s*%", lowered)
    if burn_matches:
        values["burn_severity_override"] = min(1.0, max(0.0, float(burn_matches[-1]) / 100.0))
        explicit.add("burn_severity_override")
    duration_matches = re.findall("(" + _NUMBER + r")\s*(?:h|hr|hrs|hours?)\b", lowered)
    if duration_matches:
        values["storm_duration_hr"] = max(0.01, float(duration_matches[-1]))
        explicit.add("storm_duration_hr")

    for key in values:
        if sliders and key in sliders:
            try:
                slider_value = float(sliders[key])
                if not math.isfinite(slider_value):
                    raise ValueError("slider must be finite")
                if key == "burn_severity_override":
                    slider_value = min(1.0, max(0.0, slider_value))
                elif key == "storm_duration_hr":
                    slider_value = max(0.01, slider_value)
                else:
                    slider_value = max(0.0, slider_value)
                if not math.isfinite(slider_value):
                    raise ValueError("slider must be finite")
                if key not in explicit:
                    values[key] = slider_value
                    explicit.add(key)
            except (TypeError, ValueError):
                continue

    query = str(current_query or "").strip()
    from backend.api.scan_service import known_location_coordinates
    candidate = _location_from_message(text)
    if candidate and (not query or known_location_coordinates(candidate) is not None or re.search(r"\bscan\b",text,re.I)):
        query = candidate
    values["location_query"] = query or "Chooralmala, Wayanad"
    values["magnitude_only"] = bool(re.search(r"\bM\s*w\s*" + _NUMBER, text, re.IGNORECASE)) and not pga_matches
    values["_explicit_fields"] = sorted(explicit)
    return values


def _location_from_message(message: str) -> str:
    text = message.strip()
    text = re.sub(r"\bM\s*w\s*" + _NUMBER, " ", text, flags=re.IGNORECASE)
    text = re.sub(_NUMBER + r"\s*(?:mm\b|g\b|°\s*c\b|degrees?\s*celsius\b|celsius\b|%|M\s*w\s*\d+(?:\.\d+)?)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:scan|check|assess|estimate|show|tell\s+me|landslide|risk|hazard|with|under|after|for|at|in|around|near|the|and|burn|scar|rain|rainfall|earthquake|snowmelt|warming|temperature|please)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[+,:;()\[\]{}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .-/")
    return text


class TerraShieldQwenAgent:
    """Select a local Qwen model when available and ground every briefing in tools."""

    def __init__(self, scan_service: Optional[GlobalScanService] = None, ollama_url: Optional[str] = None) -> None:
        self.scan_service = scan_service or GlobalScanService()
        self.ollama_url = (ollama_url or "http://localhost:11434").rstrip("/")

    async def scan_global_location(
        self,
        location_query: str,
        rain_delta_mm: float = 0.0,
        storm_duration_hr: float = 24.0,
        pga_override_g: float = 0.0,
        snowmelt_temp_surge_c: float = 0.0,
        burn_severity_override: float = 0.0,
    ) -> Dict[str, Any]:
        return await self.scan_service.scan(
            location_query=location_query,
            rain_delta_mm=rain_delta_mm,
            storm_duration_hr=storm_duration_hr,
            pga_override_g=pga_override_g,
            snowmelt_temp_surge_c=snowmelt_temp_surge_c,
            burn_severity_override=burn_severity_override,
        )

    async def _run_scan(self, arguments: Mapping[str, Any], agency_mode: str) -> Dict[str, Any]:
        return await self.scan_service.scan(**dict(arguments), agency_mode=agency_mode)

    def inspect_hexagon_causal_physics(self, h3_index: str) -> Dict[str, Any]:
        return self.scan_service.inspect_hexagon(h3_index)

    def trace_runout_and_evacuation(self, h3_index: str) -> Dict[str, Any]:
        return self.scan_service.trace_runout(h3_index)

    async def chat(
        self,
        message: str,
        current_query: str = "",
        agency_mode: str = "LOCAL_EOC",
        sliders: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        parsed = parse_scenario_request(message, current_query, sliders)
        model_name, tier = await self._available_model()
        tool_calls: List[Dict[str, Any]] = []
        scan_data: Optional[Dict[str, Any]] = None
        if model_name:
            try:
                scan_data = await self._ollama_tool_roundtrip(
                    model_name, message, parsed, agency_mode, tool_calls
                )
            except Exception:
                # If Ollama disconnects mid-conversation, continue with live deterministic tools.
                model_name, tier = None, 3
        if scan_data is None:
            args = self._scan_arguments(parsed)
            started = time.perf_counter()
            scan_data = await self._run_scan(args, agency_mode)
            tool_calls.append({
                "tool_name": "scan_global_location",
                "arguments": args,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            })

        clean_markdown = self._briefing(scan_data, agency_mode, bool(parsed.get("magnitude_only")))
        clean_markdown = _XML_TAG.sub("", clean_markdown)
        clean_markdown = _ANY_TAG.sub("", clean_markdown).strip()
        clean_markdown, numeric_audit_passed, briefing_guard = self._guard_briefing(clean_markdown, scan_data, agency_mode, bool(parsed.get("magnitude_only")))
        location = scan_data.get("location") or {}
        ui_action = {
            "fly_to": [location.get("longitude"), location.get("latitude")],
            "zoom": 13.4,
            "sliders": {key: parsed[key] for key in (
                "rain_delta_mm", "storm_duration_hr", "pga_override_g",
                "snowmelt_temp_surge_c", "burn_severity_override",
            )},
            "agency_mode": agency_mode,
        }
        return {
            "clean_markdown": clean_markdown,
            "tool_calls": tool_calls,
            "ui_action": ui_action,
            "updated_scan_data": scan_data,
            "numeric_audit_passed": bool(numeric_audit_passed),
            "briefing_guard": briefing_guard,
            "model_tier": tier,
            "model_name": model_name,
        }

    async def _available_model(self) -> Tuple[Optional[str], int]:
        try:
            async with httpx.AsyncClient(timeout=1.25) as client:
                response = await client.get(self.ollama_url + "/api/tags")
                response.raise_for_status()
                names = [str(item.get("name", "")) for item in response.json().get("models", [])]
            custom = next((name for name in names if name.split(":", 1)[0].lower() == "terra-shield-qwen"), None)
            if custom:
                return custom, 1
            qwen = next((name for name in names if "qwen2.5" in name.lower()), None)
            if qwen:
                return qwen, 2
        except Exception:
            pass
        return None, 3

    async def _ollama_tool_roundtrip(
        self,
        model_name: str,
        message: str,
        parsed: Mapping[str, Any],
        agency_mode: str,
        tool_calls: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "Use the supplied live tools for all measurements. Do not invent coordinates, probabilities, causes, or runout. Call scan_global_location for the requested location and perturbations. Then call inspect/trace only for an H3 index returned by the scan. Final numbers are generated by the application from tool telemetry."},
            {"role": "user", "content": message or "Scan the current location."},
        ]
        scan_data: Optional[Dict[str, Any]] = None
        async with httpx.AsyncClient(timeout=45.0) as client:
            for _ in range(4):
                response = await client.post(self.ollama_url + "/api/chat", json={
                    "model": model_name, "messages": messages, "tools": TOOL_SCHEMAS, "stream": False,
                })
                response.raise_for_status()
                assistant_message = response.json().get("message") or {}
                messages.append(assistant_message)
                calls = assistant_message.get("tool_calls") or []
                if not calls:
                    break
                for call in calls:
                    function = call.get("function") or {}
                    name = str(function.get("name") or "")
                    arguments = function.get("arguments") or {}
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except ValueError:
                            arguments = {}
                    arguments = dict(arguments) if isinstance(arguments, Mapping) else {}
                    if name == "scan_global_location":
                        # All scenario values come from explicit user text/sliders or deterministic defaults.
                        # A language model cannot silently add a perturbation that the UI does not show.
                        arguments = self._scan_arguments(parsed)
                    elif name in ("inspect_hexagon_causal_physics", "trace_runout_and_evacuation") and scan_data is None:
                        arguments_scan = self._scan_arguments(parsed)
                        started = time.perf_counter()
                        scan_data = await self._run_scan(arguments_scan, agency_mode)
                        tool_calls.append({"tool_name": "scan_global_location", "arguments": arguments_scan, "latency_ms": round((time.perf_counter() - started) * 1000.0, 2)})
                    started = time.perf_counter()
                    try:
                        result = await self._dispatch(name, arguments, agency_mode)
                        if name == "scan_global_location":
                            scan_data = result
                        tool_result = result
                    except Exception as exc:
                        tool_result = {"error": "{}: {}".format(type(exc).__name__, exc)}
                    tool_calls.append({"tool_name": name, "arguments": arguments, "latency_ms": round((time.perf_counter() - started) * 1000.0, 2)})
                    messages.append({"role": "tool", "name": name, "content": json.dumps(tool_result, ensure_ascii=False, default=str, allow_nan=False)})
        return scan_data

    async def _dispatch(self, name: str, arguments: Mapping[str, Any], agency_mode: str) -> Dict[str, Any]:
        if name == "scan_global_location":
            return await self._run_scan(self._clean_scan_arguments(arguments), agency_mode)
        if name == "inspect_hexagon_causal_physics":
            return self.inspect_hexagon_causal_physics(str(arguments.get("h3_index", "")))
        if name == "trace_runout_and_evacuation":
            return self.trace_runout_and_evacuation(str(arguments.get("h3_index", "")))
        raise ValueError("unknown tool: {}".format(name))

    @staticmethod
    def _clean_scan_arguments(arguments: Mapping[str, Any]) -> Dict[str, Any]:
        allowed = {
            "location_query", "rain_delta_mm", "storm_duration_hr", "pga_override_g",
            "snowmelt_temp_surge_c", "burn_severity_override",
        }
        return {key: value for key, value in arguments.items() if key in allowed}

    @staticmethod
    def _scan_arguments(parsed: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "location_query": parsed.get("location_query") or "Chooralmala, Wayanad",
            "rain_delta_mm": float(parsed.get("rain_delta_mm", 0.0)),
            "storm_duration_hr": float(parsed.get("storm_duration_hr", 24.0)),
            "pga_override_g": float(parsed.get("pga_override_g", 0.0)),
            "snowmelt_temp_surge_c": float(parsed.get("snowmelt_temp_surge_c", 0.0)),
            "burn_severity_override": float(parsed.get("burn_severity_override", 0.0)),
        }

    @staticmethod
    def _briefing(scan: Mapping[str, Any], agency_mode: str, magnitude_only: bool = False) -> str:
        location = scan.get("location") or {}
        summary = scan.get("summary") or {}
        metrics = summary.get("meteorological_metrics") or {}
        hotspot = summary.get("kinematics_top_hotspot") or {}
        name = location.get("display_name") or location.get("query") or "Requested location"
        lines = ["## TERRA-SHIELD scan: {}".format(name), "", "- **Alert level:** {}".format(summary.get("wmo_alert_level", "UNAVAILABLE"))]
        if agency_mode.upper() == "WMO_MET":
            lines.extend([
                "- **Daily rainfall scenario:** {} mm; storm duration: {} h".format(
                    TerraShieldQwenAgent._fmt(metrics.get("rain_24h_mm_scenario")),
                    TerraShieldQwenAgent._fmt(metrics.get("storm_duration_hr")),
                ),
                "- **Scenario PGA:** {} g; air temperature: {} °C; burn severity: {}".format(
                    TerraShieldQwenAgent._fmt(metrics.get("pga_g_scenario")),
                    TerraShieldQwenAgent._fmt(metrics.get("air_temperature_c_scenario")),
                    TerraShieldQwenAgent._fmt(metrics.get("burn_severity_override")),
                ),
                "- **P95 failure probability:** {}; **P05 dynamic factor of safety:** {}".format(
                    TerraShieldQwenAgent._fmt(summary.get("max_p_failure")),
                    TerraShieldQwenAgent._fmt(summary.get("min_fs_dynamic")),
                ),
                "- **Critical hotspots:** {}".format(summary.get("critical_hotspots_count", 0)),
            ])
        else:
            lines.extend([
                "- **P95 failure probability:** {}; **P05 dynamic factor of safety:** {}".format(
                    TerraShieldQwenAgent._fmt(summary.get("max_p_failure")),
                    TerraShieldQwenAgent._fmt(summary.get("min_fs_dynamic")),
                ),
                "- **Top exposed hotspot:** H3 `{}`; slope {}°; peak velocity {} m/s; {}.".format(
                    hotspot.get("h3_index", "unavailable"),
                    TerraShieldQwenAgent._fmt(hotspot.get("slope_deg")),
                    TerraShieldQwenAgent._fmt(hotspot.get("peak_velocity_ms")),
                    hotspot.get("failure_mode", "mode unavailable"),
                ),
                "- **Hotspot cause:** {}".format(hotspot.get("dominant_cause", summary.get("dominant_cause", "unavailable"))),
                "- **Critical hotspots:** {}; mapped households exposed: {}".format(
                    summary.get("critical_hotspots_count", 0),
                    summary.get("total_households_exposed") if summary.get("total_households_exposed") is not None else "unavailable",
                ),
            ])
        cause = hotspot.get("causal_breakdown_pct") or summary.get("causal_breakdown_pct") or {}
        if cause:
            top_causes = sorted(cause.items(), key=lambda item: float(item[1]), reverse=True)[:3]
            lines.append("- **Causal breakdown:** {}".format(", ".join("{} {}%".format(key, TerraShieldQwenAgent._fmt(value, 1)) for key, value in top_causes)))
        numbers = hotspot.get("numbers_that_matter") or summary.get("numbers_that_matter") or []
        if numbers:
            lines.append("- **Numbers that matter:** {}".format("; ".join(
                "{} {} {}".format(item.get("metric"), TerraShieldQwenAgent._fmt(item.get("value")), item.get("unit", ""))
                for item in numbers[:3]
            )))
        lines.extend([
            "",
            "Probability is an exploratory model/physics blend when a model is available; source gaps and fallback inputs are listed in the scan payload.",
            "Runout routes use display defaults that are not calibrated for operational evacuation decisions.",
        ])
        if magnitude_only:
            lines.append("Mw was noted, but no site PGA was estimated because magnitude alone does not specify shaking at the location.")
        return "\n".join(lines)

    @staticmethod
    def _fmt(value: Any, digits: int = 3) -> str:
        if value is None:
            return "unavailable"
        try:
            return format(float(value), ".{}f".format(digits))
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _guard_briefing(candidate, telemetry, agency_mode, magnitude_only=False):
        # Model prose never reaches the API: the approved output is a tool-bound template.
        canonical=TerraShieldQwenAgent._briefing(telemetry,agency_mode,magnitude_only)
        canonical=_ANY_TAG.sub('',_XML_TAG.sub('',canonical)).strip()
        valid=TerraShieldQwenAgent._numeric_parity_audit(canonical,telemetry)
        if not valid:
            return ('Verified numeric briefing unavailable. Inspect structured telemetry and source status.',
                    True, 'WITHHELD_INVALID_TEMPLATE')
        accepted=candidate==canonical and TerraShieldQwenAgent._numeric_parity_audit(candidate,telemetry)
        return canonical,True,'VERIFIED_TEMPLATE' if accepted else 'REPLACED_UNVERIFIED_PROSE'

    @staticmethod
    def _numeric_parity_audit(markdown: str, telemetry: Mapping[str, Any]) -> bool:
        available=[];strings=[]
        def collect(value):
            if isinstance(value,bool) or value is None:return
            if isinstance(value,(int,float)) and math.isfinite(float(value)):available.append(float(value))
            elif isinstance(value,str) and value:strings.append(value)
            elif isinstance(value,Mapping):
                for child in value.values():collect(child)
            elif isinstance(value,(list,tuple)):
                for child in value:collect(child)
        collect(telemetry)
        # Verbatim identifiers/location strings are not numeric measurements.
        for value in sorted(strings,key=len,reverse=True):
            if any(c.isdigit() for c in value):markdown=markdown.replace(value,'SOURCE_IDENTIFIER')
        for token in _NUMERIC_TOKEN.findall(markdown):
            precision=len(token.partition('.')[2]) if '.' in token else 0
            if not any(float(token)==float(format(value,'.{}f'.format(precision))) for value in available):return False
        return True
