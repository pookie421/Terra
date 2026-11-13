"""State-bound assistant with local or hosted prose and deterministic app plans."""

from __future__ import annotations

import json
import os
import math
import re
import time
import ipaddress
from pathlib import Path
from urllib.parse import urlsplit
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
    """Select the configured assistant service and ground briefings in scan evidence."""

    def __init__(self, scan_service: Optional[GlobalScanService] = None, ollama_url: Optional[str] = None) -> None:
        self.scan_service = scan_service or GlobalScanService()
        self._explicit_ollama_url = ollama_url
        root=Path(__file__).resolve().parents[2]
        self._endpoint_config_paths=[root/'artifacts/qwen_copilot_endpoint.json',
            root/'windows_5070ti_trainer/artifacts/qwen_copilot_endpoint.json',
            root/'training/artifacts/qwen_copilot_endpoint.json']
        self._availability_error=None

    @property
    def ollama_url(self):
        explicit=self._explicit_ollama_url or os.environ.get('TERRA_QWEN_URL')
        if explicit:return explicit.rstrip('/')
        # A training job can finish after the application has already started.
        # Only its small local loopback rendezvous file is auto-discovered.
        candidates=[]
        for path in self._endpoint_config_paths:
            try:
                if path.stat().st_size>8192:continue
                data=json.loads(path.read_text(encoding='utf-8'))
                if data.get('source')!='local_trained_adapter_server':continue
                url=data.get('url');parts=urlsplit(url)
                host=parts.hostname
                loopback=host=='localhost'
                if not loopback:
                    try:loopback=ipaddress.ip_address(host).is_loopback
                    except ValueError:continue
                if (parts.scheme!='http' or not loopback or parts.username or parts.password or
                    parts.path not in ('','/') or parts.query or parts.fragment or not parts.port):continue
                candidates.append((path.stat().st_mtime,url.rstrip('/')))
            except (OSError,ValueError,TypeError,AttributeError):continue
        return max(candidates,key=lambda item:item[0])[1] if candidates else 'http://localhost:11434'

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
        scan_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
        selected_h3_index: Optional[str] = None,
        history: Optional[List[Dict[str,str]]] = None,
        autonomous: bool = False,
        active_prediction_model: str = 'terra_physics_ml',
    ) -> Dict[str, Any]:
        if (scan_id is None)!=(scenario_id is None):raise ValueError('scan_id and scenario_id must be supplied together')
        if scan_id is not None:
            from .state_copilot import chat_active
            return await chat_active(self,message,scan_id,scenario_id,sliders or {},agency_mode,selected_h3_index,history,autonomous,active_prediction_model)
        if autonomous:
            from .app_actions import MODELS, plan_actions, validate_actions
            from .state_copilot import planning_reply
            if active_prediction_model not in MODELS:raise ValueError('Unknown active prediction model')
            if sliders:raise ValueError('Autonomous chat cannot apply manual forcing')
            planned=plan_actions(message)
            if not planned and current_query:
                planned=validate_actions([{'type':'search_location','query':current_query}])
            return planning_reply(message,planned)
        # New-location chat acquires one scan, then uses the same state-bound
        # co-pilot protocol as the map. It never asks the LLM to invent a scan.
        parsed=parse_scenario_request(message,current_query,sliders)
        scan=await self._run_scan(self._scan_arguments(parsed),agency_mode)
        from .state_copilot import chat_active
        return await chat_active(self,message,scan['scan_id'],scan['scenario_id'],{},agency_mode,selected_h3_index,history,autonomous,active_prediction_model)
    async def _available_model(self) -> Tuple[Optional[str], int]:
        self._availability_error=None
        from .model_client import assistant_config, AssistantConfigurationError
        try:config=assistant_config()
        except AssistantConfigurationError:
            self._availability_error='AssistantConfigurationError'
            return None,3
        if config.provider=='openai_compatible':
            if not config.configured:
                self._availability_error='HostedAssistantNotConfigured'
                return None,3
            # Configuration is not proof of provider availability or prose quality.
            # Those are checked by the actual bounded, verified chat request.
            return config.model,0
        try:
            async with httpx.AsyncClient(timeout=1.25) as client:
                response = await client.get(self.ollama_url + "/api/tags")
                response.raise_for_status()
                names = [str(item.get("name", "")) for item in response.json().get("models", [])]
            if config.model:
                if config.model in names:return config.model,2
                self._availability_error='ConfiguredLocalModelNotFound'
                return None,3
            custom = next((name for name in names if name.split(":", 1)[0].lower() == "terra-shield-qwen"), None)
            if custom:
                return custom, 1
            qwen = next((name for name in names if "qwen2.5" in name.lower()), None)
            if qwen:
                return qwen, 2
        except Exception as exc:
            self._availability_error=type(exc).__name__
        return None, 3

    async def _ollama_tool_roundtrip(
        self,
        model_name: str,
        message: str,
        parsed: Mapping[str, Any],
        agency_mode: str,
        tool_calls: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "Use the supplied live tools for all measurements. Do not invent coordinates, probabilities, causes, or runout. Call scan_global_location for the requested location and perturbations. Then call inspect/trace only for an H3 index returned by the scan. Write a grounded final answer using only returned telemetry."},
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
        return scan_data, assistant_message.get("content", "")

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
        if scan.get('assessment_mode')=='PHYSICS_INFORMED_REFERENCE':
            from .state_copilot import context,template
            return template(context(scan))
        if scan.get('assessment_mode')=='SOURCE_EVIDENCE_AND_EXPLICIT_GROUND_INPUTS':
            physical=scan.get('physical_assessment',{});stat=scan.get('statistical_assessment',{})
            cell=stat.get('hotspot') or {}
            lines=['TERRA-SHIELD source assessment',
                'Operational hazard verdict: unavailable; the inventory classifier is not locally or prospectively validated.']
            if cell:
                lines.append('Inventory classifier output: {} at {}, {} (H3 {}).'.format(
                    TerraShieldQwenAgent._fmt(cell.get('score')),TerraShieldQwenAgent._fmt(cell.get('latitude'),6),
                    TerraShieldQwenAgent._fmt(cell.get('longitude'),6),cell.get('h3_index')))
                same=next((r for r in scan.get('cell_comparisons',[]) if r['h3_index']==cell.get('h3_index')),{})
                lines.append('Factor of safety at that same cell: {}.'.format(TerraShieldQwenAgent._fmt(same.get('fs_dynamic'))))
            lines.append('Ground model: '+str(physical.get('reason') or physical.get('scope','unavailable')))
            lines.append('FOSM, runout speed and failure time are unavailable without supported inputs; no zero values substitute for missing evidence.')
            return '\n'.join(lines)
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
            numeric=float(value)
            if not math.isfinite(numeric):return 'unavailable'
            if numeric==0:return '0'
            if abs(numeric)<.5*10**(-digits):return format(numeric,'.2e')
            return format(numeric, ".{}f".format(digits))
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _guard_briefing(candidate, telemetry, agency_mode, magnitude_only=False):
        if re.search(r'\b(?:failure is certain|certain failure|guaranteed safe|evacuate immediately)\b',candidate or '',re.I):
            return TerraShieldQwenAgent._briefing(telemetry,agency_mode,magnitude_only),True,'VERIFIED_TEMPLATE'
        if telemetry.get('assessment_mode')=='SOURCE_EVIDENCE_AND_EXPLICIT_GROUND_INPUTS':
            canonical=TerraShieldQwenAgent._briefing(telemetry,agency_mode,magnitude_only)
            # Matching numbers cannot validate assertions of safety, certainty,
            # causation or evacuation need. Current live assessments use a
            # deterministic evidence summary instead of unverified model prose.
            return canonical,True,'SOURCE_EVIDENCE_TEMPLATE'
        cleaned=re.sub(r'<(?:ui_action|tool_call)\b[^>]*>.*?</(?:ui_action|tool_call)>','',candidate or '',flags=re.S|re.I).strip()
        if cleaned and TerraShieldQwenAgent._numeric_parity_audit(cleaned,telemetry):
            return cleaned,True,'VERIFIED_LLM_PROSE'
        canonical=TerraShieldQwenAgent._briefing(telemetry,agency_mode,magnitude_only)
        canonical=_ANY_TAG.sub('',_XML_TAG.sub('',canonical)).strip()
        if not TerraShieldQwenAgent._numeric_parity_audit(canonical,telemetry):
            return 'Verified numeric briefing unavailable. Inspect structured telemetry.',False,'WITHHELD_INVALID_TEMPLATE'
        return canonical,True,'VERIFIED_TEMPLATE' if not cleaned else 'REPLACED_UNVERIFIED_PROSE'

    @staticmethod
    def _numeric_parity_audit(markdown: str, telemetry: Mapping[str, Any]) -> bool:
        available=[];strings=[]
        def collect(value):
            if isinstance(value,bool) or value is None:return
            if isinstance(value,(int,float)) and math.isfinite(float(value)):
                available.append(float(value))
                if 0<=value<=1:available.append(float(value)*100.)
            elif isinstance(value,str) and value:strings.append(value)
            elif isinstance(value,Mapping):
                for child in value.values():collect(child)
            elif isinstance(value,(list,tuple)):
                for child in value:collect(child)
        collect(telemetry)
        # Verbatim identifiers/location strings are not numeric measurements.
        for value in sorted(strings,key=len,reverse=True):
            if any(c.isdigit() for c in value):markdown=markdown.replace(value,'SOURCE_IDENTIFIER')
        markdown=re.sub(r'(?m)^\s*\d+[\.\)]\s+','',markdown)
        markdown=re.sub(r'\[T\+\d+m\]|Class\s+[1-7]\b|\b(?:24h|48h|7d|30m)\b|\b90%','',markdown)
        for token in _NUMERIC_TOKEN.findall(markdown):
            precision=len(token.partition('.')[2]) if '.' in token else 0
            if not any(abs(float(token)-value)<=max(10**(-precision),.05) for value in available):return False
        return True
