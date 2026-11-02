"""FastAPI endpoints for the TERRA-SHIELD scan and local agent."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from xml.etree.ElementTree import Element, SubElement, tostring

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from backend.api.serialization import sanitize_json
from backend.api.scan_service import GlobalScanService
from backend.llm.qwen_agent import TerraShieldQwenAgent

ROOT = Path(__file__).resolve().parents[2]
CAP_NS = "urn:oasis:names:tc:emergency:cap:1.2"

app = FastAPI(title="TERRA-SHIELD Global Landslide API", version="0.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

scan_service = GlobalScanService()
agent = TerraShieldQwenAgent(scan_service=scan_service)


class AgentChatRequest(BaseModel):
    message: str
    current_query: str = ""
    agency_mode: str = "LOCAL_EOC"
    sliders: Dict[str, Any] = Field(default_factory=dict)


class CapXmlRequest(BaseModel):
    scan: Optional[Dict[str, Any]] = None
    agency_mode: str = "LOCAL_EOC"


@app.get("/api/v1/scan")
async def get_scan(
    query: str = "Chooralmala, Wayanad",
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    rain_delta_mm: float = 0.0,
    storm_duration_hr: float = 24.0,
    pga_override_g: float = 0.0,
    snowmelt_temp_surge_c: float = 0.0,
    burn_severity_override: float = 0.0,
    agency_mode: str = "LOCAL_EOC",
    inject_forecast_storm_mm: Optional[float] = Query(None, ge=0),
    storm_arrival_hour: int = Query(18, ge=0, le=71),
    storm_duration_hours: int = Query(12, ge=1, le=72),
) -> Dict[str, Any]:
    try:
        return sanitize_json(await scan_service.scan(
            location_query=query,
            lat=lat,
            lon=lon,
            rain_delta_mm=rain_delta_mm,
            storm_duration_hr=storm_duration_hr,
            pga_override_g=pga_override_g,
            snowmelt_temp_surge_c=snowmelt_temp_surge_c,
            burn_severity_override=burn_severity_override,
            agency_mode=agency_mode,
            inject_forecast_storm_mm=inject_forecast_storm_mm,
            storm_arrival_hour=storm_arrival_hour,
            storm_duration_hours=storm_duration_hours,
        ))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail="scan failed: {}".format(type(exc).__name__))


@app.post("/api/v1/agent/chat")
async def agent_chat(request: AgentChatRequest) -> Dict[str, Any]:
    try:
        return sanitize_json(await agent.chat(
            request.message,
            current_query=request.current_query,
            agency_mode=request.agency_mode,
            sliders=request.sliders,
        ))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail="agent scan failed: {}".format(type(exc).__name__))


@app.get("/api/v1/validation/report")
async def validation_report() -> Dict[str, Any]:
    path = ROOT / "artifacts" / "scientific_validation_report.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="scientific validation report has not been generated")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="scientific validation report is unreadable: {}".format(type(exc).__name__))
    return sanitize_json(payload)


@app.post("/api/v1/export/cap-xml")
async def export_cap_xml(request: CapXmlRequest) -> Response:
    scan = request.scan
    if scan is None:
        scan = await scan_service.scan(agency_mode=request.agency_mode)
    scan = sanitize_json(scan)
    summary = scan.get("summary") or {}
    location = scan.get("location") or {}
    level = str(summary.get("wmo_alert_level") or "GREEN").upper()
    now = datetime.now(timezone.utc).replace(microsecond=0)
    expires = now + timedelta(hours=6)

    alert = Element("alert", {"xmlns": CAP_NS})
    _cap_text(alert, "identifier", str(uuid.uuid4()))
    _cap_text(alert, "sender", "terra-shield.local")
    _cap_text(alert, "sent", now.isoformat().replace("+00:00", "Z"))
    _cap_text(alert, "status", "Test")
    _cap_text(alert, "msgType", "Alert")
    _cap_text(alert, "scope", "Private")
    _cap_text(alert, "addresses", "terra-shield.local")
    info = SubElement(alert, "info")
    _cap_text(info, "language", "en-US")
    _cap_text(info, "category", "Geo")
    _cap_text(info, "event", "Landslide hazard monitoring")
    _cap_text(info, "responseType", "Monitor")
    _cap_text(info, "urgency", "Immediate" if level == "RED" else ("Expected" if level == "ORANGE" else "Future"))
    _cap_text(info, "severity", "Extreme" if level == "RED" else ("Severe" if level == "ORANGE" else ("Moderate" if level == "YELLOW" else "Minor")))
    probability = summary.get("max_p_failure")
    _cap_text(info, "certainty", "Likely" if isinstance(probability, (int, float)) and probability >= 0.65 else "Possible")
    _cap_text(info, "effective", now.isoformat().replace("+00:00", "Z"))
    _cap_text(info, "expires", expires.isoformat().replace("+00:00", "Z"))
    _cap_text(info, "senderName", "TERRA-SHIELD local prototype")
    _cap_text(info, "headline", "{} landslide hazard level {}".format(location.get("display_name") or location.get("query") or "Location", level))
    _cap_text(info, "description", "Prototype model output for monitoring and review. This test message is not an operational warning.")
    _cap_text(info, "instruction", "Review current source status and local authority guidance before taking action.")
    area = SubElement(info, "area")
    _cap_text(area, "areaDesc", str(location.get("display_name") or location.get("query") or "Scanned location"))
    top = summary.get("kinematics_top_hotspot") or {}
    geo_features = (scan.get("h3_geojson") or {}).get("features") or []
    matching_feature = next(
        (feature for feature in geo_features if (feature.get("properties") or {}).get("h3_index") == top.get("h3_index")),
        geo_features[0] if geo_features else {},
    )
    ring = ((matching_feature.get("geometry") or {}).get("coordinates") or [])
    if ring and ring[0]:
        pairs = []
        for point in ring[0]:
            if len(point) >= 2:
                pairs.append("{:.6f},{:.6f}".format(float(point[1]), float(point[0])))
        if pairs:
            _cap_text(area, "polygon", " ".join(pairs))
    if not any(child.tag in ("polygon", "circle", "geocode") for child in list(area)):
        latitude = top.get("latitude", location.get("latitude"))
        longitude = top.get("longitude", location.get("longitude"))
        if latitude is not None and longitude is not None:
            _cap_text(area, "circle", "{:.6f},{:.6f} 0.5".format(float(latitude), float(longitude)))
    xml = tostring(alert, encoding="utf-8", xml_declaration=True)
    return Response(content=xml, media_type="application/cap+xml; charset=utf-8")


def _cap_text(parent: Element, tag: str, value: str) -> Element:
    node = SubElement(parent, tag)
    node.text = value
    return node

