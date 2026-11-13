"""FastAPI endpoints for the TERRA-SHIELD scan and local agent."""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field, ConfigDict

from backend.api.serialization import sanitize_json
from backend.api.deployment import (AdmissionBudget, AdmissionMiddleware, FrontendFiles,
                                    concurrency_limit, frontend_directory, readiness)
from backend.api.scan_service import GlobalScanService
from backend.llm.qwen_agent import TerraShieldQwenAgent

ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(title="TERRA-SHIELD Global Landslide API", version="0.4.0")
analysis_budget = AdmissionBudget(concurrency_limit('TERRA_MAX_CONCURRENT_ANALYSES', 2))
chat_budget = AdmissionBudget(concurrency_limit('TERRA_MAX_CONCURRENT_CHATS', 2))
app.add_middleware(AdmissionMiddleware, analyses=analysis_budget, chats=chat_budget)
allowed_origins = [value.strip() for value in os.environ.get('TERRA_ALLOWED_ORIGINS', '*').split(',') if value.strip()]
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
frontend_dist = frontend_directory(ROOT)

scan_service = GlobalScanService()
agent = TerraShieldQwenAgent(scan_service=scan_service)


@app.get('/api/v1/replays/{name}')
async def retained_replay(name: str):
    """Register a whitelisted retained snapshot as a fresh, stateful replay.

    This route never acquires environmental providers. The Wayanad raw snapshot
    is a local research corpus asset. Production frontend acquisition uses live coordinates and does not boot into a replay.
    """
    from .replay import load_replay
    try:raw=load_replay(name)
    except KeyError as exc:raise HTTPException(status_code=404,detail=str(exc))
    result=await scan_service.scan(_snapshot=raw)
    result['snapshot_note']='Retained provider evidence and later terrain; not current conditions'
    return sanitize_json(result)


@app.get('/api/v1/health')
async def engine_health():
    from backend.ml.physics_informed import VERSION
    ready=bool(scan_service.context_models)
    return {'status':'ok','engine':'LIVE_ENGINE_ACTIVE' if ready else 'MODEL_ARTIFACT_UNAVAILABLE',
        'model_ready':ready,'model_version':VERSION if ready else None,
        'physics_reference_available':True,'environmental_acquisition':'ON_DEMAND',
        'operational_warning_ready':False}


@app.get('/api/v1/ready')
async def deployment_readiness():
    """Check local deployment prerequisites without triggering environmental feeds."""
    return readiness(bool(scan_service.context_models), frontend_dist)


@app.get('/api/v1/connections')
async def connection_settings():
    """Redacted deployment settings, not a provider health check or key editor."""
    from backend.llm.model_client import connection_summary
    return {'scope':'deployment','assistant':connection_summary(),
        'earthdata':{'configured':bool(os.environ.get('EARTHDATA_TOKEN')),
                    'requires_personal_key':False},
        'public_feeds':{'requires_personal_key':False},
        'configuration_scope':'The deployment owner configures shared credentials. Availability is checked when used.',
        'setup_document':'docs/assistant_connections.md'}


class AgentChatRequest(BaseModel):
    message: str
    current_query: str = ""
    agency_mode: str = "LOCAL_EOC"
    sliders: Dict[str, Any] = Field(default_factory=dict)
    scan_id: Optional[str] = None
    scenario_id: Optional[str] = None
    selected_h3_index: Optional[str] = None
    history: list[Dict[str,str]] = Field(default_factory=list,max_length=12)
    autonomous: bool = True
    active_prediction_model: str = Field(default='terra_physics_ml',pattern='^(terra_physics_ml|nasa_lhasa_2|physics_only|weather_terrain_ml|rainfall_only)$')


class CapXmlRequest(BaseModel):
    scan: Optional[Dict[str, Any]] = None
    agency_mode: str = "LOCAL_EOC"


class GroundScanRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    query: str = ''
    lat: Optional[float] = Field(None,ge=-90,le=90)
    lon: Optional[float] = Field(None,ge=-180,le=180)
    latitude: Optional[float] = Field(None,ge=-90,le=90)
    longitude: Optional[float] = Field(None,ge=-180,le=180)
    autonomous: bool = True
    ground_columns: Dict[str, Any] = Field(default_factory=dict)
    rain_delta_mm: float = Field(0.,ge=0)
    storm_duration_hr: float = Field(24.,gt=0)


@app.post('/api/v1/scan')
async def scan_with_ground_inputs(request: GroundScanRequest) -> Dict[str, Any]:
    try:
        latitude=request.latitude if request.latitude is not None else request.lat
        longitude=request.longitude if request.longitude is not None else request.lon
        if request.latitude is not None and request.lat is not None and request.latitude!=request.lat:
            raise ValueError('Conflicting latitude aliases')
        if request.longitude is not None and request.lon is not None and request.longitude!=request.lon:
            raise ValueError('Conflicting longitude aliases')
        if latitude is None and longitude is None and not request.query.strip():raise ValueError('A place query or latitude/longitude is required')
        if request.autonomous and (request.rain_delta_mm or request.storm_duration_hr!=24.):
            raise ValueError('Autonomous scans use provider forcing, not manual rainfall/duration perturbations')
        return sanitize_json(await scan_service.scan(location_query=request.query,lat=latitude,lon=longitude,
            ground_columns=request.ground_columns,rain_delta_mm=request.rain_delta_mm,storm_duration_hr=request.storm_duration_hr))
    except ValueError as exc:
        raise HTTPException(status_code=422,detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502,detail='scan failed: '+type(exc).__name__)


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
@app.post("/api/v1/chat")
@app.post("/api/chat")
async def agent_chat(request: AgentChatRequest) -> Dict[str, Any]:
    try:
        # Autonomous commands return plans for the browser's guarded scan route.
        # Legacy analyst chat can acquire providers internally, so it also needs
        # an analysis slot; ordinary prose uses only the separate chat budget.
        with nullcontext() if request.autonomous else analysis_budget.slot():
            result=await agent.chat(
                request.message,
                current_query=request.current_query,
                agency_mode=request.agency_mode,
                sliders=request.sliders,
                scan_id=request.scan_id,scenario_id=request.scenario_id,selected_h3_index=request.selected_h3_index,
                history=request.history,autonomous=request.autonomous,
                active_prediction_model=request.active_prediction_model,
            )
        if 'thought_steps' not in result:
            from backend.llm.app_actions import public_steps
            result['thought_steps']=public_steps([],bool(request.scan_id))
            result['thought_steps_scope']='Public action and evidence summaries; not private model reasoning'
        return sanitize_json(result)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail="agent scan failed: {}".format(type(exc).__name__))


@app.get('/api/v1/inspect/{scan_id}/{h3_index}')
async def inspect_alias(scan_id:str,h3_index:str,scenario_id:Optional[str]=None):
    try:
        state=scan_service.latest_scan_state(scan_id,scenario_id)
        return sanitize_json(scan_service.inspect_hexagon(h3_index,scan_id,state['response']['scenario_id']))
    except KeyError as exc:raise HTTPException(status_code=404,detail=str(exc))


@app.get('/api/v1/scans/{scan_id}/forecast/{view_key}')
async def automatic_forecast_view(scan_id:str,view_key:str):
    try:return sanitize_json(await scan_service.automatic_forecast_state(scan_id,view_key))
    except KeyError as exc:raise HTTPException(status_code=404,detail=str(exc))
    except ValueError as exc:raise HTTPException(status_code=422,detail=str(exc))


@app.get('/api/v1/scans/{scan_id}/scenarios/{scenario_id}/cells/{h3_index}')
async def inspect_cell(scan_id:str,scenario_id:str,h3_index:str):
    try:return sanitize_json(scan_service.inspect_hexagon(h3_index,scan_id,scenario_id))
    except KeyError as exc:raise HTTPException(status_code=404,detail=str(exc))


@app.get('/api/v1/scans/{scan_id}/scenarios/{scenario_id}/runout/{h3_index}')
async def inspect_runout(scan_id:str,scenario_id:str,h3_index:str):
    try:return sanitize_json(scan_service.trace_runout(h3_index,scan_id,scenario_id))
    except KeyError as exc:raise HTTPException(status_code=404,detail=str(exc))


class ScenarioRequest(BaseModel):
    changes: Dict[str,Any]


@app.post('/api/v1/scans/{scan_id}/scenarios/{scenario_id}')
async def apply_scenario(scan_id:str,scenario_id:str,request:ScenarioRequest):
    try:return sanitize_json(await scan_service.apply_scenario(scan_id,scenario_id,request.changes))
    except KeyError as exc:raise HTTPException(status_code=404,detail=str(exc))
    except ValueError as exc:raise HTTPException(status_code=422,detail=str(exc))


@app.get('/api/v1/scans/{scan_id}/scenarios/{scenario_id}')
async def get_scenario(scan_id:str,scenario_id:str):
    try:return sanitize_json(scan_service.get_scan_state(scan_id,scenario_id)['response'])
    except KeyError as exc:raise HTTPException(status_code=404,detail=str(exc))


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
    # An unvalidated inventory classifier cannot issue CAP likelihood/severity
    # labels. Do not map missing evidence to GREEN/Minor/Possible defaults.
    raise HTTPException(status_code=409,detail='CAP warning export unavailable: no validated operational hazard model or warning thresholds')


# Keep this mount last so API, documentation and schema routes take precedence.
# A missing build fails readiness while leaving the diagnostic API reachable.
if frontend_dist is not None and (frontend_dist / 'index.html').is_file():
    app.mount('/', FrontendFiles(directory=str(frontend_dist), html=True), name='frontend')
