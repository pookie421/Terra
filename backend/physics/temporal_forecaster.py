"""72-hour scenario projections using linear pulse-response superposition.

This is a TRIGRS-inspired homogeneous-column approximation, not the USGS solver
or an independently validated warning system. Fixed initial hydraulic parameters
preserve the linear superposition assumption. Times are UTC hourly grid endpoints.
"""
from dataclasses import replace
from datetime import datetime, timedelta
import math
import numpy as np
from backend.ml.global_xgb import blend_failure_probability
from backend.physics.features import physics_feature_matrix
from backend.physics.multi_causal_engine import MultiCausalEngine
from backend.realtime.forecast_weather import precipitation_paths, finite, SNOW_WATER_MM_PER_CM

HORIZON = 72
DISPLAY_HOURS = (0, 6, 12, 18, 24, 36, 48, 72)


from backend.physics.reference_column import response_function


def superposed_head(infiltration_m_s, conductivity_m_s, diffusivity_m2_s, depth_m, step_seconds=3600., slope_deg=0.):
    """Head at the end of k contiguous pulses; flux has shape (k, cells)."""
    flux = np.asarray(infiltration_m_s, dtype=float)
    k = len(flux)
    if flux.ndim != 2 or not np.isfinite(flux).all() or np.any(flux < 0) or not np.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError('Invalid pulse history')
    conductivity, diffusivity, depth = np.broadcast_arrays(conductivity_m_s,diffusivity_m2_s,depth_m)
    if not all(np.isfinite(a).all() for a in (conductivity,diffusivity,depth)) or np.any(conductivity<=0) or np.any(diffusivity<0) or np.any(depth<=0):
        raise ValueError('Invalid hydraulic parameters')
    start_lags = np.arange(k,0,-1)[:,None]*step_seconds
    end_lags = np.arange(k-1,-1,-1)[:,None]*step_seconds
    transformed = 4.*diffusivity*np.cos(np.radians(slope_deg))**2
    kernels = response_function(transformed*start_lags/depth**2)-response_function(transformed*end_lags/depth**2)
    return depth*np.sum(flux/conductivity*kernels,axis=0)


def forecast_creep_failure_hour(velocities_mm_day, timestamps_hr):
    """Return an extrapolated crossing only for positive, accelerating observations.

    Failure is not observed by this fit. Reject non-accelerating or non-future roots.
    No synthetic velocities are manufactured when monitoring data are absent.
    """
    v,t = np.asarray(velocities_mm_day,float),np.asarray(timestamps_hr,float)
    if v.ndim!=1 or t.shape!=v.shape or len(v)<3 or not np.isfinite(v).all() or not np.isfinite(t).all():
        raise ValueError('At least three finite velocity/time observations are required')
    if np.any(v<=0) or np.any(np.diff(t)<=0):
        raise ValueError('Velocities must be positive and timestamps strictly increasing')
    if not np.all(np.diff(v)>0):
        return None
    centered=t-t[0]
    m,c=np.polyfit(centered,1/v,1)
    crossing=float(t[0]-c/m) if m<0 else None
    return crossing if crossing is not None and np.isfinite(crossing) and crossing>t[-1] else None


def fosm_probability_sigma(state):
    """Bounded FOSM spread heuristic, not a delta-method standard error.

    Clipping sigma_FS does not establish coverage or justify tight confidence bounds.
    """
    density=np.exp(-.5*state.beta_fosm**2)/math.sqrt(2*math.pi)
    return density*np.clip(state.sigma_fs,.03,.18)


def _risk(model, inputs, state, curvature):
    if model is None:
        return state.p_failure_physical
    x=physics_feature_matrix(inputs,state,curvature)
    return blend_failure_probability(model.predict_proba(x),state.p_failure_physical,inputs.slope_deg,state.fs_dynamic)


def validate_injection(amount, arrival, duration):
    if not np.isfinite(amount) or amount<0 or type(arrival) is not int or type(duration) is not int or arrival<0 or duration<1 or arrival+duration>72:
        raise ValueError('Storm amount must be nonnegative and its interval lie within 0..72 hours')


def forecast_72h(inputs, initial_state, curvature, model, hours, ensemble=None, river=None,
                 injected_storm_mm=None, storm_arrival_hour=18, storm_duration_hours=12):
    """Propagate the same initial snapshot along P50 and P90-like forcing paths."""
    if injected_storm_mm is not None:
        validate_injection(injected_storm_mm, storm_arrival_hour, storm_duration_hours)
    if len(hours)!=HORIZON:
        return {'status':'UNAVAILABLE','reason':'72 complete future hourly records are required'}
    try:
        stamps=[datetime.fromisoformat(r['time']) for r in hours]
        contiguous=all(b-a==timedelta(hours=1) for a,b in zip(stamps,stamps[1:]))
    except (KeyError, TypeError, ValueError):
        return {'status':'UNAVAILABLE','reason':'Invalid forecast timestamps'}
    if not contiguous:
        return {'status':'UNAVAILABLE','reason':'Forecast timestamps contain gaps or duplicates'}
    for row in hours:
        if not finite(row.get('temperature_2m')):
            return {'status':'UNAVAILABLE','reason':'Hourly air temperatures are missing'}
    injection=None
    if injected_storm_mm is not None:
        # Explicit synthetic calm-start experiment; live scan summary stays separate.
        p50=np.zeros(72);p50[storm_arrival_hour:storm_arrival_hour+storm_duration_hours]=injected_storm_mm/storm_duration_hours
        p90=p50.copy()
        modes=['INJECTED_STORM_SCENARIO']*72
        injection={'total_mm':injected_storm_mm,'arrival_hour':storm_arrival_hour,
                   'duration_hours':storm_duration_hours,'changes_current_state':True,
                   'initial_condition_quality':'SYNTHETIC_CALM_NOT_LIVE',
                   'reset_fields':['rain_24h_mm','peak_3h_intensity_mmhr','rain_15min_override_mm','pore_pressure_override_m']}
    else:
        try:
            p50,p90,modes=precipitation_paths(hours,ensemble)
        except ValueError as exc:
            return {'status':'UNAVAILABLE','reason':str(exc)}
    engine=MultiCausalEngine()
    observed_probability=_risk(model,inputs,initial_state,curvature)
    observed_state=initial_state
    if injection is not None:
        inputs=replace(inputs,rain_24h_mm=0.,peak_3h_intensity_mmhr=0.,
                       rain_15min_override_mm=0.,pore_pressure_override_m=0.)
        initial_state=engine.compute(inputs)
    soil_fallback_hours=[i for i,row in enumerate(hours,1) if not finite(row.get('soil_temperature_0_to_7cm'))]
    n=initial_state.cell_count
    mask=initial_state.effective_slope_deg>=24
    if not mask.any():mask=np.ones(n,dtype=bool)
    depth=initial_state.regolith_depth_m
    conductivity=initial_state.effective_conductivity_m_s
    diffusivity=initial_state.hydraulic_diffusivity_m2_s
    slope=np.radians(initial_state.effective_slope_deg)
    cap=depth*np.cos(slope)**2
    initial_probability=_risk(model,inputs,initial_state,curvature)
    def summarize(hour,state,probability):
        return {'hour':hour,'critical_fs':float(np.percentile(state.fs_dynamic[mask],5)),
                'risk_pct':float(100*np.percentile(probability[mask],95)),
                'max_m1_pct':float(100*np.max(state.p_failure_m1[mask]))}
    initial=summarize(0,initial_state,initial_probability)
    lapse=np.where(np.asarray(inputs.weather_reference_elevation_m) > -998,
                   .0065*(np.asarray(inputs.elevation_m)-np.asarray(inputs.weather_reference_elevation_m)), 0.)
    river_by_day={r['date']:r['surge_ratio'] for r in (river or {}).get('forecast_7d',[])}
    paths={}
    for name,rainfall in (('p50',p50),('p90',p90)):
        rows=[dict(initial)];fluxes=[];dry_seconds=np.zeros(n)
        rolling=[float(np.mean(inputs.rain_24h_mm))/24]*24
        cycles=np.broadcast_to(np.asarray(inputs.freeze_thaw_cycles_7d,float),(n,)).copy()
        previous_soil=np.broadcast_to(np.asarray(inputs.soil_temperature_c,float),(n,)).copy()
        # Snow water equivalent is a scenario estimate (density ratio 0.1), depleted by melt.
        snow_water=np.broadcast_to(np.asarray(inputs.snow_depth_m,float),(n,)).copy()*100.
        for k,(row,liquid) in enumerate(zip(hours,rainfall),1):
            rolling=(rolling+[float(liquid)])[-24:]
            air=row['temperature_2m']-lapse+np.asarray(inputs.warming_delta_c)
            soil_value=row.get('soil_temperature_0_to_7cm')
            soil=(soil_value if finite(soil_value) else row['temperature_2m'])-lapse+.5*np.asarray(inputs.warming_delta_c)
            cycles+=((previous_soil<0)&(soil>=0))|((previous_soil>0)&(soil<=0))
            previous_soil=soil
            snowfall=row.get('snowfall')
            if finite(snowfall):snow_water+=max(0.,snowfall)*SNOW_WATER_MM_PER_CM
            potential=np.where(air>0,np.maximum(0.,4.5*air+.0126*liquid*24*air)/24,0.)
            melt=np.minimum(snow_water,potential);snow_water-=melt
            q=(liquid+melt)*1e-3/3600
            infiltration=np.minimum(q,conductivity)
            fluxes.append(infiltration)
            # Recession of the retained pre-origin finite storm, using the same
            # linear kernel as future pulses. No additional drainage multiplier.
            elapsed=k*3600.
            duration=np.asarray(inputs.storm_duration_hr)*3600.
            transformed=4*diffusivity*np.cos(slope)**2/depth**2
            past=depth*initial_state.effective_infiltration_m_s/conductivity*(
                response_function(transformed*(duration+elapsed))-response_function(transformed*elapsed))
            base=np.asarray(inputs.antecedent_pressure_head_m)
            transient=past+superposed_head(np.stack(fluxes),conductivity,diffusivity,depth,slope_deg=initial_state.effective_slope_deg)
            future=replace(inputs,rain_24h_mm=sum(rolling),storm_duration_hr=24.,
                peak_3h_intensity_mmhr=max(rolling[-3:]),rain_15min_override_mm=.25*max(rolling[-3:]),air_temperature_c=air,soil_temperature_c=soil,
                snow_depth_m=snow_water/100.,snowmelt_override_mm_day=melt*24,
                freeze_thaw_cycles_7d=cycles,pore_pressure_override_m=-1.,
                transient_pressure_head_m=transient,antecedent_pressure_head_m=base,
                river_surge_ratio=river_by_day.get(row['time'][:10],inputs.river_surge_ratio))
            state=engine.compute(future)
            probability=_risk(model,future,state,curvature)
            result=summarize(k,state,probability)
            result.update(liquid_rain_mm=float(liquid),max_surface_runoff_mm=float(np.max(np.maximum(q-infiltration,0)*3600*1000)))
            rows.append(result)
        crossing=next((r['hour'] for r in rows if r['critical_fs']<1 or r['risk_pct']>=65),None)
        peak=max(rows,key=lambda r:r['risk_pct'])
        paths[name]={'hourly':rows,'peak_risk_pct':peak['risk_pct'],'peak_hour':peak['hour'],
                     'minimum_fs':min(r['critical_fs'] for r in rows),'crossing_hour':crossing}
    primary=paths['p50'];worst=paths['p90']
    return {'status':'PROJECTED_SCENARIO','forecast_origin_utc':(stamps[0]-timedelta(hours=1)).isoformat(),
        'observed_current_risk_t0_pct':summarize(0,observed_state,observed_probability)['risk_pct'],
        'initial_condition_quality':'SYNTHETIC_CALM_NOT_LIVE' if injection is not None else 'LIVE_INPUT_MODELED_STATE',
        'soil_temperature_fallback_hours':soil_fallback_hours,
        'current_risk_t0_pct':initial['risk_pct'],'peak_72h_risk_p50_pct':primary['peak_risk_pct'],
        'peak_72h_risk_p90_pct':worst['peak_risk_pct'],
        'min_72h_fs_dynamic':min(primary['minimum_fs'],worst['minimum_fs']),
        'peak_failure_hour_offset':f"+{primary['peak_hour']}h",
        'predicted_threshold_crossing_hour':primary['crossing_hour'],
        'advance_warning_lead_time_hours':primary['crossing_hour'],
        'p90_threshold_crossing_hour':worst['crossing_hour'],
        'already_critical_at_t0':primary['crossing_hour']==0,
        'hourly_horizon_table':[{'hour':h,'p50':primary['hourly'][h],'p90':worst['hourly'][h]} for h in DISPLAY_HOURS],
        'paths':paths,'forcing_modes':sorted(set(modes)),'injected_storm':injection,
        'creep_failure_hour':None,'creep_status':'No observed velocity series supplied',
        'assumptions':['Linear pulse superposition with fixed initial K,D,Z; not full TRIGRS',
            'Finite pre-origin rainfall pulse plus future pulses; specified steady initial head; surface head cap in live wrapper',
            'Hourly time-grid crossing, not exact real-world failure time; lead refers to P50 regional summary',
            'P90 is pointwise forcing envelope, not 90% probability of failure or a worst-case bound',
            'Explicit warming perturbation persists through the future horizon',
            'Live head is modeled; injected storms explicitly reset initial rain and head to zero',
            'Missing hourly soil temperature uses that hour air temperature, with the same lapse/scenario adjustment',
            'Initial 24h rain distributed uniformly for rolling intensity history',
            'Hourly peak intensity is an R15 proxy; subhourly peaks are unresolved',
            'Initial snow SWE uses 0.1 density ratio; future snowfall uses 10/7 mm/cm',
            'Future soil moisture/snow-depth fields retained in source telemetry; column state is propagated, not overwritten',
            'No validated forecast-skill or warning-lead-time claim']}
