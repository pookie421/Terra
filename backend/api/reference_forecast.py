"""Automatic provider-forced pressure paths and state-bound forecast columns.

Future maps retain the current material column and propagate the downloaded
hourly forcing. Current observations, forecast states and injected research
scenarios are distinct. Ensemble quantiles are forcing paths, not calibrated
landslide probability intervals or coherent ensemble members.
"""
from datetime import timedelta
import copy
import math
import numpy as np
from scipy.special import ndtr,erfc
from scipy.stats import truncnorm
from backend.physics.reference_column import infinite_slope_fs
from backend.realtime.forecast_weather import precipitation_paths
from backend.ml.rainfall_features import utc
from .reference_assessment import assess

PARAMETERS=('cohesion_kpa','friction_angle_deg','unit_weight_kn_m3','depth_m',
    'conductivity_m_s','diffusivity_m2_s','initial_pressure_head_m','slope_deg','precipitation_scale')


def _response(t):
    t=np.asarray(t,float);safe=np.maximum(t,1e-300)
    return np.where(t>0,np.sqrt(safe/np.pi)*np.exp(-1/safe)-erfc(1/np.sqrt(safe)),0.)


def _solve(parameters,rain,observed_duration_hr,hour,observed=None):
    """Vectorized Iverson pulse superposition, including recession of current rain."""
    x=np.asarray(parameters,float)
    c,phi,gamma,z,ks,d,psi0,slope,scale=np.moveaxis(x,-1,0)
    angle=np.radians(slope);cos2=np.cos(angle)**2
    origin=observed_duration_hr*3600.;at=origin+hour*3600.
    # rain[0] is retained current-24h accumulation, then 72 provider hours.
    starts=np.r_[0.,origin+np.arange(hour)*3600.]
    ends=np.r_[origin,origin+np.arange(1,hour+1)*3600.]
    amounts=np.r_[rain[0],rain[1:hour+1]]
    head=np.asarray(psi0,float).copy()
    for index,(start,end,amount) in enumerate(zip(starts,ends,amounts)):
        if index==0 and observed is not None:
            original=np.asarray(observed,float);amount=np.repeat(original,len(x)//len(original))
        flux=np.minimum(amount*.001/(end-start)*scale,ks)
        a=4*d*cos2*max(at-start,0.)/z**2
        b=4*d*cos2*max(at-end,0.)/z**2
        head+=z*flux/ks*(_response(a)-_response(b))
    head=np.minimum(head,z*cos2)
    return infinite_slope_fs(c,phi,gamma,z,slope,head),head


def _joint(x,sd,corr,rain,duration,hour,observed=None):
    covariance=sd[:,:,None]*corr*sd[:,None,:];jac=np.zeros_like(x)
    for k in range(x.shape[1]):
        step=np.maximum(abs(x[:,k])*1e-5,sd[:,k]*1e-4+1e-10)
        lo=x.copy();hi=x.copy();lo[:,k]-=step;hi[:,k]+=step
        lo[:,k]=np.maximum(lo[:,k],0. if k in (0,6,8) else 1e-12)
        if k in (1,7):hi[:,k]=np.minimum(hi[:,k],89.999)
        jac[:,k]=(_solve(hi,rain,duration,hour,observed)[0]-_solve(lo,rain,duration,hour,observed)[0])/(hi[:,k]-lo[:,k])
    return np.maximum(np.einsum('ni,nij,nj->n',jac,covariance,jac),0.),jac,covariance


def _uncertainty(x,sd,corr,rain,duration,hour,observed=None):
    variance,jac,cov=_joint(x,sd,corr,rain,duration,hour,observed)
    nominal,head=_solve(x,rain,duration,hour,observed)
    # Use the same bounded-angle / conditional log-FS approximation as NOW.
    nodes,weights=np.polynomial.legendre.leggauss(32);u=(nodes+1)/2;weights=weights/2
    sigma=np.maximum(sd[:,7],1e-12)
    angles=truncnorm.ppf(u[None,:],(-x[:,7]/sigma)[:,None],((89.999-x[:,7])/sigma)[:,None],loc=x[:,7,None],scale=sigma[:,None])
    q=np.repeat(x[:,None,:],len(u),axis=1);q[:,:,7]=angles
    conditional=np.repeat(sd[:,None,:],len(u),axis=1);conditional[:,:,7]=0.
    expanded_corr=np.repeat(corr[:,None,:,:],len(u),axis=1).reshape(-1,9,9)
    conditional_var=_joint(q.reshape(-1,9),conditional.reshape(-1,9),expanded_corr,rain,duration,hour,observed)[0].reshape(len(x),-1)
    f=_solve(q.reshape(-1,9),rain,duration,hour,observed)[0].reshape(len(x),-1)
    log_mu=np.log(np.maximum(f,1e-12));log_sd=np.maximum(np.sqrt(conditional_var)/np.maximum(f,1e-12),1e-12)
    probability=ndtr(-log_mu/log_sd)@weights
    quantiles=[]
    for level in (.05,.5,.95):
        lo=np.min(log_mu-9*log_sd,axis=1);hi=np.max(log_mu+9*log_sd,axis=1)
        for _ in range(40):
            mid=(lo+hi)/2;cdf=ndtr((mid[:,None]-log_mu)/log_sd)@weights
            lo=np.where(cdf<level,mid,lo);hi=np.where(cdf>=level,mid,hi)
        quantiles.append(np.exp((lo+hi)/2))
    return nominal,head,variance,jac,cov,probability,np.column_stack(quantiles)


def _inputs(cells):
    x=np.asarray([[c['input_values'][k] for k in PARAMETERS[:-1]]+[1.] for c in cells],float)
    sd=np.asarray([[c['standard_deviations'][k] for k in PARAMETERS[:-1]]+[.2] for c in cells],float)
    corr=np.repeat(np.eye(9)[None,:,:],len(cells),axis=0)
    corr[:,0,1]=corr[:,1,0]=-.25;corr[:,4,5]=corr[:,5,4]=.6
    return x,sd,corr


def forecast(scan,models,ground=None,injected=None,arrival=18,duration=12,physical=None,duration_hr=24.):
    hours=getattr(scan.weather,'forecast_hourly_72h',[]) if scan.weather else []
    if injected is None and len(hours)!=72:
        return {'status':'UNAVAILABLE','automatic':True,'reason':'72 complete future forcing records absent; current physical columns remain computed','views':[]}
    if injected is None:
        try:
            times=[utc(r['time']) for r in hours]
            if any(b-a!=timedelta(hours=1) for a,b in zip(times,times[1:])):raise ValueError('Hourly gaps or duplicates')
            p50,p90,modes=precipitation_paths(hours,scan.ensemble)
        except (KeyError,TypeError,ValueError) as exc:
            return {'status':'UNAVAILABLE','automatic':True,'reason':str(exc),'views':[]}
    else:
        p50=np.zeros(72);p50[arrival:arrival+duration]=injected/duration;p90=p50.copy();modes=['INJECTED_STORM_SCENARIO']
        base=utc(scan.generated_at_utc).replace(minute=0,second=0,microsecond=0)
        times=[base+timedelta(hours=i+1) for i in range(72)]
    physical=physical or assess(scan,ground)
    cells=[c for c in physical['cells'] if c['fs_mid'] is not None]
    if not cells:return {'status':'UNAVAILABLE','reason':'No positive driving-shear columns; current zero-shear states remain available','views':[]}
    x,_,_=_inputs(cells);paths={}
    observed_by_cell=np.asarray([c['input_values']['rain_24h_mm'] for c in cells],float)
    observed=float(observed_by_cell[0])
    for name,amounts in [('p50',p50),('p90',p90)]:
        rain=np.r_[observed,amounts];minimum=[]
        # All cells are evaluated in a vectorized call at each provider hour.
        for hour in range(73):
            fs,head=_solve(x,rain,duration_hr,hour,observed_by_cell)
            minimum.append(float(np.min(fs)))
        cross=next((i for i,v in enumerate(minimum) if v<1),None)
        paths[name]={'hourly':[{'hour':i,'valid_time_utc':(times[0]+timedelta(hours=i-1)).isoformat(),
            'minimum_fs':value,'rain_mm':0. if i==0 else float(amounts[i-1])} for i,value in enumerate(minimum)],
            'crossing_hour':cross,'minimum_fs':min(minimum),'rain_72h_mm':float(np.sum(amounts)),
            'ensemble_quantile_available':not any('DETERMINISTIC_ONLY' in mode for mode in modes)}
    peak=int(np.argmin([r['minimum_fs'] for r in paths['p50']['hourly']]))
    views=[{'key':key,'hour':hour,'valid_time_utc':(times[0]+timedelta(hours=hour-1)).isoformat(),
            'minimum_fs':paths['p50']['hourly'][hour]['minimum_fs'],'forcing_path':'p50'}
           for key,hour in [('now',0),('t24',24),('t48',48),('t72',72),('peak',peak)]] if injected is None else []
    return {'status':'PROJECTED_REFERENCE_SCENARIO','automatic':injected is None,'paths':paths,'forcing_modes':sorted(set(modes)),
        'views':views,'peak_hour':peak,'peak_selection':'minimum_sector_nominal_fs_p50',
        'min_72h_fs_dynamic':min(p['minimum_fs'] for p in paths.values()),
        'predicted_threshold_crossing_hour':paths['p50']['crossing_hour'],
        'scope':'Automatic provider-forced conditional column trajectories; no exact failure-time or globally calibrated probability claim. P90 is not a coherent member.',
        'initial_condition_quality':'CURRENT_REFERENCE_COLUMN_RETAINED; prior rain recession is propagated',
        'process_scope':'Liquid rainfall infiltration and prior-pressure recession with fixed current material priors. Future earthquakes, rock-joint failure, post-fire basin runoff and erosion are not inferred from this weather projection.',
        'projection_context':{'rain_p50_mm':p50.tolist(),'rain_p90_mm':p90.tolist(),
            'observed_rain_mm':observed,'observed_rain_by_cell':dict(zip([c['h3_index'] for c in cells],map(float,observed_by_cell))),'observed_duration_hr':duration_hr,'times':[t.isoformat() for t in times]}}


def projected_columns(scan,base_physical,projection,hour):
    """Full joint uncertainty at an automatic forecast horizon, fixed material priors."""
    if hour==0:return copy.deepcopy(base_physical['cells'])
    source=[c for c in base_physical['cells'] if c['fs_mid'] is not None]
    x,sd,corr=_inputs(source)
    rain=np.r_[projection['observed_rain_mm'],projection['rain_p50_mm']]
    fs,head,var,jac,cov,p,q=_uncertainty(x,sd,corr,rain,projection['observed_duration_hr'],hour,[projection['observed_rain_by_cell'][c['h3_index']] for c in source])
    records=[]
    for i,base in enumerate(source):
        r=copy.deepcopy(base);v=r['input_values'];angle=math.radians(v['slope_deg']);normal=v['unit_weight_kn_m3']*v['depth_m']*math.cos(angle)**2
        driving=v['unit_weight_kn_m3']*v['depth_m']*math.sin(angle)*math.cos(angle)
        low=min(float(fs[i]),float(q[i,0]));high=max(float(fs[i]),float(q[i,2]))
        critical=(normal-(driving-v['cohesion_kpa'])/max(math.tan(math.radians(v['friction_angle_deg'])),1e-12))/9.81
        deficit=max(0.,critical-head[i]);response=float(_response(4*v['diffusivity_m2_s']*math.cos(angle)**2*86400/v['depth_m']**2))
        attainable=critical<=v['depth_m']*math.cos(angle)**2 and deficit<=v['depth_m']*response and v['cohesion_kpa']<=driving
        rain_deficit=deficit*v['conductivity_m_s']*86400/max(v['depth_m']*response,1e-300)*1000 if attainable else None
        f=r['mechanistic_features'];f.update(fs_mid=float(fs[i]),fs_low=low,fs_high=high,
            pore_pressure_ratio=9.81*float(head[i])/max(normal,1e-12),critical_pressure_ratio_deficit=9.81*deficit/max(normal,1e-12),
            critical_rainfall_deficit_ratio=rain_deficit/1000 if rain_deficit is not None else None)
        r.update(fs_mid=float(fs[i]),fs_dynamic=float(fs[i]),fs_low=low,fs_high=high,fs_interval=[low,float(fs[i]),high],
            pressure_head_m=float(head[i]),fosm_variance=float(var[i]),sigma_fs=float(np.sqrt(var[i])),
            p_failure_physical=float(p[i]),fs_distribution_quantiles_05_50_95=q[i].tolist(),
            fosm_log_variance=float(var[i]/max(fs[i]**2,1e-24)),
            critical_pressure_head_m=critical,critical_rainfall_deficit_mm=rain_deficit,
            rainfall_threshold_attainable_at_duration=bool(attainable),joint_parameter_names=list(PARAMETERS),
            fs_jacobian=dict(zip(PARAMETERS,map(float,jac[i]))),joint_covariance=cov[i].tolist(),
            uncertainty_variance_allocation=dict(zip(PARAMETERS,map(float,jac[i]*(cov[i]@jac[i])))),
            forecast_hour=hour,uncertainty_status='PROJECTED_JOINT_REFERENCE_FOSM_WITH_20PCT_FORCING_SCALE')
        r['standard_deviations']['precipitation_scale']=.2;r['input_values']['precipitation_scale']=1.
        prior_fs=float(infinite_slope_fs(v['cohesion_kpa'],v['friction_angle_deg'],v['unit_weight_kn_m3'],v['depth_m'],v['slope_deg'],r['initial_pressure_head_m']))
        r['causal_attributions']['rain_pulse_fs_reduction']=max(0.,prior_fs-fs[i])
        r['newmark_displacement_cm']=None
        r['newmark_status']='Future seismic shaking is not forecast; current catalog events remain NOW context'
        records.append(r)
    records.extend(copy.deepcopy(c) for c in base_physical['cells'] if c['fs_mid'] is None)
    return records


def projected_window(scan,projection,hour):
    base=dict(scan.weather.rainfall_source_window)
    history=base.get('liquid_rain_hourly_mm')
    if not isinstance(history,list) or len(history)!=168:
        return {'status':'FORCING_PRIOR','reason':'Complete observed hourly window absent; projected learned score is unavailable'}
    rain=(history+projection['rain_p50_mm'][:hour])[-168:]
    moisture=None
    if hour==0:return base
    if hour>=24:
        values=[r.get('soil_moisture_0_to_7cm') for r in scan.weather.forecast_hourly_72h[hour-24:hour]]
        if len(values)==24 and all(v is not None and math.isfinite(v) for v in values):moisture=float(np.mean(values))
    return {**base,'status':'AVAILABLE','rain_24h_mm':float(sum(rain[-24:])),
        'rain_prior_6d_mm':float(sum(rain[:-24])),'liquid_rain_hourly_mm':rain,
        'soil_moisture_0_7cm_mean_m3m3':moisture,'cutoff_utc':projection['times'][hour-1],
        'product':'AUTOMATIC_FORECAST_ROLLING_WINDOW','moisture_support':'Forecast 24h mean' if moisture is not None else 'Missing complete 24h forecast moisture; fitted model missingness handling applies'}
