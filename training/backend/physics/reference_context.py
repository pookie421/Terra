"""Conditional regional columns, water balance and joint delta-method uncertainty.

Hydraulics: Saxton & Rawls (2006), doi:10.2136/sssaj2005.0117;
pressure response: Iverson (2000), doi:10.1029/2000WR900090.
Strength/depth are broad project reference priors, NOT purported published fits
or borehole observations. Their coefficients/ranges and joint covariance are
versioned and exposed. This model is for shallow regolith, not rock-joint failure.
"""
import math
import numpy as np
from scipy.special import ndtr
from scipy.stats import truncnorm
from .reference_column import response_function,infinite_slope_fs

VERSION='pedotransfer_reference_column_v1'
PARAMETERS=('cohesion_kpa','friction_angle_deg','unit_weight_kn_m3','depth_m',
            'conductivity_m_s','diffusivity_m2_s','initial_pressure_head_m','slope_deg','rain_24h_mm')
CONTEXT_FEATURES=('fs_mid','fs_low','fs_high','shear_friction_ratio','cohesion_stability_number',
                  'pore_pressure_ratio','critical_pressure_ratio_deficit','critical_rainfall_deficit_ratio')
CONSTRAINTS=(-1,-1,-1,1,-1,1,-1,-1)
SOURCES={'hydraulics':'https://doi.org/10.2136/sssaj2005.0117',
         'infiltration':'https://doi.org/10.1029/2000WR900090',
         'mechanics':'https://pubs.usgs.gov/of/2008/1159/',
         'strength_depth':'Project texture/slope reference prior; coefficients are assumptions, not a universal published transfer fit'}


def _solve(x,duration_s):
    values=np.moveaxis(np.asarray(x,float),-1,0)
    c,phi,gamma,z,ks,d,psi0,slope,rain=values[:9]
    kh=values[9] if len(values)>9 else 0.
    angle=np.radians(slope);cap=z*np.cos(angle)**2
    flux=np.minimum(rain*.001/duration_s,ks)
    response=response_function(4*d*np.cos(angle)**2*duration_s/z**2)
    psi=np.minimum(cap,psi0+z*flux/ks*response)
    fs=infinite_slope_fs(c,phi,gamma,z,slope,psi,kh)
    return fs,psi


def joint_fosm(parameters,standard_deviations,duration_s=86400.,correlation=None):
    """J Sigma J^T over strength, geometry, initial state, forcing and hydraulics.

    Central differences use domain-aware steps. At saturation/contact kinks this
    is a local linear approximation, not a coverage guarantee.
    """
    x=np.atleast_2d(np.asarray(parameters,float));sd=np.broadcast_to(standard_deviations,x.shape)
    corr=np.eye(x.shape[1]) if correlation is None else np.asarray(correlation,float)
    if not np.isfinite(x).all() or not np.isfinite(sd).all() or np.any(sd<0):raise ValueError('Finite parameters and nonnegative spreads required')
    if not np.allclose(corr,corr.T) or np.linalg.eigvalsh(corr).min() < -1e-10:
        raise ValueError('Joint correlation must be symmetric positive semidefinite')
    covariance=sd[:,:,None]*corr[None,:,:]*sd[:,None,:]
    jac=np.zeros_like(x)
    for k in range(x.shape[1]):
        step=np.maximum(np.abs(x[:,k])*1e-5,sd[:,k]*1e-4+1e-10)
        lo=x.copy();hi=x.copy();lo[:,k]-=step;hi[:,k]+=step
        floor=0. if k in (0,6,8,9) else 1e-12
        lo[:,k]=np.maximum(lo[:,k],floor)
        if k in (1,7):hi[:,k]=np.minimum(hi[:,k],89.999)
        jac[:,k]=(_solve(hi,duration_s)[0]-_solve(lo,duration_s)[0])/(hi[:,k]-lo[:,k])
    var=np.einsum('ni,nij,nj->n',jac,covariance,jac)
    return np.maximum(var,0.),jac,covariance


def angular_reliability(parameters,spreads,duration_s,correlation,order=32):
    """Integrate bounded slope uncertainty before approximating material response.

    A local FS Taylor expansion fails near zero driving shear: its derivative
    diverges although every plausible small positive slope remains stable.
    Probability-space Gauss-Legendre quadrature integrates a 0..89.999-degree
    truncated normal slope. Conditional log-FS FOSM handles the remaining joint
    parameters. Full unconditional J Sigma J^T is still reported independently.
    Slope is independent in the declared reference covariance.
    """
    x=np.asarray(parameters,float);sd=np.asarray(spreads,float);n=len(x)
    nodes,weights=np.polynomial.legendre.leggauss(order);u=(nodes+1)/2;weights=weights/2
    sigma=np.maximum(sd[:,7],1e-12)
    angles=truncnorm.ppf(u[None,:],(-x[:,7]/sigma)[:,None],((89.999-x[:,7])/sigma)[:,None],loc=x[:,7,None],scale=sigma[:,None])
    q=np.repeat(x[:,None,:],order,axis=1);q[:,:,7]=angles
    conditional=np.repeat(sd[:,None,:],order,axis=1);conditional[:,:,7]=0.
    variance=joint_fosm(q.reshape(-1,x.shape[1]),conditional.reshape(-1,x.shape[1]),duration_s,correlation)[0].reshape(n,order)
    fs=_solve(q.reshape(-1,x.shape[1]),duration_s)[0].reshape(n,order)
    log_mu=np.log(np.maximum(fs,1e-12));log_sigma=np.sqrt(variance)/np.maximum(fs,1e-12)
    log_sigma=np.maximum(log_sigma,1e-12)
    probability=ndtr(-log_mu/log_sigma)@weights
    quantiles=[]
    for level in (.05,.5,.95):
        lo=np.min(log_mu-9*log_sigma,axis=1);hi=np.max(log_mu+9*log_sigma,axis=1)
        for _ in range(40):
            mid=(lo+hi)/2;cdf=ndtr((mid[:,None]-log_mu)/log_sigma)@weights
            lo=np.where(cdf<level,mid,lo);hi=np.where(cdf>=level,mid,hi)
        quantiles.append(np.exp((lo+hi)/2))
    return probability,np.column_stack(quantiles)


def columns(cells,window,soil=None,rain_delta_mm=0.,duration_hr=24.,overrides=None):
    cells=list(cells);n=len(cells)
    if not n:return []
    from backend.realtime.soilgrids import estimate_pedotransfer,derive_soil_properties
    from .multi_causal_engine import MultiCausalEngine,PhysicsInputs
    records=[];properties=[]
    for cell in cells:
        raw=dict(soil or {})
        if str(raw.get('source','')).startswith('static_'):raw={}
        if raw.get('property_sources'):
            local_prior=estimate_pedotransfer(cell.slope_deg,cell.profile_curvature,cell.twi)
            for name,origin in raw['property_sources'].items():
                if 'prior' in origin:raw[name]=local_prior[name]
        if any(raw.get(k) is None for k in ('sand_g_kg','clay_g_kg','soc_g_kg','bdod_kg_m3')):
            raw=estimate_pedotransfer(cell.slope_deg,cell.profile_curvature,cell.twi)
        try:properties.append(derive_soil_properties(raw))
        except (ValueError,TypeError):properties.append(derive_soil_properties(estimate_pedotransfer(cell.slope_deg,cell.profile_curvature,cell.twi)))
    sand=np.array([p['sand_fraction'] for p in properties]);clay=np.array([p['clay_fraction'] for p in properties])
    rho=np.array([p['bulk_density_g_cm3'] for p in properties]);porosity=1-rho/2.65
    slope=np.array([c.slope_deg for c in cells]);angle=np.radians(slope)
    # Reuse the tested Saxton-Rawls hydraulic equations; no wrapper FS is used.
    state=MultiCausalEngine().compute(PhysicsInputs(sand,clay,np.array([p['organic_matter_fraction'] for p in properties]),rho,
        np.maximum(slope,1e-6),np.maximum([c.twi for c in cells],1e-6),.5,0.,duration_hr,0.,10.,8.,0.,0.,.5,0.,9999.,9999.))
    z=state.regolith_depth_m;ks=state.hydraulic_conductivity_m_s;d=state.hydraulic_diffusivity_m2_s
    field=np.minimum(state.theta_33,porosity*.95)
    storage=field*z
    history=window.get('liquid_rain_hourly_mm')
    prior=history[:-24] if history is not None and len(history)==168 else [float(window.get('rain_prior_6d_mm') or 0.)/144.]*144
    for rain in prior:
        recharge=np.minimum(float(rain)*.001,ks*3600.)
        # Gravity drainage from water above field capacity; basal K ratio is an
        # explicit reference assumption. Water accounting includes ET and overflow.
        storage+=recharge
        excess=np.maximum(storage-field*z,0.)
        storage-=np.minimum(excess,ks*.01*3600.)
        storage=np.maximum(0.,np.minimum(storage,porosity*z)-.002/24.)
    theta=window.get('soil_moisture_0_7cm_mean_m3m3')
    compatible=np.full(n,False) if theta is None else (float(theta)>=0)&(float(theta)<=porosity)
    # Surface moisture is an optional noisy storage constraint; never a measured
    # water table. Incompatible provider/prior pairs retain water-balance state.
    if theta is not None:
        # The provider's surface layer constrains only 7 cm of storage. It cannot
        # replace moisture throughout a metre-scale regolith profile.
        correction=(float(theta)-field)*np.minimum(z,.07)
        storage=np.where(compatible,np.clip(storage+correction,0.,porosity*z),storage)
    drainable=np.maximum(porosity-field,.03)
    perched=np.maximum(storage-field*z,0.)/drainable
    initial=np.minimum(perched,z)*np.cos(angle)**2
    saturation=np.clip(storage/(porosity*z),0.,1.)
    gamma=(rho+saturation*porosity)*9.81
    rain=float(window.get('rain_24h_mm') or 0.)+rain_delta_mm
    x=np.column_stack([state.soil_cohesion_kpa,state.friction_angle_deg,gamma,z,ks,d,initial,slope,np.full(n,rain)])
    provenance=[{} for _ in cells]
    aliases={'pressure_head_m':6,**{name:i for i,name in enumerate(PARAMETERS)}}
    for j,cell in enumerate(cells):
        for name,entry in (overrides or {}).get(cell.cell_id,{}).items():
            if name in aliases:x[j,aliases[name]]=float(entry['value']);provenance[j][name]=entry
        if 'pressure_head_m' in (overrides or {}).get(cell.cell_id,{}):x[j,8]=rain_delta_mm
    # Nonzero pressure uncertainty even for a zero nominal head. Lognormal-like
    # positive parameter spreads are represented by local first/second moments.
    sd=np.column_stack([.4*x[:,0]+.5,np.full(n,4.),.08*x[:,2],.35*x[:,3],.5*x[:,4],.6*x[:,5],
                        .15*x[:,3]*np.cos(angle)**2+.02,np.full(n,1.),.2*x[:,8]+.5])
    if window.get('status') not in (None,'AVAILABLE'):sd[:,8]=100.
    corr=np.eye(9);corr[0,1]=corr[1,0]=-.25;corr[4,5]=corr[5,4]=.6
    zero=slope==0
    safe=x.copy();safe[:,7]=np.maximum(safe[:,7],1e-6)
    fs,head=_solve(safe,duration_hr*3600.)
    var,jac,cov=joint_fosm(safe,sd,duration_hr*3600.,corr)
    sigma=np.sqrt(var)
    # FS is nonnegative and ratio-distributed. Apply the same delta method on
    # log(FS) for reliability/envelopes, avoiding a normal FS distribution's
    # impossible negative resistance tail on strong, nearly flat columns.
    # Preserve J Sigma J^T in FS units as well as the linear-normal diagnostic.
    sigma_log=sigma/np.maximum(fs,1e-12)
    with np.errstate(over='ignore',invalid='ignore'):
        low=fs*np.exp(-1.645*sigma_log);high=fs*np.exp(1.645*sigma_log)
    boundary=fs<=1e-9
    low=np.where(boundary,0.,low);high=np.where(boundary,fs+1.645*sigma,high)
    p=ndtr(-np.log(np.maximum(fs,1e-12))/np.maximum(sigma_log,1e-12));p[zero]=0.
    linear_probability=ndtr((1-fs)/np.maximum(sigma,1e-12));linear_probability[zero]=0.
    p=np.where(boundary,linear_probability,p)
    # Apply the same boundary-aware reliability model everywhere, rather than a
    # geographic or low-slope score gate. Zero-resistance cases retain the
    # explicitly reported linear boundary approximation.
    positive=~boundary
    quantiles=np.column_stack([low,fs,high])
    if positive.any():
        angular_p,angular_q=angular_reliability(safe[positive],sd[positive],duration_hr*3600.,corr)
        p[positive]=angular_p;quantiles[positive]=angular_q
        low[positive]=np.minimum(angular_q[:,0],fs[positive])
        high[positive]=np.maximum(angular_q[:,2],fs[positive])
    p[zero]=0.
    c,phi,gamma,z,ks,d,psi0,actual_slope,rains=x.T
    sin=np.sin(np.radians(actual_slope));cos=np.cos(np.radians(actual_slope));tanphi=np.tan(np.radians(phi))
    driving=gamma*z*sin*cos;normal=gamma*z*cos*cos
    critical_head=(normal-(driving-c)/np.maximum(tanphi,1e-12))/9.81
    deficit=np.maximum(critical_head-head,0.)
    response=response_function(4*d*cos*cos*duration_hr*3600./z**2)
    max_gain=z*response
    rain_needed=np.divide(deficit*ks*duration_hr*3600.,max_gain,out=np.full(n,np.inf),where=max_gain>0)*1000
    # Deficit is additional rain at fixed duration; infinity denotes capacity-limited unattainability.
    attainable=(deficit<=np.maximum(max_gain-(head-psi0),0.))&(critical_head<=z*cos*cos)&(c<=driving)
    rain_needed=np.where(attainable,rain_needed,np.inf)
    for j,cell in enumerate(cells):
        features=[fs[j],low[j],high[j],math.tan(math.radians(actual_slope[j]))/max(tanphi[j],1e-12),
                  c[j]/max(driving[j],1e-12),9.81*head[j]/max(normal[j],1e-12),
                  9.81*deficit[j]/max(normal[j],1e-12),rain_needed[j]/1000.]
        features=[float(v) if np.isfinite(v) else None for v in features]
        if zero[j]:features[:3]=[None,None,None]
        dry_fs=float(infinite_slope_fs(c[j],phi[j],gamma[j],z[j],max(actual_slope[j],1e-6),0.))
        initial_fs=float(infinite_slope_fs(c[j],phi[j],gamma[j],z[j],max(actual_slope[j],1e-6),psi0[j]))
        records.append({'h3_index':cell.cell_id,'status':'NO_DRIVING_SHEAR' if zero[j] else 'REFERENCE_COLUMN',
            'provenance_label':'pedotransfer_reference_column','model_version':VERSION,
            'fs_dynamic':features[0],'fs_mid':features[0],'fs_low':features[1],'fs_high':features[2],
            'fs_interval':[features[1],features[0],features[2]],'sigma_fs':float(sigma[j]) if not zero[j] else 0.,
            'fs_distribution_quantiles_05_50_95':[float(v) if np.isfinite(v) else None for v in quantiles[j]],
            'fosm_variance':float(var[j]) if not zero[j] else 0.,'p_failure_physical':float(p[j]),
            'fosm_log_variance':float(sigma_log[j]**2) if not boundary[j] else None,'p_failure_linear_normal_diagnostic':float(linear_probability[j]),
            'pressure_head_m':float(head[j]),'initial_pressure_head_m':float(psi0[j]),
            'critical_pressure_head_m':float(critical_head[j]),'critical_rainfall_deficit_mm':float(rain_needed[j]) if np.isfinite(rain_needed[j]) else None,
            'rainfall_threshold_attainable_at_duration':bool(attainable[j]),
            'mechanistic_features':dict(zip(CONTEXT_FEATURES,features)),
            'input_values':dict(zip(PARAMETERS,map(float,x[j]))),'standard_deviations':dict(zip(PARAMETERS,map(float,sd[j]))),
            'fs_jacobian':dict(zip(PARAMETERS,map(float,jac[j]))),'joint_covariance':cov[j].tolist(),
            'uncertainty_variance_allocation':dict(zip(PARAMETERS,map(float,jac[j]*(cov[j]@jac[j])))),
            'causal_attributions':{'scope':'Counterfactual mechanical margin changes at fixed material/geometry; not identified real-world causal shares',
                'antecedent_water_fs_reduction':max(0.,dry_fs-initial_fs),'rain_pulse_fs_reduction':max(0.,initial_fs-fs[j])},
            'ground_provenance':provenance[j],'soil_source':properties[j].get('source'),
            'moisture_assimilation':'depth_limited_0_to_7cm_storage_constraint' if compatible[j] else 'incompatible_or_missing_surface_moisture; antecedent_water_balance',
            'sources':SOURCES,'initialization':{'basal_conductivity_ratio':.01,'et_mm_day':2.,'history':'168h pulses' if history else 'uniform antecedent-total reconstruction'},
            'interval_scope':'Bounded-slope quadrature + conditional log-FS FOSM; envelope includes nominal FS and 5th/95th quantiles; assumed covariance, not calibrated event probability',
            'fosm_variance_scope':'Unconditional local J Sigma J^T diagnostic; near-zero slope requires the separately reported bounded-slope reliability integration',
            'scope':'Shallow regolith reference column; rock joints and liquefaction require separate models'})
    for record in records:
        pga=window.get('pga_reference_g')
        record['pga_reference_g']=pga
        record['newmark_displacement_cm']=None
        record['newmark_status']='No positive source/scenario PGA; no measured zero shaking inferred'
        fs=record['fs_mid']
        if pga is not None and pga>0 and fs is not None and fs>1:
            from .reference_column import jibson_displacement_cm
            critical=(fs-1)*math.sin(math.radians(record['input_values']['slope_deg']))
            record['newmark_displacement_cm']=float(jibson_displacement_cm(critical,pga))
            record['newmark_status']='Stable-block Jibson regression; PGA is a catalog/scenario proxy, not measured motion'
        elif pga is not None and pga>0:
            record['newmark_status']='Outside stable-block regression domain (FS <= 1 or no driving shear)'
    return records
