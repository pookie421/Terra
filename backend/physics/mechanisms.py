"""Published equation forms with explicitly identified scenario input proxies.

References and domain restrictions: docs/physics_sources.md. The ice factors and
post-fire yield index are project scenario parameterizations, not paper equations.
"""
import numpy as np

def sigmoid(x):
    return 1.0/(1.0+np.exp(-np.clip(x,-700,700)))

def postfire(burn,slope_deg,clay,sand,rain,peak_mmhr,depth,r15_override=-1.):
    r15=np.maximum(.25*np.maximum(peak_mmhr,0),.12*np.maximum(rain,0))
    r15=np.where(np.asarray(r15_override)>=0,r15_override,r15)
    prop=burn*sigmoid((slope_deg-21.)/2.5)
    dnbr=.85*burn
    kf=np.clip(.18+.35*(1-clay)-.15*sand,.15,.45)
    link=-3.63+(.41*prop+.67*dnbr+.70*kf)*r15
    active=(burn>.05)&(slope_deg>=12.)
    probability=np.where(active,sigmoid(link),0.)
    # Gartner emergency model coefficients; area and relief here are cell proxies.
    area=np.maximum(.105*burn,.01)
    relief=np.maximum(depth*50*np.sin(np.radians(slope_deg)),10.)
    ln_volume=4.22+.39*np.sqrt(4*r15)+.36*np.log(area)+.13*np.sqrt(relief)
    volume=np.where(active,np.exp(np.clip(ln_volume,-700,700)),0.)
    yield_index=np.full_like(probability,5.)  # deprecated diagnostic, never mechanical FS
    return probability,volume,yield_index,r15

def suction_cohesion(saturation,tan_phi,active_head,alpha_kpa_inv=.1,n=2.):
    if alpha_kpa_inv<=0 or n<=1:raise ValueError('van Genuchten alpha >0 and n >1 required')
    se=np.clip((saturation-.08)/.92,.02,.999)
    stress=-se/alpha_kpa_inv*np.expm1(np.clip(np.log(se)*n/(1-n),0.,700.))**(1/n)
    # Smooth switch reaches zero at Se=.85, avoiding a discontinuous cutoff.
    taper=np.clip((.85-se)/.10,0,1);taper=taper*taper*(3-2*taper)
    return np.where(active_head,0.,np.clip(-stress*tan_phi,0,8)*taper)

def ice_factors(elevation,soil_temp,snow,cycles,warming,eligible=None):
    if eligible is None:
        from .alpine_weather import cryospheric_eligibility
        eligible=cryospheric_eligibility(snow,soil_temp,elevation,0.)
    # Requested joint-temperature proxy, NOT an observed joint temperature.
    joint=np.clip(np.asarray(soil_temp)-1.5-.30*np.asarray(cycles)+.50*np.asarray(warming),-8.,-.5)
    strength=mamot_joint_strength(joint,200.)
    friction_raw=.42-.21*joint
    cohesion_raw=strength-200.*friction_raw
    cohesion=np.where(eligible,np.clip(cohesion_raw/(53.3-73.5*(-8.)),.14,1.),1.)
    friction=np.where(eligible,np.clip(friction_raw/(.42-.21*(-8.)),.40,1.),1.)
    return eligible,cohesion,friction

def mamot_joint_strength(temperature_c,normal_stress_kpa):
    """Mamot 2018 Eqs.4–6, for verified ice-filled joints in the tested domain."""
    t,stress=np.broadcast_arrays(np.asarray(temperature_c,float),np.asarray(normal_stress_kpa,float))
    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(stress)) or np.any((t < -8)|(t > -.5)|(stress <100)|(stress>400)):
        raise ValueError('Mamot calibration requires -8..-0.5 C and 100..400 kPa, with known ice-filled joints')
    cohesion=53.3-73.5*t
    friction=.42-.21*t
    return cohesion+stress*friction
