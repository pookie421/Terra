# Exact core source for external audit

Line-number prefixes are added; text after each prefix is verbatim source. Inclusion is not a claim that every equation is literature-verified. See docs/physics_sources.md for proxy and domain disclosures.

## backend/physics/multi_causal_engine.py

SHA-256: `3da799e34a72e804354c82890e528f0ac91bf92c514dc6249a7d8ae31c250f9b`

```python
0001 """Vectorized five-regime slope-stability equations for H3 cells."""
0002 
0003 from __future__ import annotations
0004 
0005 import math
0006 from dataclasses import asdict, dataclass, field
0007 from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
0008 
0009 import numpy as np
0010 from .mechanisms import postfire, suction_cohesion, ice_factors
0011 
0012 ArrayInput = Union[float, int, Sequence[float], np.ndarray]
0013 GAMMA_W_KN_M3 = 9.81
0014 PARTICLE_DENSITY_G_CM3 = 2.65
0015 
0016 
0017 @dataclass
0018 class PhysicsInputs:
0019     """Measured or explicitly scenario-defined inputs; no silent imputation is applied."""
0020 
0021     sand_fraction: ArrayInput
0022     clay_fraction: ArrayInput
0023     organic_matter_fraction: ArrayInput
0024     bulk_density_g_cm3: ArrayInput
0025     slope_deg: ArrayInput
0026     twi: ArrayInput
0027     soil_saturation_fraction: ArrayInput
0028     rain_24h_mm: ArrayInput
0029     storm_duration_hr: ArrayInput
0030     snow_depth_m: ArrayInput
0031     air_temperature_c: ArrayInput
0032     soil_temperature_c: ArrayInput
0033     freeze_thaw_cycles_7d: ArrayInput
0034     burn_severity: ArrayInput
0035     ndvi_norm: ArrayInput
0036     pga_g: ArrayInput
0037     road_cut_distance_m: ArrayInput
0038     river_distance_m: ArrayInput
0039     profile_curvature: ArrayInput = 0.0
0040     elevation_m: ArrayInput = 0.0
0041     warming_delta_c: ArrayInput = 0.0
0042     peak_3h_intensity_mmhr: ArrayInput = 0.0
0043     basal_conductivity_ratio: ArrayInput = 0.01
0044     vg_alpha_kpa_inv: ArrayInput = 0.10
0045     vg_n: ArrayInput = 2.0
0046     latitude_deg: ArrayInput = 0.0
0047     baseline_air_temperature_c: ArrayInput = -999.0
0048     observed_snow_depth_m: ArrayInput = -1.0
0049     block_width_m: ArrayInput = 35.0
0050     hollow_confinement_ratio: ArrayInput = 1.0
0051 
0052 
0053 @dataclass
0054 class PhysicsState:
0055     theta_s: np.ndarray
0056     theta_1500: np.ndarray
0057     theta_33: np.ndarray
0058     hydraulic_conductivity_m_s: np.ndarray
0059     friction_angle_deg: np.ndarray
0060     soil_cohesion_kpa: np.ndarray
0061     regolith_depth_m: np.ndarray
0062     snowmelt_mm_day: np.ndarray
0063     thaw_factor: np.ndarray
0064     root_cohesion_kpa: np.ndarray
0065     fire_runoff_multiplier: np.ndarray
0066     effective_conductivity_m_s: np.ndarray
0067     effective_infiltration_m_s: np.ndarray
0068     local_saturation: np.ndarray
0069     hydraulic_diffusivity_m2_s: np.ndarray
0070     dimensionless_time: np.ndarray
0071     iverson_response: np.ndarray
0072     pore_pressure_head_m: np.ndarray
0073     toe_undercut_multiplier: np.ndarray
0074     wet_unit_weight_kn_m3: np.ndarray
0075     effective_cohesion_kpa: np.ndarray
0076     kh: np.ndarray
0077     fs_static: np.ndarray
0078     fs_dynamic: np.ndarray
0079     sigma_fs: np.ndarray
0080     beta_fosm: np.ndarray
0081     p_failure_physical: np.ndarray
0082     critical_acceleration_g: np.ndarray
0083     newmark_displacement_cm: np.ndarray
0084     unburned_regolith_depth_m: np.ndarray
0085     fs_static_dry: np.ndarray
0086     p_failure_seismic: np.ndarray
0087     p_failure_fosm: np.ndarray
0088     effective_slope_deg: np.ndarray
0089     effective_soil_cohesion_kpa: np.ndarray
0090     fire_driving_multiplier: np.ndarray
0091     fs_mohr_coulomb: np.ndarray
0092     fs_postfire_yield_index: np.ndarray
0093     p_failure_m1: np.ndarray
0094     debris_volume_m3: np.ndarray
0095     rain_15min_proxy_mm: np.ndarray
0096     suction_cohesion_kpa: np.ndarray
0097     cryo_perched_head_m: np.ndarray
0098     ice_friction_factor: np.ndarray
0099     equivalent_water_mm: np.ndarray
0100     effective_burn_severity: np.ndarray
0101     cryospheric_eligible: np.ndarray
0102     lateral_resistance_kpa: np.ndarray
0103     warnings: List[str] = field(default_factory=list)
0104 
0105     @property
0106     def cell_count(self) -> int:
0107         return int(self.fs_dynamic.size)
0108 
0109     def to_dict(self) -> Dict[str, Any]:
0110         result = asdict(self)
0111         for key, value in result.items():
0112             if isinstance(value, np.ndarray):
0113                 result[key] = value.tolist()
0114         return result
0115 
0116 
0117 class MultiCausalEngine:
0118     """Evaluate soil hydraulics, five causal mechanisms, FS, FOSM, and Newmark."""
0119 
0120     def compute(self, inputs: PhysicsInputs) -> PhysicsState:
0121         values = self._broadcast_inputs(inputs)
0122         sand = np.clip(values["sand_fraction"], 0.05, 0.85)
0123         clay = np.clip(values["clay_fraction"], 0.05, 0.65)
0124         organic = 100.0 * np.clip(values["organic_matter_fraction"], 0.005, 0.06)  # coefficients use percent OM
0125         bulk_density = np.clip(values["bulk_density_g_cm3"], 1.15, 1.70)
0126         slope_deg = np.clip(values["slope_deg"], 0.0, 89.9)
0127         curvature = np.abs(values["profile_curvature"])
0128         amplified = slope_deg * (1.0 + 0.20 * np.clip(curvature / 0.015, 0.25, 1.0))
0129         # Never flatten a DEM slope already steeper than the 48 degree scenario cap.
0130         effective_slope = np.where(slope_deg >= 15.0, np.maximum(slope_deg, np.minimum(amplified, 48.0)), slope_deg)
0131         slope = np.radians(effective_slope)
0132         sin_slope, cos_slope = np.sin(slope), np.cos(slope)
0133         cos2 = cos_slope**2
0134         twi = values["twi"]
0135         initial_saturation = np.clip(values["soil_saturation_fraction"], 0.0, 1.0)
0136         rain = np.maximum(values["rain_24h_mm"], 0.0)
0137         storm_hr = values["storm_duration_hr"]
0138         snow_depth = np.maximum(values["snow_depth_m"], 0.0)
0139         air_temp = values["air_temperature_c"]
0140         soil_temp = values["soil_temperature_c"]
0141         freeze_cycles = np.maximum(values["freeze_thaw_cycles_7d"], 0.0)
0142         ndvi = np.clip(values["ndvi_norm"], 0.0, 1.0)
0143         burn = np.clip(values["burn_severity"], 0.0, 1.0) * np.clip(ndvi / .22, 0.0, 1.0)
0144         pga = np.maximum(values["pga_g"], 0.0)
0145         road_distance = np.maximum(values["road_cut_distance_m"], 0.0)
0146         river_distance = np.maximum(values["river_distance_m"], 0.0)
0147         if np.any(storm_hr <= 0):
0148             raise ValueError("storm_duration_hr must be positive for every cell")
0149         if np.any(twi <= 0):
0150             raise ValueError("TWI must be positive for every cell")
0151 
0152         # Saxton & Rawls 2006 equations, retaining the specified clip bounds.
0153         theta_s = np.clip(1.0 - bulk_density / PARTICLE_DENSITY_G_CM3, 0.34, 0.58)
0154         theta_1500_t = (
0155             -0.024 * sand
0156             + 0.487 * clay
0157             + 0.006 * organic
0158             + 0.005 * sand * organic
0159             - 0.013 * clay * organic
0160             + 0.068 * sand * clay
0161             + 0.031
0162         )
0163         theta_1500 = np.clip(theta_1500_t + 0.14 * theta_1500_t - 0.02, 0.05, 0.38)
0164         theta_33_t = (
0165             -0.251 * sand
0166             + 0.195 * clay
0167             + 0.011 * organic
0168             + 0.006 * sand * organic
0169             - 0.027 * clay * organic
0170             + 0.452 * sand * clay
0171             + 0.299
0172         )
0173         theta_33_raw = theta_33_t + 1.283 * theta_33_t**2 - 0.374 * theta_33_t - 0.015
0174         theta_33 = np.clip(theta_33_raw, theta_1500 + 0.05, theta_s - 0.04)
0175         retention_gap = np.log(theta_33) - np.log(theta_1500)
0176         b_parameter = np.divide(
0177             math.log(1500.0) - math.log(33.0),
0178             retention_gap,
0179             out=np.full_like(retention_gap, np.inf),
0180             where=np.abs(retention_gap) > 1e-12,
0181         )
0182         lambda_parameter = np.divide(
0183             1.0,
0184             b_parameter,
0185             out=np.zeros_like(b_parameter),
0186             where=np.isfinite(b_parameter) & (np.abs(b_parameter) > 1e-12),
0187         )
0188         ksat_raw = (
0189             1930.0
0190             * np.power(np.maximum(theta_s - theta_33, 1e-12), 3.0 - lambda_parameter)
0191             * (1e-3 / 3600.0)
0192         )
0193         ksat = np.clip(ksat_raw, 1e-7, 1e-3)
0194         friction_deg = np.clip(24.0 + 14.0 * sand - 8.0 * clay, 18.0, 39.0)
0195         tan_phi = np.tan(np.radians(friction_deg))
0196         cohesion_soil = np.clip(4.0 + 16.0 * clay - 3.5 * sand, 2.0, 20.0)
0197         base_depth = np.clip(2.8 * np.exp(-0.85 * np.tan(slope)), 0.60, 3.00)
0198         # Keep the slab thickness independent of the separate post-fire runoff branch.
0199         regolith_depth = base_depth.copy()  # slab depth is independent of the runoff branch
0200 
0201         snowmelt = np.where(
0202             (snow_depth > 0.01) & (air_temp > 0.0),
0203             np.maximum(0.0, 4.5 * air_temp + 0.0126 * rain * air_temp),
0204             0.0,
0205         )
0206         from .alpine_weather import cryospheric_eligibility
0207         baseline_air = np.where(values['baseline_air_temperature_c'] > -998,
0208                                 values['baseline_air_temperature_c'], air_temp-values['warming_delta_c'])
0209         observed_snow = np.where(values['observed_snow_depth_m'] >= 0,
0210                                  values['observed_snow_depth_m'], snow_depth)
0211         cryo_active = cryospheric_eligibility(observed_snow, baseline_air, values['elevation_m'], values['latitude_deg'])
0212         _, thaw, ice_friction = ice_factors(values['elevation_m'],soil_temp,snow_depth,freeze_cycles,values['warming_delta_c'],eligible=cryo_active)
0213         tan_phi_base = tan_phi.copy()
0214         tan_phi = tan_phi * ice_friction
0215         friction_deg = np.degrees(np.arctan(tan_phi))
0216         root_cohesion_baseline = 12.0 * np.power(ndvi, 1.35)
0217         root_cohesion = root_cohesion_baseline * (1.0 - 0.90 * burn)
0218         fire_multiplier = np.ones_like(burn)
0219         keff = ksat.copy()
0220 
0221         median_twi = float(np.median(twi))
0222         local_saturation = np.clip(initial_saturation * np.power(twi / median_twi, 0.28), 0.10, 0.99)
0223 
0224         fire_driving = np.ones_like(burn)
0225         soil_cohesion_effective = cohesion_soil.copy()
0226         infiltration = (
0227             rain * 1e-3 / (storm_hr * 3600.0) + snowmelt * 1e-3 / 86400.0
0228         ) * fire_multiplier
0229         hydraulic_diffusivity = 4.0 * keff * cos2 / np.maximum(theta_s * (1.0 - local_saturation), 0.015)
0230         dimensionless_time = storm_hr * 3600.0 * hydraulic_diffusivity / (regolith_depth**2)
0231         safe_time = np.maximum(dimensionless_time, 1e-12)
0232         iverson_response = (
0233             np.sqrt(safe_time / math.pi) * np.exp(-1.0 / safe_time)
0234             - self._erfc(1.0 / np.sqrt(safe_time))
0235         )
0236         pressure_head = np.clip(
0237             regolith_depth * local_saturation * cos2
0238             + regolith_depth * np.minimum(infiltration / keff, 4.0) * iverson_response,
0239             0.0,
0240             regolith_depth * cos2,
0241         )
0242 
0243         # Bounded active-layer water storage above an assumed low-permeability base.
0244         # This mass-balance approximation is not the TRIGRS PDE solution.
0245         basal_ratio=np.clip(values['basal_conductivity_ratio'],0,1)
0246         perched=np.where(cryo_active & (snowmelt>0),
0247             np.minimum(regolith_depth*cos2, np.maximum(0.,infiltration-ksat*basal_ratio)*storm_hr*3600/np.maximum(theta_s,.01)),0.)
0248         pressure_head=np.minimum(regolith_depth*cos2,pressure_head+perched)
0249         se=np.clip((local_saturation-.08)/.92,.02,.999)
0250         alpha=values['vg_alpha_kpa_inv'];n=values['vg_n']
0251         if np.any(alpha<=0) or np.any(n<=1):raise ValueError('Invalid van Genuchten parameters')
0252         suction_stress=-se/alpha*np.expm1(np.log(se)*n/(1-n))**(1/n)
0253         taper=np.clip((.85-se)/.10,0,1);taper=taper*taper*(3-2*taper)
0254         storm_activation=np.clip((rain+snowmelt)/25.,0,1)
0255         suction=np.clip(-suction_stress*tan_phi,0,8)*taper*(1-storm_activation)
0256         suction=np.where(perched>0,0.,suction)
0257         p_m1,volume,fs_fire,r15=postfire(burn,effective_slope,clay,sand,rain,values['peak_3h_intensity_mmhr'],base_depth)
0258 
0259         toe_multiplier = (
0260             1.0
0261             + 0.35 * np.exp(-road_distance / 60.0) * (slope_deg > 16.0)
0262             + 0.15 * np.exp(-river_distance / 80.0) * (twi / 15.0)
0263         )
0264         wet_unit_weight = (bulk_density + local_saturation * theta_s) * GAMMA_W_KN_M3
0265         effective_cohesion = (soil_cohesion_effective + root_cohesion) * thaw + suction
0266         kh = 0.50 * pga
0267 
0268         width = values['block_width_m']; confinement = values['hollow_confinement_ratio']
0269         if np.any(width <= 0) or np.any(confinement < 1):
0270             raise ValueError('Block width must be positive and hollow confinement ratio >=1')
0271         k0 = 1.0 - np.sin(np.radians(friction_deg))
0272         side_normal = np.maximum(0., wet_unit_weight*regolith_depth*cos_slope - pressure_head*GAMMA_W_KN_M3)
0273         lateral = (2.*regolith_depth/width) * (effective_cohesion + .5*k0*side_normal*tan_phi) / confinement
0274         # Requested finite-block adaptation, not the complete Milledge 2014 MD-STAB model.
0275         normal_static = wet_unit_weight * regolith_depth * cos2 - pressure_head * GAMMA_W_KN_M3
0276         driving_static = wet_unit_weight * regolith_depth * sin_slope * cos_slope * toe_multiplier * fire_driving
0277         fs_static_raw = (effective_cohesion + lateral + normal_static * tan_phi) / np.maximum(driving_static, 1.0)
0278         fs_static = np.clip(fs_static_raw, 0.10, 5.00)
0279 
0280         normal_dynamic = (
0281             wet_unit_weight * regolith_depth * (cos2 - kh * sin_slope * cos_slope)
0282             - pressure_head * GAMMA_W_KN_M3
0283         )
0284         driving_dynamic = (
0285             wet_unit_weight
0286             * regolith_depth
0287             * (sin_slope * cos_slope + kh * cos2)
0288             * toe_multiplier * fire_driving
0289         )
0290         denominator_dynamic = np.maximum(driving_dynamic, 1.0)
0291         fs_dynamic_raw = (effective_cohesion + lateral + normal_dynamic * tan_phi) / denominator_dynamic
0292         fs_mohr = np.clip(fs_dynamic_raw, 0.10, 5.00)
0293         fs_dynamic = np.minimum(fs_mohr,fs_fire)
0294 
0295         # Delta-method FOSM, including the derivative of the specified FS clip.
0296         sigma_cohesion = 0.18 * effective_cohesion
0297         lateral_coefficient = 2.*regolith_depth/width/confinement
0298         sigma_tan_phi = 0.12 * tan_phi
0299         sigma_pressure = 0.15 * pressure_head
0300         clip_active = (fs_dynamic_raw > 0.10) & (fs_dynamic_raw < 5.00)
0301         d_fs_dc = np.where(clip_active, (1.+lateral_coefficient) / denominator_dynamic, 0.0)
0302         d_fs_dtan = np.where(clip_active, (normal_dynamic + .5*lateral_coefficient*side_normal*(k0-tan_phi/(1.+tan_phi**2)**1.5)) / denominator_dynamic, 0.0)
0303         d_fs_dpressure = np.where(
0304             clip_active, -GAMMA_W_KN_M3 * tan_phi*(1.+.5*lateral_coefficient*k0*(side_normal>0)) / denominator_dynamic, 0.0
0305         )
0306         sigma_fs = np.sqrt(
0307             (d_fs_dc * sigma_cohesion) ** 2
0308             + (d_fs_dtan * sigma_tan_phi) ** 2
0309             + (d_fs_dpressure * sigma_pressure) ** 2
0310         )
0311         beta = (fs_mohr - 1.0) / np.maximum(sigma_fs, 0.05)
0312         p_failure = 0.5 * self._erfc(beta / math.sqrt(2.0))
0313 
0314         # Pre-storm, unsaturated, unburned baseline; excludes seismic load and thaw.
0315         dry_gamma = bulk_density * GAMMA_W_KN_M3
0316         dry_normal = dry_gamma * base_depth * cos2
0317         dry_driving = dry_gamma * base_depth * sin_slope * cos_slope * toe_multiplier
0318         dry_phi=np.arctan(tan_phi_base)
0319         dry_lateral=(2.*base_depth/width/confinement)*(cohesion_soil+root_cohesion_baseline
0320                      +.5*(1.-np.sin(dry_phi))*dry_gamma*base_depth*cos_slope*tan_phi_base)
0321         fs_static_dry = np.clip((cohesion_soil + root_cohesion_baseline + dry_lateral + dry_normal * tan_phi_base)
0322                                / np.maximum(dry_driving, 1.0), 0.10, 5.0)
0323         critical_acceleration = np.clip((fs_static_dry - 1.0) * sin_slope, 0.05, 0.60)
0324         pga_ratio = np.clip(
0325             critical_acceleration / np.maximum(pga, 1e-12), 1e-12, 1.0 - 1e-12
0326         )
0327         newmark = np.where(
0328             pga > critical_acceleration,
0329             np.power(10.0, 0.215 + 2.341 * np.log10(1.0 - pga_ratio) - 1.438 * np.log10(pga_ratio)),
0330             0.0,
0331         )
0332 
0333         # User-specified detachment cap; retain a separate regional empirical component.
0334         newmark = np.minimum(newmark, 120.0)
0335         p_seismic = 0.335 * (-np.expm1(-0.048 * np.power(newmark, 1.565)))
0336         # Requested probabilistic union; this treats the three components as independent.
0337         p_fosm = p_failure.copy()
0338         p_failure = 1.0-(1.0-p_fosm)*(1.0-p_seismic)*(1.0-p_m1)
0339         warnings = ["M1/Gartner inputs are cell proxies; postfire FS is a yield index. Ice weakening, basal drainage and VG parameters are scenario assumptions. Probability union assumes independent components."]
0340         reversed_bounds = np.where(theta_33 <= theta_1500)[0]
0341         if reversed_bounds.size:
0342             warnings.append(
0343                 "Saxton-Rawls bounds produced theta_33 <= theta_1500 for {} cell(s); exact requested clipping was retained".format(
0344                     int(reversed_bounds.size)
0345                 )
0346             )
0347         return PhysicsState(
0348             theta_s=theta_s,
0349             theta_1500=theta_1500,
0350             theta_33=theta_33,
0351             hydraulic_conductivity_m_s=ksat,
0352             friction_angle_deg=friction_deg,
0353             soil_cohesion_kpa=cohesion_soil,
0354             regolith_depth_m=regolith_depth,
0355             snowmelt_mm_day=snowmelt,
0356             thaw_factor=thaw,
0357             root_cohesion_kpa=root_cohesion,
0358             fire_runoff_multiplier=fire_multiplier,
0359             effective_conductivity_m_s=keff,
0360             effective_infiltration_m_s=infiltration,
0361             local_saturation=local_saturation,
0362             hydraulic_diffusivity_m2_s=hydraulic_diffusivity,
0363             dimensionless_time=dimensionless_time,
0364             iverson_response=iverson_response,
0365             pore_pressure_head_m=pressure_head,
0366             toe_undercut_multiplier=toe_multiplier,
0367             wet_unit_weight_kn_m3=wet_unit_weight,
0368             effective_cohesion_kpa=effective_cohesion,
0369             kh=kh,
0370             fs_static=fs_static,
0371             fs_dynamic=fs_dynamic,
0372             sigma_fs=sigma_fs,
0373             beta_fosm=beta,
0374             p_failure_physical=p_failure,
0375             critical_acceleration_g=critical_acceleration,
0376             newmark_displacement_cm=newmark,
0377             effective_slope_deg=effective_slope, effective_soil_cohesion_kpa=soil_cohesion_effective,
0378             fire_driving_multiplier=fire_driving,
0379             unburned_regolith_depth_m=base_depth, fs_static_dry=fs_static_dry,
0380             p_failure_seismic=p_seismic, p_failure_fosm=p_fosm, warnings=warnings,
0381             fs_mohr_coulomb=fs_mohr, fs_postfire_yield_index=fs_fire, p_failure_m1=p_m1,
0382             debris_volume_m3=volume, rain_15min_proxy_mm=r15, suction_cohesion_kpa=suction,
0383             cryo_perched_head_m=perched, ice_friction_factor=ice_friction,
0384             equivalent_water_mm=rain+1.35*snowmelt,
0385             effective_burn_severity=burn, cryospheric_eligible=cryo_active, lateral_resistance_kpa=lateral,
0386         )
0387 
0388     @staticmethod
0389     def _broadcast_inputs(inputs: PhysicsInputs) -> Dict[str, np.ndarray]:
0390         arrays = {}
0391         for name in PhysicsInputs.__dataclass_fields__:
0392             value = np.asarray(getattr(inputs, name), dtype=np.float64)
0393             if value.ndim > 1:
0394                 raise ValueError("{} must be a scalar or one-dimensional per-cell array".format(name))
0395             arrays[name] = np.atleast_1d(value)
0396         broadcasted = np.broadcast_arrays(*arrays.values())
0397         result = {name: array.copy() for name, array in zip(arrays, broadcasted)}
0398         for name, array in result.items():
0399             if not np.isfinite(array).all():
0400                 raise ValueError("{} contains missing or non-finite values; provide explicit observations/scenarios".format(name))
0401         return result
0402 
0403     @staticmethod
0404     def _erfc(values: np.ndarray) -> np.ndarray:
0405         # math.erfc is correctly rounded by the platform libm; vectorize without SciPy dependency.
0406         function = np.frompyfunc(math.erfc, 1, 1)
0407         return function(values).astype(np.float64)
```

## backend/physics/mechanisms.py

SHA-256: `aa3a9832506e64137a03ca630ae6a18b09f4a12029a17f985f4bda7744208735`

```python
0001 """Published equation forms with explicitly identified scenario input proxies.
0002 
0003 References and domain restrictions: docs/physics_sources.md. The ice factors and
0004 post-fire yield index are requested scenario parameterizations, not paper equations.
0005 """
0006 import numpy as np
0007 
0008 def sigmoid(x):
0009     return 1.0/(1.0+np.exp(-np.clip(x,-700,700)))
0010 
0011 def postfire(burn,slope_deg,clay,sand,rain,peak_mmhr,depth):
0012     r15=np.maximum(.25*np.maximum(peak_mmhr,0),.12*np.maximum(rain,0))
0013     prop=burn*sigmoid((slope_deg-21.)/2.5)
0014     dnbr=.85*burn
0015     kf=np.clip(.18+.35*(1-clay)-.15*sand,.15,.45)
0016     link=-3.63+(.41*prop+.67*dnbr+.70*kf)*r15
0017     active=(burn>.05)&(slope_deg>=12.)
0018     probability=np.where(active,sigmoid(link),0.)
0019     # Gartner emergency model coefficients; area and relief here are cell proxies.
0020     area=np.maximum(.105*burn,.01)
0021     relief=np.maximum(depth*50*np.sin(np.radians(slope_deg)),10.)
0022     ln_volume=4.22+.39*np.sqrt(4*r15)+.36*np.log(area)+.13*np.sqrt(relief)
0023     volume=np.where(active,np.exp(np.clip(ln_volume,-700,700)),0.)
0024     yield_index=np.where(probability>.5,np.clip(1.05-1.5*(probability-.5),.3,2.5),5.)
0025     return probability,volume,yield_index,r15
0026 
0027 def suction_cohesion(saturation,tan_phi,active_head,alpha_kpa_inv=.1,n=2.):
0028     if alpha_kpa_inv<=0 or n<=1:raise ValueError('van Genuchten alpha >0 and n >1 required')
0029     se=np.clip((saturation-.08)/.92,.02,.999)
0030     stress=-se/alpha_kpa_inv*np.expm1(np.log(se)*n/(1-n))**(1/n)
0031     # Smooth switch reaches zero at Se=.85, avoiding a discontinuous cutoff.
0032     taper=np.clip((.85-se)/.10,0,1);taper=taper*taper*(3-2*taper)
0033     return np.where(active_head,0.,np.clip(-stress*tan_phi,0,8)*taper)
0034 
0035 def ice_factors(elevation,soil_temp,snow,cycles,warming,eligible=None):
0036     if eligible is None:
0037         from .alpine_weather import cryospheric_eligibility
0038         eligible=cryospheric_eligibility(snow,soil_temp,elevation,0.)
0039     # Delta is an explicit temperature perturbation, not an observed joint history.
0040     delta=np.clip(warming,0,7.9)
0041     cohesion=np.where(eligible,np.clip(1-.12*delta-.10*cycles,.14,1),1.)
0042     friction=np.where(eligible,np.clip(1-.08*delta-.05*cycles,.40,1),1.)
0043     return eligible,cohesion,friction
0044 
0045 def mamot_joint_strength(temperature_c,normal_stress_kpa):
0046     """Mamot 2018 Eqs.4–6, for verified ice-filled joints in the tested domain."""
0047     t,stress=np.broadcast_arrays(np.asarray(temperature_c,float),np.asarray(normal_stress_kpa,float))
0048     if np.any(~np.isfinite(t)) or np.any(~np.isfinite(stress)) or np.any((t < -8)|(t > -.5)|(stress <100)|(stress>400)):
0049         raise ValueError('Mamot calibration requires -8..-0.5 C and 100..400 kPa, with known ice-filled joints')
0050     cohesion=53.3-73.5*t
0051     friction=.42-.21*t
0052     return cohesion+stress*friction
```

## backend/physics/alpine_weather.py

SHA-256: `8f18c1f642d9ce64d093437f01b4a9ee13604aca51e677d8f9d25803c566c230`

```python
0001 """Baseline-gated cryospheric scenarios; warming alone cannot create snow."""
0002 from dataclasses import replace
0003 import numpy as np
0004 
0005 ALPINE_ASSUMPTION = "Lapse-rate and latitude/elevation screen are scenario proxies, not observed permafrost. Snow/cycle floors apply only inside the baseline cryospheric screen."
0006 
0007 def cryospheric_eligibility(snow_depth_m, baseline_temperature_c, elevation_m, latitude_deg):
0008     line = np.maximum(1800.0, 4600.0 - 55.0 * np.abs(latitude_deg))
0009     return (np.asarray(snow_depth_m) > .005) | (np.asarray(baseline_temperature_c) <= 2.5) | (np.asarray(elevation_m) >= line)
0010 
0011 def apply_alpine_weather(inputs, elevations_m, snowmelt_temp_surge_c=0.0, latitude_deg=None):
0012     elevation = np.asarray(elevations_m, dtype=float)
0013     latitude = inputs.latitude_deg if latitude_deg is None else latitude_deg
0014     lapse = .0065 * np.maximum(0.0, elevation - elevation.min())
0015     baseline = np.asarray(inputs.air_temperature_c) - lapse
0016     eligible = cryospheric_eligibility(inputs.snow_depth_m, baseline, elevation, latitude)
0017     snow_floor = .40 * np.clip((elevation - 1800.0) / 1000.0, .2, 1.5)
0018     snow = np.where(eligible, np.maximum(inputs.snow_depth_m, snow_floor), inputs.snow_depth_m)
0019     cycles = np.where(eligible, np.maximum(inputs.freeze_thaw_cycles_7d, 3.), inputs.freeze_thaw_cycles_7d)
0020     return replace(inputs, air_temperature_c=baseline + snowmelt_temp_surge_c,
0021                    soil_temperature_c=np.asarray(inputs.soil_temperature_c)-lapse+.5*snowmelt_temp_surge_c,
0022                    snow_depth_m=snow, freeze_thaw_cycles_7d=cycles, elevation_m=elevation,
0023                    latitude_deg=latitude, warming_delta_c=snowmelt_temp_surge_c,
0024                    baseline_air_temperature_c=baseline, observed_snow_depth_m=inputs.snow_depth_m)
```

## backend/physics/kinematics.py

SHA-256: `a92b0ddc63977bae571b4b52750142d6d7fa3f48d7d6916ac1b72d7887bfad69`

```python
0001 """Four-branch landslide velocity solver and H3 Voellmy-style routing."""
0002 
0003 from __future__ import annotations
0004 
0005 import math
0006 from dataclasses import asdict, dataclass, field
0007 from typing import Dict, List, Mapping, Optional, Sequence, Tuple
0008 
0009 import h3
0010 import numpy as np
0011 
0012 from .multi_causal_engine import PhysicsInputs, PhysicsState
0013 
0014 GRAVITY_M_S2 = 9.81
0015 
0016 
0017 @dataclass
0018 class KinematicsState:
0019     peak_velocity_m_s: np.ndarray
0020     velocity_class: np.ndarray
0021     velocity_class_name: np.ndarray
0022     failure_mode: np.ndarray
0023 
0024     def to_dict(self) -> Dict[str, object]:
0025         return {
0026             "peak_velocity_m_s": self.peak_velocity_m_s.tolist(),
0027             "velocity_class": self.velocity_class.tolist(),
0028             "velocity_class_name": self.velocity_class_name.tolist(),
0029             "failure_mode": self.failure_mode.tolist(),
0030         }
0031 
0032 
0033 @dataclass
0034 class RunoutRoute:
0035     source_cell: str
0036     path: List[str]
0037     arrival_time_min: float
0038     final_energy_head_m: float
0039     intersected_roads: int
0040     intersected_buildings: int
0041     intersected_households: Optional[int]
0042     stopped_at: str
0043 
0044     def to_dict(self) -> Dict[str, object]:
0045         return asdict(self)
0046 
0047 
0048 def solve_kinematics(inputs: PhysicsInputs, state: PhysicsState) -> KinematicsState:
0049     """Apply the ordered rapid-flow, coseismic, slump, and creep branches."""
0050     slope = np.radians(np.clip(state.effective_slope_deg, 0.0, 89.9))
0051     burn = state.effective_burn_severity
0052     rain = np.maximum(np.asarray(inputs.rain_24h_mm, dtype=np.float64), 0.0)
0053     snow_depth = np.maximum(np.asarray(inputs.snow_depth_m, dtype=np.float64), 0.0)
0054     pga = np.maximum(np.asarray(inputs.pga_g, dtype=np.float64), 0.0)
0055     tan_phi = np.tan(np.radians(state.friction_angle_deg))
0056 
0057     rapid = (state.fs_dynamic <= 1.0) & (((state.fs_dynamic <= 1.0) & (state.local_saturation >= 0.68)) | ((burn >= 0.35) & (rain >= 25.0)))
0058     coseismic = (
0059         (~rapid) & (state.fs_dynamic <= 1.0)
0060         & (pga > state.critical_acceleration_g)
0061         & (state.newmark_displacement_cm >= 2.0)
0062         & (np.asarray(inputs.slope_deg) >= 28.0)
0063     )
0064     slump = (~rapid) & (~coseismic) & (state.fs_dynamic <= 1.25)
0065     stable = (~rapid) & (~coseismic) & (~slump)
0066 
0067     flow_depth = state.regolith_depth_m * np.clip(
0068         state.pore_pressure_head_m / (state.regolith_depth_m * np.cos(slope) ** 2), 0.25, 1.0
0069     )
0070     rapid_speed = np.sqrt(
0071         np.maximum(
0072             0.0,
0073             (300.0 + 150.0 * burn)
0074             * flow_depth
0075             * np.maximum(0.002, np.sin(slope) - 0.18 * np.cos(slope)),
0076         )
0077     )
0078     coseismic_speed = np.sqrt(
0079         2.0
0080         * GRAVITY_M_S2
0081         * 120.0
0082         * np.maximum(
0083             0.005,
0084             np.sin(slope)
0085             + state.kh * np.cos(slope)
0086             - 0.65 * tan_phi * np.cos(slope),
0087         )
0088     )
0089     slump_speed = (
0090         0.08
0091         * np.sin(slope)
0092         * np.power(np.maximum(0.0001, (1.28 / state.fs_dynamic) - 1.0), 2.2)
0093     )
0094     creep_speed = np.zeros_like(state.fs_dynamic)  # stable: no resolved mass movement
0095     speed = np.select([rapid, coseismic, slump, stable], [rapid_speed, coseismic_speed, slump_speed, creep_speed])
0096     velocity_class = np.select(
0097         [
0098             speed >= 5.0,
0099             speed >= 5e-2,
0100             speed >= 5e-4,
0101             speed >= 5e-6,
0102             speed >= 5e-8,
0103             speed >= 5e-10,
0104         ],
0105         [7, 6, 5, 4, 3, 2],
0106         default=1,
0107     ).astype(np.int8)
0108     class_names = np.asarray(
0109         [
0110             "Stable",
0111             "Extremely Slow",
0112             "Very Slow",
0113             "Slow",
0114             "Moderate",
0115             "Rapid",
0116             "Very Rapid",
0117             "Extremely Rapid",
0118         ],
0119         dtype=object,
0120     )[velocity_class]
0121     modes = np.select(
0122         [rapid, coseismic, slump, stable],
0123         ["Rapid fluidized debris flow / avalanche", "Coseismic rock avalanche / block slide", "Cohesive slump / thaw slump / toe-cut wedge", "Stable / sub-perception creep"],
0124         default="Unclassified",
0125     )
0126     return KinematicsState(speed, velocity_class, class_names, modes)
0127 
0128 
0129 def select_runout_hotspots(state: PhysicsState, probability: Optional[np.ndarray] = None) -> np.ndarray:
0130     """Indices meeting the requested FS or probability hotspot thresholds."""
0131     selected = state.fs_dynamic <= 1.05
0132     if probability is not None:
0133         probability = np.asarray(probability, dtype=np.float64)
0134         if probability.shape != state.fs_dynamic.shape:
0135             raise ValueError("probability must have the same shape as PhysicsState")
0136         selected = selected | (probability >= 0.65)
0137     return np.where(selected)[0]
0138 
0139 
0140 def route_runout(
0141     source_cell: str,
0142     terrain_cells: Mapping[str, object],
0143     peak_velocity_m_s: float,
0144     mu_b: float,
0145     xi: float,
0146     infrastructure_by_cell: Optional[Mapping[str, Mapping[str, int]]] = None,
0147     delta_s_by_edge: Optional[Mapping[Tuple[str, str], float]] = None,
0148     max_steps: int = 50,
0149 ) -> RunoutRoute:
0150     """Route positive energy head over lower H3 neighbors.
0151 
0152     For each edge, Δs is supplied in ``delta_s_by_edge`` when available; otherwise
0153     it is the current cell's dimensionless sin(slope) factor. The user-specified
0154     Voellmy head change is applied exactly as ΔE = Δz - μb Δx - (v²/(g ξ)) Δs.
0155     Friction and ξ are required inputs because the prompt gives no calibrated values.
0156     """
0157     if source_cell not in terrain_cells:
0158         raise KeyError("source cell is absent from the terrain graph")
0159     if not math.isfinite(mu_b) or not math.isfinite(xi) or mu_b < 0 or xi <= 0 or max_steps < 1:
0160         raise ValueError("mu_b must be nonnegative, xi positive, and max_steps at least 1")
0161     if peak_velocity_m_s < 0 or not math.isfinite(peak_velocity_m_s):
0162         raise ValueError("peak velocity must be finite and nonnegative")
0163 
0164     energy_head = peak_velocity_m_s**2 / (2.0 * GRAVITY_M_S2)
0165     path = [source_cell]
0166     elapsed_seconds = 0.0
0167     current = source_cell
0168     visited = {source_cell}
0169     infra = infrastructure_by_cell or {}
0170     edge_s = delta_s_by_edge or {}
0171 
0172     for _ in range(min(int(max_steps), 50)):
0173         if energy_head <= 0.0:
0174             break
0175         cell = terrain_cells[current]
0176         neighbors = list(getattr(cell, "downslope_neighbors", []) or [])
0177         candidates = []
0178         current_elevation = float(getattr(cell, "elevation_m"))
0179         current_speed = math.sqrt(max(0.0, 2.0 * GRAVITY_M_S2 * energy_head))
0180         for neighbor_id in neighbors:
0181             if neighbor_id in visited or neighbor_id not in terrain_cells:
0182                 continue
0183             neighbor = terrain_cells[neighbor_id]
0184             dz = current_elevation - float(getattr(neighbor, "elevation_m"))
0185             if dz <= 0.0:
0186                 continue
0187             dx = _distance_m(cell, neighbor)
0188             delta_s = max(0.0, float(edge_s.get((current, neighbor_id), math.sin(math.radians(float(getattr(cell, "slope_deg")))))))
0189             delta_energy = dz - mu_b * dx - (current_speed**2 / (GRAVITY_M_S2 * xi)) * delta_s
0190             next_head = energy_head + delta_energy
0191             if next_head > 0.0:
0192                 candidates.append((next_head, neighbor_id, dx))
0193         if not candidates:
0194             break
0195         next_head, next_cell, dx = max(candidates, key=lambda candidate: candidate[0])
0196         next_speed = math.sqrt(2.0 * GRAVITY_M_S2 * next_head)
0197         mean_speed = max(1e-4, (current_speed + next_speed) / 2.0)
0198         elapsed_seconds += dx / mean_speed
0199         energy_head = next_head
0200         current = next_cell
0201         visited.add(current)
0202         path.append(current)
0203 
0204     road_hits = sum(int(infra.get(cell_id, {}).get("roads", 0)) for cell_id in path)
0205     building_hits = sum(int(infra.get(cell_id, {}).get("buildings", 0)) for cell_id in path)
0206     household_counts = [infra.get(cell_id, {}).get("households") for cell_id in path]
0207     household_hits = (
0208         sum(int(value) for value in household_counts)
0209         if all(value is not None for value in household_counts)
0210         else None
0211     )
0212     return RunoutRoute(
0213         source_cell=source_cell,
0214         path=path,
0215         arrival_time_min=elapsed_seconds / 60.0,
0216         final_energy_head_m=energy_head,
0217         intersected_roads=road_hits,
0218         intersected_buildings=building_hits,
0219         intersected_households=household_hits,
0220         stopped_at=path[-1],
0221     )
0222 
0223 
0224 def route_hotspots(
0225     terrain_cells: Mapping[str, object],
0226     cell_ids: Sequence[str],
0227     state: PhysicsState,
0228     kinematics: KinematicsState,
0229     mu_b: float,
0230     xi: float,
0231     probability: Optional[np.ndarray] = None,
0232     infrastructure_by_cell: Optional[Mapping[str, Mapping[str, int]]] = None,
0233     max_routes: int = 10,
0234 ) -> List[RunoutRoute]:
0235     """Route up to ``max_routes`` cells meeting FS≤1.05 or failure probability≥0.65."""
0236     if len(cell_ids) != state.cell_count or kinematics.peak_velocity_m_s.size != state.cell_count:
0237         raise ValueError("cell_ids, PhysicsState, and KinematicsState must have matching cell counts")
0238     if max_routes < 1:
0239         return []
0240     hotspots = select_runout_hotspots(state, probability)
0241     if probability is not None:
0242         p = np.asarray(probability, dtype=np.float64)
0243         hotspots = sorted(hotspots, key=lambda idx: (p[idx], -state.fs_dynamic[idx]), reverse=True)
0244     else:
0245         hotspots = sorted(hotspots, key=lambda idx: (state.p_failure_physical[idx], -state.fs_dynamic[idx]), reverse=True)
0246     routes = []
0247     for idx in hotspots[:max_routes]:
0248         cell_id = cell_ids[int(idx)]
0249         if cell_id not in terrain_cells:
0250             continue
0251         routes.append(
0252             route_runout(
0253                 cell_id,
0254                 terrain_cells,
0255                 float(kinematics.peak_velocity_m_s[idx]),
0256                 mu_b,
0257                 xi,
0258                 infrastructure_by_cell=infrastructure_by_cell,
0259             )
0260         )
0261     return routes
0262 
0263 
0264 def _distance_m(a: object, b: object) -> float:
0265     a_lat, a_lon = float(getattr(a, "latitude")), float(getattr(a, "longitude"))
0266     b_lat, b_lon = float(getattr(b, "latitude")), float(getattr(b, "longitude"))
0267     try:
0268         return float(h3.great_circle_distance((a_lat, a_lon), (b_lat, b_lon), unit="m"))
0269     except TypeError:
0270         lat1, lat2 = math.radians(a_lat), math.radians(b_lat)
0271         dlat, dlon = lat2 - lat1, math.radians(b_lon - a_lon)
0272         hav = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
0273         return 2.0 * 6371008.8 * math.asin(math.sqrt(hav))
```

## backend/physics/causal_attribution.py

SHA-256: `2beffcddf47b5236adca948f0811879eca77755423362009dae4d2e5727e2fe1`

```python
0001 """Regime-level stress decomposition and location-specific threshold cards."""
0002 
0003 from __future__ import annotations
0004 
0005 import math
0006 from dataclasses import asdict, dataclass, replace
0007 from typing import Dict, List, Mapping, Optional
0008 
0009 import numpy as np
0010 
0011 from .multi_causal_engine import GAMMA_W_KN_M3, MultiCausalEngine, PhysicsInputs, PhysicsState
0012 
0013 CAUSES = (
0014     "hydro_meteorological",
0015     "seismic_tectonic",
0016     "cryospheric",
0017     "post_wildfire_biotic",
0018     "anthropogenic_fluvial",
0019 )
0020 
0021 
0022 @dataclass
0023 class CausalBreakdown:
0024     delta_tau_kpa: Dict[str, np.ndarray]
0025     physical_percentages: Dict[str, np.ndarray]
0026     shap_percentages: Optional[Dict[str, np.ndarray]]
0027     combined_percentages: Dict[str, np.ndarray]
0028     absolute_shap_log_odds: Optional[Dict[str, np.ndarray]]
0029     valid: np.ndarray
0030 
0031     def to_dict(self) -> Dict[str, object]:
0032         return {
0033             "delta_tau_kpa": {key: value.tolist() for key, value in self.delta_tau_kpa.items()},
0034             "physical_percentages": {key: value.tolist() for key, value in self.physical_percentages.items()},
0035             "shap_percentages": None
0036             if self.shap_percentages is None
0037             else {key: value.tolist() for key, value in self.shap_percentages.items()},
0038             "combined_percentages": {key: value.tolist() for key, value in self.combined_percentages.items()},
0039             "absolute_shap_log_odds": None
0040             if self.absolute_shap_log_odds is None
0041             else {key: value.tolist() for key, value in self.absolute_shap_log_odds.items()},
0042             "valid": self.valid.tolist(),
0043         }
0044 
0045 
0046 @dataclass
0047 class NumbersThatMatterItem:
0048     metric: str
0049     value: float
0050     unit: str
0051     threshold: str
0052     status: str
0053     causal_group: str
0054     weight_percent: float
0055 
0056     def to_dict(self) -> Dict[str, object]:
0057         return asdict(self)
0058 
0059 
0060 def explain_causes(
0061     inputs: PhysicsInputs,
0062     state: PhysicsState,
0063     shap_values: Optional[np.ndarray] = None,
0064     feature_names: Optional[List[str]] = None,
0065 ) -> CausalBreakdown:
0066     """Return five physical shear reductions and optional absolute TreeSHAP shares.
0067 
0068     Physical contributions remain in kPa. They are normalized separately from
0069     dimensionless SHAP log-odds values; the combined percentage is the mean of
0070     the two normalized distributions when both are available.
0071     """
0072     slope = np.radians(np.clip(state.effective_slope_deg, 0.0, 89.9))
0073     ndvi = np.clip(np.asarray(inputs.ndvi_norm, dtype=np.float64), 0.0, 1.0)
0074     burn = state.effective_burn_severity
0075     tan_phi = np.tan(np.radians(state.friction_angle_deg))
0076     root_cohesion_unburned = 12.0 * ndvi**1.35
0077     slope_cos2 = np.cos(slope) ** 2
0078     slope_deg = np.asarray(inputs.slope_deg, dtype=np.float64)
0079     twi = np.maximum(np.asarray(inputs.twi, dtype=np.float64), 0.01)
0080     road_distance = np.broadcast_to(np.asarray(inputs.road_cut_distance_m, dtype=np.float64), (state.cell_count,))
0081     river_distance = np.broadcast_to(np.asarray(inputs.river_distance_m, dtype=np.float64), (state.cell_count,))
0082     field_capacity_saturation = np.clip(state.theta_33 / np.maximum(state.theta_s, 1e-6), 0.0, 0.95)
0083     field_capacity_head = state.regolith_depth_m * field_capacity_saturation * slope_cos2
0084     excess_pore_pressure_head = np.maximum(0.0, state.pore_pressure_head_m - field_capacity_head)
0085     snowmelt_activation = np.clip(state.snowmelt_mm_day / 60.0, 0.0, 0.75)
0086     cryospheric_degradation = np.maximum(1.0 - state.thaw_factor, snowmelt_activation)
0087     if np.any(burn > 0.0):
0088         unburned_state = MultiCausalEngine().compute(replace(inputs, burn_severity=np.zeros_like(burn)))
0089         unburned_excess_head = np.maximum(
0090             0.0, unburned_state.pore_pressure_head_m * state.regolith_depth_m / unburned_state.regolith_depth_m - field_capacity_head
0091         )
0092         fire_amplified_excess_head = np.maximum(
0093             0.0, excess_pore_pressure_head - unburned_excess_head
0094         )
0095         fire_runoff_stress = fire_amplified_excess_head * GAMMA_W_KN_M3 * tan_phi
0096     else:
0097         fire_runoff_stress = np.zeros(state.cell_count, dtype=np.float64)
0098     road_undercut = 0.35 * np.exp(-road_distance / 60.0) * (slope_deg > 16.0)
0099     # River distance is estimated from high-TWI DEM cells, not a mapped stream;
0100     # discount that proxy contribution while retaining measured road-cut effects.
0101     fluvial_proxy_undercut = 0.15 * np.exp(-river_distance / 80.0) * (twi / 15.0)
0102     fire_drag_stress = state.wet_unit_weight_kn_m3 * state.regolith_depth_m * (
0103         np.sin(slope)*np.cos(slope) + state.kh*slope_cos2
0104     ) * state.toe_undercut_multiplier * (state.fire_driving_multiplier - 1.0)
0105     fire_cohesion_loss = (state.soil_cohesion_kpa-state.effective_soil_cohesion_kpa + root_cohesion_unburned-state.root_cohesion_kpa) * state.thaw_factor
0106     stress = {
0107         "hydro_meteorological": np.maximum(
0108             0.0, excess_pore_pressure_head * GAMMA_W_KN_M3 * tan_phi - fire_runoff_stress
0109         ),
0110         "seismic_tectonic": np.maximum(
0111             0.0,
0112             state.wet_unit_weight_kn_m3
0113             * state.regolith_depth_m
0114             * state.kh
0115             * np.cos(slope) ** 2,
0116         ),
0117         "cryospheric": np.maximum(
0118             0.0,
0119             (state.effective_soil_cohesion_kpa + state.root_cohesion_kpa)
0120             * cryospheric_degradation,
0121         ),
0122         "post_wildfire_biotic": np.maximum(
0123             0.0, fire_cohesion_loss + fire_runoff_stress + fire_drag_stress + state.p_failure_m1 * state.wet_unit_weight_kn_m3 * state.regolith_depth_m * np.sin(slope)
0124         ),
0125         "anthropogenic_fluvial": np.maximum(
0126             0.0,
0127             state.wet_unit_weight_kn_m3
0128             * state.regolith_depth_m
0129             * np.sin(slope)
0130             * np.cos(slope)
0131             * (road_undercut + 0.5 * fluvial_proxy_undercut),
0132         ),
0133     }
0134     physical = _normalize(stress)
0135     shap_by_cause = None
0136     shap_pct = None
0137     if shap_values is not None:
0138         shap = np.asarray(shap_values, dtype=np.float64)
0139         if shap.ndim != 2 or shap.shape[0] != state.cell_count:
0140             raise ValueError("shap_values must have shape (cell_count, feature_count)")
0141         if feature_names is None or len(feature_names) != shap.shape[1]:
0142             raise ValueError("feature_names matching shap_values columns are required")
0143         index = {name: idx for idx, name in enumerate(feature_names)}
0144         feature_groups = {
0145             "hydro_meteorological": ("psi_head_m", "soil_saturation", "rain_24h_mm"),
0146             "seismic_tectonic": ("pga_g", "newmark_dn_cm"),
0147             "cryospheric": ("snowmelt_mm_day", "thaw_degradation"),
0148             "post_wildfire_biotic": ("burn_severity",),
0149             "anthropogenic_fluvial": ("toe_undercut_mult",),
0150         }
0151         shap_by_cause = {}
0152         for cause, names in feature_groups.items():
0153             columns = [index[name] for name in names if name in index]
0154             shap_by_cause[cause] = (
0155                 np.abs(shap[:, columns]).sum(axis=1)
0156                 if columns
0157                 else np.zeros(state.cell_count, dtype=np.float64)
0158             )
0159         shap_pct = _normalize(shap_by_cause)
0160 
0161     if shap_pct is None:
0162         combined = physical
0163     else:
0164         physical_total = np.sum(np.stack(list(stress.values()), axis=0), axis=0)
0165         shap_total = np.sum(np.stack(list(shap_by_cause.values()), axis=0), axis=0)
0166         combined = {}
0167         for cause in CAUSES:
0168             combined[cause] = np.where(
0169                 (physical_total > 0) & (shap_total > 0),
0170                 0.5 * (physical[cause] + shap_pct[cause]),
0171                 np.where(shap_total > 0, shap_pct[cause], physical[cause]),
0172             )
0173     valid = np.sum(np.stack(list(stress.values()), axis=0), axis=0) > 1e-8
0174     if shap_by_cause is not None:
0175         valid = valid | (np.sum(np.stack(list(shap_by_cause.values()), axis=0), axis=0) > 0)
0176     return CausalBreakdown(stress, physical, shap_pct, combined, shap_by_cause, valid)
0177 
0178 
0179 def select_numbers_that_matter(
0180     index: int,
0181     inputs: PhysicsInputs,
0182     state: PhysicsState,
0183     breakdown: CausalBreakdown,
0184     probability_failure: Optional[np.ndarray] = None,
0185 ) -> List[NumbersThatMatterItem]:
0186     """Return exactly six complete threshold metrics weighted by active cause share."""
0187     if index < 0 or index >= state.cell_count:
0188         raise IndexError("cell index outside PhysicsState")
0189     slope = float(_value(inputs.slope_deg, index))
0190     pga = float(_value(inputs.pga_g, index))
0191     saturation = float(state.local_saturation[index])
0192     rain = float(_value(inputs.rain_24h_mm, index))
0193     snow = float(state.snowmelt_mm_day[index])
0194     burn = float(state.effective_burn_severity[index])
0195     freeze = float(_value(inputs.freeze_thaw_cycles_7d, index))
0196     prob = float(probability_failure[index]) if probability_failure is not None else float(state.p_failure_physical[index])
0197     cause_weights = {cause: float(breakdown.combined_percentages[cause][index]) for cause in CAUSES}
0198     candidates = [
0199         _item("dynamic_factor_of_safety", state.fs_dynamic[index], "ratio", "failure threshold 1.00; rapid-deformation watch 1.25", "critical" if state.fs_dynamic[index] <= 1.0 else ("watch" if state.fs_dynamic[index] <= 1.25 else "below threshold"), "hydro_meteorological", cause_weights["hydro_meteorological"]),
0200         _item(
0201             "blended_failure_probability" if probability_failure is not None else "physical_failure_probability",
0202             prob * 100.0,
0203             "%",
0204             "hotspot threshold 65%",
0205             "critical" if prob >= 0.65 else ("watch" if prob >= 0.20 else "below threshold"),
0206             "hydro_meteorological",
0207             cause_weights["hydro_meteorological"],
0208         ),
0209         _item("pore_pressure_head", state.pore_pressure_head_m[index], "m", "full saturation head {:.3f} m".format(state.regolith_depth_m[index] * math.cos(math.radians(slope)) ** 2), "critical" if state.pore_pressure_head_m[index] >= 0.8 * state.regolith_depth_m[index] * math.cos(math.radians(slope)) ** 2 else "below threshold", "hydro_meteorological", cause_weights["hydro_meteorological"]),
0210         _item("24_hour_rainfall", rain, "mm", "25 mm when burn severity ≥0.35 activates rapid-flow branch", "critical" if burn >= 0.35 and rain >= 25.0 else "below threshold", "hydro_meteorological", cause_weights["hydro_meteorological"]),
0211         _item("local_soil_saturation", saturation * 100.0, "%", "68% rapid-flow branch threshold", "critical" if saturation >= 0.68 else "below threshold", "hydro_meteorological", cause_weights["hydro_meteorological"]),
0212         _item("pga", pga, "g", "critical acceleration {:.4f} g".format(state.critical_acceleration_g[index]), "critical" if pga > state.critical_acceleration_g[index] else "below threshold", "seismic_tectonic", cause_weights["seismic_tectonic"]),
0213         _item("newmark_displacement", state.newmark_displacement_cm[index], "cm", "2 cm coseismic velocity branch threshold", "critical" if state.newmark_displacement_cm[index] >= 2.0 else "below threshold", "seismic_tectonic", cause_weights["seismic_tectonic"]),
0214         _item("snowmelt_flux", snow, "mm/day", "positive melt requires snow depth >0.01 m and air temperature >0°C", "active" if snow > 0 else "inactive", "cryospheric", cause_weights["cryospheric"]),
0215         _item("freeze_thaw_cycles_7d", freeze, "cycles", "thaw degradation increases at ≥1 cycle", "active" if freeze >= 1 else "inactive", "cryospheric", cause_weights["cryospheric"]),
0216         _item("burn_severity", burn, "0–1", "0.35 for fire-rain rapid-flow branch", "critical" if burn >= 0.35 else "below threshold", "post_wildfire_biotic", cause_weights["post_wildfire_biotic"]),
0217         _item("root_cohesion_after_fire", state.root_cohesion_kpa[index], "kPa", "computed from NDVI and burn severity", "derived", "post_wildfire_biotic", cause_weights["post_wildfire_biotic"]),
0218         _item("toe_undercut_multiplier", state.toe_undercut_multiplier[index], "ratio", "1.0 baseline; >1 adds undercut demand", "active" if state.toe_undercut_multiplier[index] > 1.0 else "inactive", "anthropogenic_fluvial", cause_weights["anthropogenic_fluvial"]),
0219         _item("slope_angle", slope, "degrees", "16° road-cut; 28° coseismic branch thresholds", "critical" if slope >= 28 else ("watch" if slope > 16 else "below threshold"), "anthropogenic_fluvial", cause_weights["anthropogenic_fluvial"]),
0220     ]
0221     candidates.sort(key=lambda item: (item.weight_percent, item.status in ("CRITICAL", "WATCH")), reverse=True)
0222     return candidates[:6]
0223 
0224 
0225 def _normalize(values: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
0226     clean = {key: np.maximum(0.0, np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)) for key,value in values.items()}
0227     total = np.sum(np.stack(list(clean.values()), axis=0), axis=0)
0228     # A neutral prior preserves the UI contract when no stress is active; valid=False signals no attribution.
0229     return {key: np.divide(value*100.0,total,out=np.full_like(value,100.0/len(clean),dtype=float),where=total>1e-8) for key,value in clean.items()}
0230 
0231 
0232 def _value(value: object, index: int) -> float:
0233     array = np.atleast_1d(np.asarray(value, dtype=np.float64))
0234     return float(array[0] if array.size == 1 else array[index])
0235 
0236 
0237 def _item(metric: str, value: object, unit: str, threshold: str, status: str, cause: str, weight: float) -> NumbersThatMatterItem:
0238     status = {"critical":"CRITICAL", "watch":"WATCH", "active":"WATCH"}.get(status, "NORMAL")
0239     numeric = float(value)
0240     return NumbersThatMatterItem(metric, numeric if math.isfinite(numeric) else 0.0, unit, threshold, status, cause, float(weight))
```

## backend/ml/global_xgb.py

SHA-256: `4828ff5a1facff66b124782f603860090fa7f1895182011a8528796076c0a0ca`

```python
0001 """Monotonic XGBoost with spatially grouped isotonic calibration."""
0002 
0003 from __future__ import annotations
0004 
0005 from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
0006 
0007 import numpy as np
0008 
0009 from backend.realtime.device import detect_xgboost_device
0010 
0011 FEATURE_CONSTRAINTS: Tuple[Tuple[str, int], ...] = (
0012     ("fs_dynamic", -1),
0013     ("p_f_fosm", 1),
0014     ("psi_head_m", 1),
0015     ("newmark_dn_cm", 1),
0016     ("slope_max_deg", 1),
0017     ("twi", 1),
0018     ("soil_saturation", 1),
0019     ("rain_24h_mm", 1),
0020     ("snowmelt_mm_day", 1),
0021     ("pga_g", 1),
0022     ("thaw_degradation", 1),
0023     ("burn_severity", 1),
0024     ("toe_undercut_mult", 1),
0025     ("profile_curvature", 0),
0026 )
0027 FEATURE_NAMES = tuple(name for name, _ in FEATURE_CONSTRAINTS)
0028 MONOTONE_VECTOR = tuple(direction for _, direction in FEATURE_CONSTRAINTS)
0029 
0030 
0031 class TrainingDataError(ValueError):
0032     """Raised when labeled spatial training data is insufficient or malformed."""
0033 
0034 
0035 def spatial_group_splits(y: Sequence[int], groups: Sequence[Any], n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
0036     """Build deterministic grouped folds with a positive-bearing block in each holdout."""
0037     labels = np.asarray(y, dtype=np.int8)
0038     group_values = np.asarray(groups)
0039     simulated = np.asarray([str(group).startswith("synthetic:") for group in group_values])
0040     unique = np.unique(group_values[~simulated])
0041     positive_groups = [group for group in unique if np.any((group_values == group) & (labels == 1))]
0042     if n_splits < 2 or len(positive_groups) < n_splits:
0043         raise TrainingDataError("{} positive-bearing spatial groups cannot support {} holdout folds".format(len(positive_groups), n_splits))
0044     assignments: Dict[Any, int] = {}
0045     for index, group in enumerate(positive_groups):
0046         key = group.item() if hasattr(group, "item") else group
0047         assignments[key] = index % n_splits
0048     negative_only = [group for group in unique if group not in positive_groups]
0049     for index, group in enumerate(negative_only):
0050         key = group.item() if hasattr(group, "item") else group
0051         assignments[key] = index % n_splits
0052     splits = []
0053     for fold in range(n_splits):
0054         held_out = [group for group in unique if assignments[group.item() if hasattr(group, "item") else group] == fold]
0055         test = np.flatnonzero(np.isin(group_values, held_out))
0056         train = np.flatnonzero(~np.isin(group_values, held_out))
0057         splits.append((train, test))
0058     return splits
0059 
0060 
0061 class GlobalMonotonicXGB:
0062     """XGB binary classifier; construction and training require optional ML packages."""
0063 
0064     def __init__(self, n_estimators: int = 500, max_depth: int = 6, learning_rate: float = 0.04, min_child_weight: float = 1.0) -> None:
0065         self.n_estimators = n_estimators
0066         self.max_depth = max_depth
0067         self.learning_rate = learning_rate
0068         self.min_child_weight = min_child_weight
0069         self.device = detect_xgboost_device()
0070         self.model: Optional[Any] = None
0071         self.feature_names_in_ = FEATURE_NAMES
0072         self.conformal_ = None
0073 
0074     def fit(self, X: np.ndarray, y: np.ndarray, spatial_groups: Sequence[Any], n_splits: int = 5) -> "GlobalMonotonicXGB":
0075         matrix, labels = _validate_xy(X, y)
0076         groups = np.asarray(spatial_groups)
0077         if groups.ndim != 1 or groups.size != labels.size:
0078             raise TrainingDataError("spatial_groups must provide one coarse-region ID per row")
0079         unique_groups = np.unique(groups)
0080         if unique_groups.size < n_splits:
0081             raise TrainingDataError(
0082                 "{} spatial blocks are available; at least {} are required for grouped isotonic calibration".format(
0083                     unique_groups.size, n_splits
0084                 )
0085             )
0086         try:
0087             from sklearn.calibration import CalibratedClassifierCV
0088         except ImportError as exc:
0089             raise RuntimeError("Install requirements-ml.txt to train the CUDA/CPU XGBoost model") from exc
0090 
0091         splits = spatial_group_splits(labels, groups, n_splits)
0092         for train_idx, test_idx in splits:
0093             if np.unique(labels[train_idx]).size < 2 or np.unique(labels[test_idx]).size < 2:
0094                 raise TrainingDataError(
0095                     "each spatial calibration fold must contain both event and non-event labels; revise the blocks or data"
0096                 )
0097         estimator = make_base_estimator(
0098             n_estimators=self.n_estimators,
0099             max_depth=self.max_depth,
0100             learning_rate=self.learning_rate,
0101             min_child_weight=getattr(self, "min_child_weight", 1.0),
0102             device=self.device,
0103         )
0104         self.model = CalibratedClassifierCV(estimator=estimator, method="isotonic", cv=splits, ensemble=True)
0105         weights = np.asarray([0.25 if str(g).startswith("synthetic:") else 1.0 for g in groups])
0106         self.model.fit(matrix, labels, sample_weight=weights)
0107         return self
0108 
0109     def predict_proba(self, X: np.ndarray) -> np.ndarray:
0110         model = self._require_fitted()
0111         matrix = _validate_X(X)
0112         return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
0113 
0114     def calibrate_conformal(self, probabilities, labels, groups, training_groups, alpha=.10):
0115         """Split conformal with block-maximum |Y-score|; calibration never trains the model."""
0116         if not np.isfinite(alpha) or not 0 < alpha < 1:
0117             raise ValueError('Conformal alpha must be between zero and one')
0118         p=np.asarray(probabilities,float); y=np.asarray(labels,float); g=np.asarray(groups).astype(str)
0119         if p.ndim!=1 or p.shape!=y.shape or p.shape!=g.shape or not len(p):
0120             raise ValueError('Conformal vectors must be nonempty and matching')
0121         if not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not np.isin(y,[0,1]).all():
0122             raise ValueError('Conformal probabilities and binary labels are invalid')
0123         if set(g)&set(map(str,training_groups)):
0124             raise TrainingDataError('Conformal blocks overlap training/isotonic folds')
0125         scores=np.array([np.max(np.abs(y[g==block]-p[g==block])) for block in np.unique(g)])
0126         rank=int(np.ceil((len(scores)+1)*(1-alpha)))
0127         q=float(np.sort(scores)[rank-1]) if rank<=len(scores) else 1.0
0128         self.conformal_={'q':q,'alpha':alpha,'blocks':len(scores),'rows':len(y),
0129                          'target':'binary_event_label_not_true_probability',
0130                          'assumption':'exchangeable calibration/test blocks; not established for global spatial/scenario shift'}
0131         return self.conformal_
0132 
0133     def conformal_intervals(self, probabilities):
0134         p=np.asarray(probabilities,float)
0135         if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
0136             raise ValueError('Invalid conformal prediction score')
0137         calibration=getattr(self,'conformal_',None)
0138         q=calibration['q'] if calibration else 1.0
0139         lower=np.maximum(0.,p-q); upper=np.minimum(1.,p+q)
0140         width=upper-lower
0141         classes=np.char.add(np.where(p>=.5,'HIGH_SUSCEPTIBILITY_','LOW_SUSCEPTIBILITY_'),
0142                             np.where(width>=.5,'HIGH_UNCERTAINTY','LOW_UNCERTAINTY'))
0143         return np.column_stack((lower,upper)),width,classes
0144 
0145     def shap_log_odds_contributions(self, X: np.ndarray, absolute: bool = True) -> np.ndarray:
0146         """Average absolute TreeSHAP contributions from calibrated fold models.
0147 
0148         XGBoost's ``pred_contribs`` are exact TreeSHAP values in raw-margin
0149         (log-odds) space. The bias column is intentionally omitted.
0150         """
0151         model = self._require_fitted()
0152         matrix = _validate_X(X)
0153         try:
0154             import xgboost as xgb
0155         except ImportError as exc:
0156             raise RuntimeError("Install requirements-ml.txt for TreeSHAP extraction") from exc
0157         contributions = []
0158         for calibrated in getattr(model, "calibrated_classifiers_", []):
0159             estimator = getattr(calibrated, "estimator", None)
0160             if estimator is None:
0161                 estimator = getattr(calibrated, "base_estimator", None)
0162             if estimator is None or not hasattr(estimator, "get_booster"):
0163                 continue
0164             # Training uses NumPy matrices without column metadata; retain that same
0165             # positional feature contract for booster contribution predictions.
0166             dmatrix = xgb.DMatrix(matrix)
0167             raw_contrib = estimator.get_booster().predict(dmatrix, pred_contribs=True)
0168             contributions.append(np.asarray(raw_contrib[:, :-1], dtype=np.float64))
0169         if not contributions:
0170             raise RuntimeError("calibrated model does not expose XGBoost fold estimators for TreeSHAP")
0171         return np.mean(np.stack([np.abs(value) if absolute else value for value in contributions], axis=0), axis=0)
0172 
0173     def feature_importance(self) -> Dict[str, float]:
0174         model = self._require_fitted()
0175         importances = []
0176         for calibrated in getattr(model, "calibrated_classifiers_", []):
0177             estimator = getattr(calibrated, "estimator", None)
0178             if estimator is not None and hasattr(estimator, "feature_importances_"):
0179                 importances.append(np.asarray(estimator.feature_importances_, dtype=np.float64))
0180         if not importances:
0181             return {name: 0.0 for name in FEATURE_NAMES}
0182         mean = np.mean(np.stack(importances, axis=0), axis=0)
0183         return {name: float(value) for name, value in zip(FEATURE_NAMES, mean)}
0184 
0185     def _require_fitted(self) -> Any:
0186         if self.model is None:
0187             raise RuntimeError("GlobalMonotonicXGB has no fitted model; labeled training data is required")
0188         return self.model
0189 
0190 
0191 def mechanical_stability_gate(slope_deg: np.ndarray, fs_dynamic: np.ndarray) -> np.ndarray:
0192     """User-specified smooth mechanical screen; not a calibrated probability."""
0193     slope, fs = np.broadcast_arrays(np.asarray(slope_deg, dtype=float), np.asarray(fs_dynamic, dtype=float))
0194     if not np.isfinite(slope).all() or not np.isfinite(fs).all():
0195         raise ValueError("slope and factor of safety must be finite")
0196     return (1.0 / (1.0 + np.exp(np.clip(-(slope - 8.5) / 0.65, -700, 700)))) * (1.0 / (1.0 + np.exp(np.clip((fs - 2.15) / 0.18, -700, 700))))
0197 
0198 
0199 def blend_failure_probability(p_xgb: np.ndarray, p_physical: np.ndarray,
0200                               slope_deg: np.ndarray, fs_dynamic: np.ndarray) -> np.ndarray:
0201     ml, physical, slope, fs = np.broadcast_arrays(
0202         np.asarray(p_xgb, dtype=float), np.asarray(p_physical, dtype=float),
0203         np.asarray(slope_deg, dtype=float), np.asarray(fs_dynamic, dtype=float))
0204     if not np.isfinite(ml).all() or not np.isfinite(physical).all():
0205         raise ValueError("probabilities must be finite")
0206     weight = .35 + .50 / (1 + np.exp(np.clip((fs-1.05)/.06,-700,700)))
0207     return mechanical_stability_gate(slope, fs) * np.clip((1-weight)*ml+weight*physical,0.,1.)
0208 
0209 
0210 def make_base_estimator(
0211     n_estimators: int = 500,
0212     max_depth: int = 6,
0213     learning_rate: float = 0.04,
0214     device: Optional[str] = None,
0215     min_child_weight: float = 1.0,
0216 ) -> Any:
0217     """Construct the constrained XGB estimator, preferring CUDA when available."""
0218     try:
0219         from xgboost import XGBClassifier
0220     except ImportError as exc:
0221         raise RuntimeError("Install requirements-ml.txt for XGBoost") from exc
0222     selected_device = device or detect_xgboost_device()
0223     return XGBClassifier(
0224         n_estimators=n_estimators,
0225         max_depth=max_depth,
0226         learning_rate=learning_rate,
0227         min_child_weight=min_child_weight,
0228         objective="binary:logistic",
0229         eval_metric="logloss",
0230         tree_method="hist",
0231         device=selected_device,
0232         monotone_constraints=MONOTONE_VECTOR,
0233         n_jobs=-1,
0234     )
0235 
0236 
0237 def feature_matrix(features: Mapping[str, Any]) -> np.ndarray:
0238     missing = [name for name in FEATURE_NAMES if name not in features]
0239     if missing:
0240         raise ValueError("missing model features: {}".format(", ".join(missing)))
0241     arrays = [np.atleast_1d(np.asarray(features[name], dtype=np.float64)) for name in FEATURE_NAMES]
0242     matrix = np.column_stack(np.broadcast_arrays(*arrays))
0243     return _validate_X(matrix)
0244 
0245 
0246 def _validate_xy(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
0247     matrix = _validate_X(X)
0248     labels = np.asarray(y, dtype=np.int8)
0249     if labels.ndim != 1 or labels.size != matrix.shape[0]:
0250         raise TrainingDataError("labels must be a one-dimensional vector matching X rows")
0251     if not np.isin(labels, (0, 1)).all() or np.unique(labels).size != 2:
0252         raise TrainingDataError("binary training labels must contain both 0 (no failure) and 1 (failure)")
0253     return matrix, labels
0254 
0255 
0256 def _validate_X(X: np.ndarray) -> np.ndarray:
0257     matrix = np.asarray(X, dtype=np.float32)
0258     if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
0259         raise ValueError("X must have shape (n_cells, 14) in FEATURE_NAMES order")
0260     if not np.isfinite(matrix).all():
0261         raise ValueError("X contains missing or non-finite values; imputation must be explicit and trained")
0262     return matrix
```

## backend/ml/conformal_calibration.py

SHA-256: `fbadc5d8402106ac7da5d8df122fe8043f735848ae797fcdc8c0837c10cc9a28`

```python
0001 """Fit uncertainty on separately reserved, unfiltered spatial blocks."""
0002 import numpy as np
0003 from .global_xgb import FEATURE_NAMES, blend_failure_probability
0004 
0005 def fit_reserved_conformal(model, data_directory, training_groups):
0006     import pyarrow.parquet as pq
0007     path=data_directory/'conformal_calibration.parquet'
0008     rows=pq.read_table(path).to_pylist()
0009     x=np.array([[r[k] for k in FEATURE_NAMES] for r in rows],dtype=np.float32)
0010     p=blend_failure_probability(model.predict_proba(x),np.array([r['p_failure_physical'] for r in rows]),x[:,4],x[:,0])
0011     return model.calibrate_conformal(p,[r['label'] for r in rows],[r['region_block'] for r in rows],training_groups)
0012 
0013 def heldout_conformal_diagnostics(model, rows):
0014     """Descriptive coverage on untouched holdout labels, never used to choose q."""
0015     x=np.array([[r[k] for k in FEATURE_NAMES] for r in rows],dtype=np.float32)
0016     p=blend_failure_probability(model.predict_proba(x),np.array([r['p_failure_physical'] for r in rows]),x[:,4],x[:,0])
0017     bounds,width,_=model.conformal_intervals(p)
0018     y=np.array([r['label'] for r in rows]);g=np.array([r['region_block'] for r in rows])
0019     covered=(bounds[:,0]<=y)&(y<=bounds[:,1])
0020     return {'target':'binary event labels, not the latent probability',
0021             'rows':len(rows),'blocks':len(set(g)),
0022             'row_coverage':float(np.mean(covered)),
0023             'whole_block_coverage':float(np.mean([np.all(covered[g==v]) for v in set(g)])),
0024             'mean_interval_width':float(np.mean(width)),
0025             'q':model.conformal_['q'],
0026             'empty_binary_prediction_sets':int(np.sum((bounds[:,0]>0)&(bounds[:,1]<1))),
0027             'scope':'Descriptive on fixed source-location proxies; not a global coverage guarantee'}
```

## backend/physics/cascading.py

SHA-256: `c4614ae442243200f17927ce59c31d7a0942f7e2ef058c14755ae368479c0786`

```python
0001 """DBI in consistent SI units; unresolved dam geometry never receives invented values."""
0002 import math
0003 import h3
0004 
0005 def larsen_volume(area_m2):
0006     if not math.isfinite(area_m2) or area_m2<=0:raise ValueError('Positive mapped/scenario area required')
0007     # Larsen et al. 2010, Fig. 2a: log10(alpha)=-.836, gamma=1.332, all materials.
0008     return 10.**(-.836) * area_m2**1.332
0009 
0010 def dimensionless_blockage_index(catchment_area_m2, dam_height_m, deposited_volume_m3):
0011     values=(catchment_area_m2,dam_height_m,deposited_volume_m3)
0012     if any(not math.isfinite(v) or v<=0 for v in values):raise ValueError('Positive finite SI geometry required')
0013     return math.log10(catchment_area_m2*dam_height_m/deposited_volume_m3)
0014 
0015 def cascading_screen(source_cell, route_path, terrain_cells, dam_geometry=None):
0016     channel=next((h for h in route_path[1:] if terrain_cells[h].twi>=14.5),None)
0017     result={'cascading_dbi':None,'impoundment_status':'NO_MODELED_CHANNEL_INTERSECTION',
0018             'channel_h3_index':channel,'dam_height_m':None,'dam_volume_m3':None,'catchment_area_m2':None}
0019     if channel is None:return result
0020     area=h3.cell_area(source_cell,unit='m^2')
0021     result.update(impoundment_status='POTENTIAL_CHANNEL_INTERSECTION_GEOMETRY_UNRESOLVED',
0022                   source_area_scenario_m2=area,landslide_volume_scenario_m3=larsen_volume(area),
0023                   volume_quality='Full H3 source-cell failure scenario; not mapped scar or deposited dam volume',
0024                   catchment_area_in_view_m2=float(terrain_cells[channel].contributing_area_m2),
0025                   catchment_quality='Viewport DEM accumulation; full upstream catchment not established')
0026     if dam_geometry is None:return result
0027     a=dam_geometry['catchment_area_m2'];h=dam_geometry['dam_height_m'];v=dam_geometry['deposited_volume_m3']
0028     dbi=dimensionless_blockage_index(a,h,v)
0029     result.update(cascading_dbi=dbi,dam_height_m=h,dam_volume_m3=v,catchment_area_m2=a,
0030                   impoundment_status=('EMPIRICAL_STABLE_DAM_DOMAIN' if dbi<2.75 else
0031                                       'EMPIRICAL_UNSTABLE_DAM_DOMAIN' if dbi>3.08 else 'EMPIRICAL_TRANSITION_DOMAIN'),
0032                   geometry_source=dam_geometry.get('source','explicit scenario geometry'),
0033                   dbi_quality='Ermini-Casagli empirical domain; not evidence that a dam has formed')
0034     return result
```

## backend/llm/qwen_agent.py

SHA-256: `d5f33049c3dc845d778a22c232d179a2d0d1ee861ce56c1391d36d84c0f8060f`

```python
0001 """Three-tier Ollama tool agent with a deterministic live-scan fallback."""
0002 
0003 from __future__ import annotations
0004 
0005 import json
0006 import math
0007 import re
0008 import time
0009 from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
0010 
0011 import httpx
0012 
0013 from backend.api.scan_service import GlobalScanService
0014 
0015 
0016 TOOL_SCHEMAS: List[Dict[str, Any]] = [
0017     {
0018         "type": "function",
0019         "function": {
0020             "name": "scan_global_location",
0021             "description": "Run a live global landslide scan with the requested scenario perturbations.",
0022             "parameters": {
0023                 "type": "object",
0024                 "properties": {
0025                     "location_query": {"type": "string"},
0026                     "rain_delta_mm": {"type": "number", "default": 0.0},
0027                     "storm_duration_hr": {"type": "number", "default": 24.0},
0028                     "pga_override_g": {"type": "number", "default": 0.0},
0029                     "snowmelt_temp_surge_c": {"type": "number", "default": 0.0},
0030                     "burn_severity_override": {"type": "number", "default": 0.0},
0031                 },
0032                 "required": ["location_query"],
0033             },
0034         },
0035     },
0036     {
0037         "type": "function",
0038         "function": {
0039             "name": "inspect_hexagon_causal_physics",
0040             "description": "Inspect the active scan's physics and causal breakdown for one H3 cell.",
0041             "parameters": {"type": "object", "properties": {"h3_index": {"type": "string"}}, "required": ["h3_index"]},
0042         },
0043     },
0044     {
0045         "type": "function",
0046         "function": {
0047             "name": "trace_runout_and_evacuation",
0048             "description": "Trace a provisional Voellmy runout path and list mapped exposure intersections.",
0049             "parameters": {"type": "object", "properties": {"h3_index": {"type": "string"}}, "required": ["h3_index"]},
0050         },
0051     },
0052 ]
0053 
0054 _NUMBER = r"[-+]?\d+(?:\.\d+)?"
0055 _NUMERIC_TOKEN = re.compile(r"(?<![\w])" + _NUMBER + r"(?:[eE][-+]?\d+)?")
0056 _XML_TAG = re.compile(r"<\s*/?\s*(?:ui_action|tool_call)\b[^>]*>", re.IGNORECASE)
0057 _ANY_TAG = re.compile(r"<\s*/?[A-Za-z][^>]*>")
0058 
0059 
0060 def parse_scenario_request(
0061     message: str, current_query: str = "", sliders: Optional[Mapping[str, Any]] = None
0062 ) -> Dict[str, Any]:
0063     """Extract only explicit perturbations; Mw is never converted into PGA."""
0064     text = str(message or "")
0065     lowered = text.lower()
0066     values: Dict[str, Any] = {
0067         "rain_delta_mm": 0.0,
0068         "storm_duration_hr": 24.0,
0069         "pga_override_g": 0.0,
0070         "snowmelt_temp_surge_c": 0.0,
0071         "burn_severity_override": 0.0,
0072     }
0073     explicit = set()
0074     rain_matches = re.findall("(" + _NUMBER + r")\s*mm\b", lowered)
0075     if rain_matches:
0076         values["rain_delta_mm"] = max(0.0, float(rain_matches[-1]))
0077         explicit.add("rain_delta_mm")
0078     pga_matches = re.findall("(" + _NUMBER + r")\s*g\b", lowered)
0079     if pga_matches:
0080         values["pga_override_g"] = max(0.0, float(pga_matches[-1]))
0081         explicit.add("pga_override_g")
0082     temp_matches = re.findall("(" + _NUMBER + r")\s*(?:°\s*c|degrees?\s*celsius|celsius)\b", lowered)
0083     if temp_matches:
0084         values["snowmelt_temp_surge_c"] = max(0.0, float(temp_matches[-1]))
0085         explicit.add("snowmelt_temp_surge_c")
0086     burn_matches = re.findall("(" + _NUMBER + r")\s*%", lowered)
0087     if burn_matches:
0088         values["burn_severity_override"] = min(1.0, max(0.0, float(burn_matches[-1]) / 100.0))
0089         explicit.add("burn_severity_override")
0090     duration_matches = re.findall("(" + _NUMBER + r")\s*(?:h|hr|hrs|hours?)\b", lowered)
0091     if duration_matches:
0092         values["storm_duration_hr"] = max(0.01, float(duration_matches[-1]))
0093         explicit.add("storm_duration_hr")
0094 
0095     for key in values:
0096         if sliders and key in sliders:
0097             try:
0098                 slider_value = float(sliders[key])
0099                 if not math.isfinite(slider_value):
0100                     raise ValueError("slider must be finite")
0101                 if key == "burn_severity_override":
0102                     slider_value = min(1.0, max(0.0, slider_value))
0103                 elif key == "storm_duration_hr":
0104                     slider_value = max(0.01, slider_value)
0105                 else:
0106                     slider_value = max(0.0, slider_value)
0107                 if not math.isfinite(slider_value):
0108                     raise ValueError("slider must be finite")
0109                 if key not in explicit:
0110                     values[key] = slider_value
0111                     explicit.add(key)
0112             except (TypeError, ValueError):
0113                 continue
0114 
0115     query = str(current_query or "").strip()
0116     from backend.api.scan_service import known_location_coordinates
0117     candidate = _location_from_message(text)
0118     if candidate and (not query or known_location_coordinates(candidate) is not None or re.search(r"\bscan\b",text,re.I)):
0119         query = candidate
0120     values["location_query"] = query or "Chooralmala, Wayanad"
0121     values["magnitude_only"] = bool(re.search(r"\bM\s*w\s*" + _NUMBER, text, re.IGNORECASE)) and not pga_matches
0122     values["_explicit_fields"] = sorted(explicit)
0123     return values
0124 
0125 
0126 def _location_from_message(message: str) -> str:
0127     text = message.strip()
0128     text = re.sub(r"\bM\s*w\s*" + _NUMBER, " ", text, flags=re.IGNORECASE)
0129     text = re.sub(_NUMBER + r"\s*(?:mm\b|g\b|°\s*c\b|degrees?\s*celsius\b|celsius\b|%|M\s*w\s*\d+(?:\.\d+)?)", " ", text, flags=re.IGNORECASE)
0130     text = re.sub(r"\b(?:scan|check|assess|estimate|show|tell\s+me|landslide|risk|hazard|with|under|after|for|at|in|around|near|the|and|burn|scar|rain|rainfall|earthquake|snowmelt|warming|temperature|please)\b", " ", text, flags=re.IGNORECASE)
0131     text = re.sub(r"[+,:;()\[\]{}]", " ", text)
0132     text = re.sub(r"\s+", " ", text).strip(" .-/")
0133     return text
0134 
0135 
0136 class TerraShieldQwenAgent:
0137     """Select a local Qwen model when available and ground every briefing in tools."""
0138 
0139     def __init__(self, scan_service: Optional[GlobalScanService] = None, ollama_url: Optional[str] = None) -> None:
0140         self.scan_service = scan_service or GlobalScanService()
0141         self.ollama_url = (ollama_url or "http://localhost:11434").rstrip("/")
0142 
0143     async def scan_global_location(
0144         self,
0145         location_query: str,
0146         rain_delta_mm: float = 0.0,
0147         storm_duration_hr: float = 24.0,
0148         pga_override_g: float = 0.0,
0149         snowmelt_temp_surge_c: float = 0.0,
0150         burn_severity_override: float = 0.0,
0151     ) -> Dict[str, Any]:
0152         return await self.scan_service.scan(
0153             location_query=location_query,
0154             rain_delta_mm=rain_delta_mm,
0155             storm_duration_hr=storm_duration_hr,
0156             pga_override_g=pga_override_g,
0157             snowmelt_temp_surge_c=snowmelt_temp_surge_c,
0158             burn_severity_override=burn_severity_override,
0159         )
0160 
0161     async def _run_scan(self, arguments: Mapping[str, Any], agency_mode: str) -> Dict[str, Any]:
0162         return await self.scan_service.scan(**dict(arguments), agency_mode=agency_mode)
0163 
0164     def inspect_hexagon_causal_physics(self, h3_index: str) -> Dict[str, Any]:
0165         return self.scan_service.inspect_hexagon(h3_index)
0166 
0167     def trace_runout_and_evacuation(self, h3_index: str) -> Dict[str, Any]:
0168         return self.scan_service.trace_runout(h3_index)
0169 
0170     async def chat(
0171         self,
0172         message: str,
0173         current_query: str = "",
0174         agency_mode: str = "LOCAL_EOC",
0175         sliders: Optional[Mapping[str, Any]] = None,
0176     ) -> Dict[str, Any]:
0177         parsed = parse_scenario_request(message, current_query, sliders)
0178         model_name, tier = await self._available_model()
0179         tool_calls: List[Dict[str, Any]] = []
0180         scan_data: Optional[Dict[str, Any]] = None
0181         if model_name:
0182             try:
0183                 scan_data = await self._ollama_tool_roundtrip(
0184                     model_name, message, parsed, agency_mode, tool_calls
0185                 )
0186             except Exception:
0187                 # If Ollama disconnects mid-conversation, continue with live deterministic tools.
0188                 model_name, tier = None, 3
0189         if scan_data is None:
0190             args = self._scan_arguments(parsed)
0191             started = time.perf_counter()
0192             scan_data = await self._run_scan(args, agency_mode)
0193             tool_calls.append({
0194                 "tool_name": "scan_global_location",
0195                 "arguments": args,
0196                 "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
0197             })
0198 
0199         clean_markdown = self._briefing(scan_data, agency_mode, bool(parsed.get("magnitude_only")))
0200         clean_markdown = _XML_TAG.sub("", clean_markdown)
0201         clean_markdown = _ANY_TAG.sub("", clean_markdown).strip()
0202         clean_markdown, numeric_audit_passed, briefing_guard = self._guard_briefing(clean_markdown, scan_data, agency_mode, bool(parsed.get("magnitude_only")))
0203         location = scan_data.get("location") or {}
0204         ui_action = {
0205             "fly_to": [location.get("longitude"), location.get("latitude")],
0206             "zoom": 13.4,
0207             "sliders": {key: parsed[key] for key in (
0208                 "rain_delta_mm", "storm_duration_hr", "pga_override_g",
0209                 "snowmelt_temp_surge_c", "burn_severity_override",
0210             )},
0211             "agency_mode": agency_mode,
0212         }
0213         return {
0214             "clean_markdown": clean_markdown,
0215             "tool_calls": tool_calls,
0216             "ui_action": ui_action,
0217             "updated_scan_data": scan_data,
0218             "numeric_audit_passed": bool(numeric_audit_passed),
0219             "briefing_guard": briefing_guard,
0220             "model_tier": tier,
0221             "model_name": model_name,
0222         }
0223 
0224     async def _available_model(self) -> Tuple[Optional[str], int]:
0225         try:
0226             async with httpx.AsyncClient(timeout=1.25) as client:
0227                 response = await client.get(self.ollama_url + "/api/tags")
0228                 response.raise_for_status()
0229                 names = [str(item.get("name", "")) for item in response.json().get("models", [])]
0230             custom = next((name for name in names if name.split(":", 1)[0].lower() == "terra-shield-qwen"), None)
0231             if custom:
0232                 return custom, 1
0233             qwen = next((name for name in names if "qwen2.5" in name.lower()), None)
0234             if qwen:
0235                 return qwen, 2
0236         except Exception:
0237             pass
0238         return None, 3
0239 
0240     async def _ollama_tool_roundtrip(
0241         self,
0242         model_name: str,
0243         message: str,
0244         parsed: Mapping[str, Any],
0245         agency_mode: str,
0246         tool_calls: List[Dict[str, Any]],
0247     ) -> Optional[Dict[str, Any]]:
0248         messages: List[Dict[str, Any]] = [
0249             {"role": "system", "content": "Use the supplied live tools for all measurements. Do not invent coordinates, probabilities, causes, or runout. Call scan_global_location for the requested location and perturbations. Then call inspect/trace only for an H3 index returned by the scan. Final numbers are generated by the application from tool telemetry."},
0250             {"role": "user", "content": message or "Scan the current location."},
0251         ]
0252         scan_data: Optional[Dict[str, Any]] = None
0253         async with httpx.AsyncClient(timeout=45.0) as client:
0254             for _ in range(4):
0255                 response = await client.post(self.ollama_url + "/api/chat", json={
0256                     "model": model_name, "messages": messages, "tools": TOOL_SCHEMAS, "stream": False,
0257                 })
0258                 response.raise_for_status()
0259                 assistant_message = response.json().get("message") or {}
0260                 messages.append(assistant_message)
0261                 calls = assistant_message.get("tool_calls") or []
0262                 if not calls:
0263                     break
0264                 for call in calls:
0265                     function = call.get("function") or {}
0266                     name = str(function.get("name") or "")
0267                     arguments = function.get("arguments") or {}
0268                     if isinstance(arguments, str):
0269                         try:
0270                             arguments = json.loads(arguments)
0271                         except ValueError:
0272                             arguments = {}
0273                     arguments = dict(arguments) if isinstance(arguments, Mapping) else {}
0274                     if name == "scan_global_location":
0275                         # All scenario values come from explicit user text/sliders or deterministic defaults.
0276                         # A language model cannot silently add a perturbation that the UI does not show.
0277                         arguments = self._scan_arguments(parsed)
0278                     elif name in ("inspect_hexagon_causal_physics", "trace_runout_and_evacuation") and scan_data is None:
0279                         arguments_scan = self._scan_arguments(parsed)
0280                         started = time.perf_counter()
0281                         scan_data = await self._run_scan(arguments_scan, agency_mode)
0282                         tool_calls.append({"tool_name": "scan_global_location", "arguments": arguments_scan, "latency_ms": round((time.perf_counter() - started) * 1000.0, 2)})
0283                     started = time.perf_counter()
0284                     try:
0285                         result = await self._dispatch(name, arguments, agency_mode)
0286                         if name == "scan_global_location":
0287                             scan_data = result
0288                         tool_result = result
0289                     except Exception as exc:
0290                         tool_result = {"error": "{}: {}".format(type(exc).__name__, exc)}
0291                     tool_calls.append({"tool_name": name, "arguments": arguments, "latency_ms": round((time.perf_counter() - started) * 1000.0, 2)})
0292                     messages.append({"role": "tool", "name": name, "content": json.dumps(tool_result, ensure_ascii=False, default=str, allow_nan=False)})
0293         return scan_data
0294 
0295     async def _dispatch(self, name: str, arguments: Mapping[str, Any], agency_mode: str) -> Dict[str, Any]:
0296         if name == "scan_global_location":
0297             return await self._run_scan(self._clean_scan_arguments(arguments), agency_mode)
0298         if name == "inspect_hexagon_causal_physics":
0299             return self.inspect_hexagon_causal_physics(str(arguments.get("h3_index", "")))
0300         if name == "trace_runout_and_evacuation":
0301             return self.trace_runout_and_evacuation(str(arguments.get("h3_index", "")))
0302         raise ValueError("unknown tool: {}".format(name))
0303 
0304     @staticmethod
0305     def _clean_scan_arguments(arguments: Mapping[str, Any]) -> Dict[str, Any]:
0306         allowed = {
0307             "location_query", "rain_delta_mm", "storm_duration_hr", "pga_override_g",
0308             "snowmelt_temp_surge_c", "burn_severity_override",
0309         }
0310         return {key: value for key, value in arguments.items() if key in allowed}
0311 
0312     @staticmethod
0313     def _scan_arguments(parsed: Mapping[str, Any]) -> Dict[str, Any]:
0314         return {
0315             "location_query": parsed.get("location_query") or "Chooralmala, Wayanad",
0316             "rain_delta_mm": float(parsed.get("rain_delta_mm", 0.0)),
0317             "storm_duration_hr": float(parsed.get("storm_duration_hr", 24.0)),
0318             "pga_override_g": float(parsed.get("pga_override_g", 0.0)),
0319             "snowmelt_temp_surge_c": float(parsed.get("snowmelt_temp_surge_c", 0.0)),
0320             "burn_severity_override": float(parsed.get("burn_severity_override", 0.0)),
0321         }
0322 
0323     @staticmethod
0324     def _briefing(scan: Mapping[str, Any], agency_mode: str, magnitude_only: bool = False) -> str:
0325         location = scan.get("location") or {}
0326         summary = scan.get("summary") or {}
0327         metrics = summary.get("meteorological_metrics") or {}
0328         hotspot = summary.get("kinematics_top_hotspot") or {}
0329         name = location.get("display_name") or location.get("query") or "Requested location"
0330         lines = ["## TERRA-SHIELD scan: {}".format(name), "", "- **Alert level:** {}".format(summary.get("wmo_alert_level", "UNAVAILABLE"))]
0331         if agency_mode.upper() == "WMO_MET":
0332             lines.extend([
0333                 "- **Daily rainfall scenario:** {} mm; storm duration: {} h".format(
0334                     TerraShieldQwenAgent._fmt(metrics.get("rain_24h_mm_scenario")),
0335                     TerraShieldQwenAgent._fmt(metrics.get("storm_duration_hr")),
0336                 ),
0337                 "- **Scenario PGA:** {} g; air temperature: {} °C; burn severity: {}".format(
0338                     TerraShieldQwenAgent._fmt(metrics.get("pga_g_scenario")),
0339                     TerraShieldQwenAgent._fmt(metrics.get("air_temperature_c_scenario")),
0340                     TerraShieldQwenAgent._fmt(metrics.get("burn_severity_override")),
0341                 ),
0342                 "- **P95 failure probability:** {}; **P05 dynamic factor of safety:** {}".format(
0343                     TerraShieldQwenAgent._fmt(summary.get("max_p_failure")),
0344                     TerraShieldQwenAgent._fmt(summary.get("min_fs_dynamic")),
0345                 ),
0346                 "- **Critical hotspots:** {}".format(summary.get("critical_hotspots_count", 0)),
0347             ])
0348         else:
0349             lines.extend([
0350                 "- **P95 failure probability:** {}; **P05 dynamic factor of safety:** {}".format(
0351                     TerraShieldQwenAgent._fmt(summary.get("max_p_failure")),
0352                     TerraShieldQwenAgent._fmt(summary.get("min_fs_dynamic")),
0353                 ),
0354                 "- **Top exposed hotspot:** H3 `{}`; slope {}°; peak velocity {} m/s; {}.".format(
0355                     hotspot.get("h3_index", "unavailable"),
0356                     TerraShieldQwenAgent._fmt(hotspot.get("slope_deg")),
0357                     TerraShieldQwenAgent._fmt(hotspot.get("peak_velocity_ms")),
0358                     hotspot.get("failure_mode", "mode unavailable"),
0359                 ),
0360                 "- **Hotspot cause:** {}".format(hotspot.get("dominant_cause", summary.get("dominant_cause", "unavailable"))),
0361                 "- **Critical hotspots:** {}; mapped households exposed: {}".format(
0362                     summary.get("critical_hotspots_count", 0),
0363                     summary.get("total_households_exposed") if summary.get("total_households_exposed") is not None else "unavailable",
0364                 ),
0365             ])
0366         cause = hotspot.get("causal_breakdown_pct") or summary.get("causal_breakdown_pct") or {}
0367         if cause:
0368             top_causes = sorted(cause.items(), key=lambda item: float(item[1]), reverse=True)[:3]
0369             lines.append("- **Causal breakdown:** {}".format(", ".join("{} {}%".format(key, TerraShieldQwenAgent._fmt(value, 1)) for key, value in top_causes)))
0370         numbers = hotspot.get("numbers_that_matter") or summary.get("numbers_that_matter") or []
0371         if numbers:
0372             lines.append("- **Numbers that matter:** {}".format("; ".join(
0373                 "{} {} {}".format(item.get("metric"), TerraShieldQwenAgent._fmt(item.get("value")), item.get("unit", ""))
0374                 for item in numbers[:3]
0375             )))
0376         lines.extend([
0377             "",
0378             "Probability is an exploratory model/physics blend when a model is available; source gaps and fallback inputs are listed in the scan payload.",
0379             "Runout routes use display defaults that are not calibrated for operational evacuation decisions.",
0380         ])
0381         if magnitude_only:
0382             lines.append("Mw was noted, but no site PGA was estimated because magnitude alone does not specify shaking at the location.")
0383         return "\n".join(lines)
0384 
0385     @staticmethod
0386     def _fmt(value: Any, digits: int = 3) -> str:
0387         if value is None:
0388             return "unavailable"
0389         try:
0390             return format(float(value), ".{}f".format(digits))
0391         except (TypeError, ValueError):
0392             return str(value)
0393 
0394     @staticmethod
0395     def _guard_briefing(candidate, telemetry, agency_mode, magnitude_only=False):
0396         # Model prose never reaches the API: the approved output is a tool-bound template.
0397         canonical=TerraShieldQwenAgent._briefing(telemetry,agency_mode,magnitude_only)
0398         canonical=_ANY_TAG.sub('',_XML_TAG.sub('',canonical)).strip()
0399         valid=TerraShieldQwenAgent._numeric_parity_audit(canonical,telemetry)
0400         if not valid:
0401             return ('Verified numeric briefing unavailable. Inspect structured telemetry and source status.',
0402                     True, 'WITHHELD_INVALID_TEMPLATE')
0403         accepted=candidate==canonical and TerraShieldQwenAgent._numeric_parity_audit(candidate,telemetry)
0404         return canonical,True,'VERIFIED_TEMPLATE' if accepted else 'REPLACED_UNVERIFIED_PROSE'
0405 
0406     @staticmethod
0407     def _numeric_parity_audit(markdown: str, telemetry: Mapping[str, Any]) -> bool:
0408         available=[];strings=[]
0409         def collect(value):
0410             if isinstance(value,bool) or value is None:return
0411             if isinstance(value,(int,float)) and math.isfinite(float(value)):available.append(float(value))
0412             elif isinstance(value,str) and value:strings.append(value)
0413             elif isinstance(value,Mapping):
0414                 for child in value.values():collect(child)
0415             elif isinstance(value,(list,tuple)):
0416                 for child in value:collect(child)
0417         collect(telemetry)
0418         # Verbatim identifiers/location strings are not numeric measurements.
0419         for value in sorted(strings,key=len,reverse=True):
0420             if any(c.isdigit() for c in value):markdown=markdown.replace(value,'SOURCE_IDENTIFIER')
0421         for token in _NUMERIC_TOKEN.findall(markdown):
0422             precision=len(token.partition('.')[2]) if '.' in token else 0
0423             if not any(float(token)==float(format(value,'.{}f'.format(precision))) for value in available):return False
0424         return True
```
