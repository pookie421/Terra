# v5 boundary and literature audit

| Requested target | Result |
|---|---|
| blue_mountains_no_warming_created_cryo | PASS |
| blue_mountains_critical_fs_gt_1 | PASS |
| stage2_original_auc_gte_0_82 | NOT MET |
| stage2_pinls_auc_gte_0_82 | NOT MET |
| wayanad_score_gte_0_80 | NOT MET |
| all_cold_scans_lt_3_5_seconds | NOT MET |
| all_cold_scans_lt_2_5_seconds | NOT MET |
| zero_unverified_constitutive_terms | NOT MET |
| observed_geometry_for_live_dbi | NOT MET |

| Cold scan | Seconds |
|---|---:|
| Blue Mountains | 8.324551 |
| Wayanad | 12.76275375 |
| New Delhi | 9.348349625 |

- Supplied Milledge DOI is Pudasaini 2012. Implemented lateral term is an adaptation of finite-block reasoning, not the full Milledge 2014 model.
- FS prior >1.20 is the requested negative selection rule; the cited Do 2026 primary full text was not verified.
- Live ice weakening and post-fire yield-index transformation remain explicitly marked scenario proxies, not Mamot or Staley constitutive equations.
- RGB vegetation proxy is not measured pre-fire NDVI. Snow floors, width 35 m and hollow factor are scenario assumptions.
- Conformal intervals predict binary labels under exchangeable blocks; they are not confidence intervals on true failure probability.
- Dam height, deposited dam volume and full upstream catchment are unavailable. DBI is null rather than fabricated.
