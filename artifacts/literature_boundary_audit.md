# v6 boundary and literature audit

| Requested target | Result |
|---|---|
| blue_mountains_no_warming_created_cryo | PASS |
| blue_mountains_critical_fs_gt_1 | PASS |
| stage2_original_auc_gte_0_82 | NOT MET |
| stage2_pinls_auc_gte_0_82 | NOT MET |
| wayanad_score_gte_0_80 | PASS |
| all_cold_scans_lt_3_5_seconds | PASS |
| all_cold_scans_lt_2_5_seconds | NOT MET |
| zero_unverified_constitutive_terms | NOT MET |
| observed_geometry_for_live_dbi | NOT MET |

| Cold scan | Seconds |
|---|---:|
| Blue Mountains | 3.271876541 |
| Wayanad | 3.0758632919999993 |
| New Delhi | 3.0618152500000004 |

- Unverified lateral resistance and probability-derived FS terms have been removed; mechanical FS is Mohr-Coulomb.
- FS prior >1.20 is the requested negative selection rule; the cited Do 2026 primary full text was not verified.
- Mamot strength law is used within its calibrated temperature range; the soil-to-joint temperature mapping is an unvalidated proxy. M1 probability remains separate from mechanical FS.
- RGB vegetation is not pre-fire NDVI. Snow floors, hollow factor, and mapped-geology strength adjustments are scenario assumptions. River discharge is modeled, not gauged.
- Ensemble-scaled conformal intervals target binary labels and assume cell exchangeability; spatial dependence prevents a nominal coverage guarantee. They are not true-probability confidence intervals.
- Dam height, deposited dam volume and full upstream catchment are unavailable. DBI is null rather than fabricated.
