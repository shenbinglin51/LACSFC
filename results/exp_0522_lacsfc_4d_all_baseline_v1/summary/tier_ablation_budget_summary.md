# LACSFC Tier Ablation + Budget-Amortization Summary

This table is designed to test the progression from Tier1 heuristic to Tier2 optimization and Tier3 meta amortization.

## Aggregate by workload and mode

| workload | mode | tier | budget_evals | curve | n | query_s_mean | query_s_std | intervals_s_mean | avg_amp_mean | avg_seg_mean | build_s_mean | query_s_delta_vs_hilbert_pct | query_s_gain_vs_lacsfc_h_pct | query_s_gain_vs_same_budget_opt_pct | query_s_gap_to_opt_fast_pct | meta_loaded_rate | fallback_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| geonames_time_cat_4d | adacurve | Unknown |  | AdaCurve | 3 | 0.0208333 | 0.00238374 | 0.00243333 | 13.7474 | 1.2292 | 0.00336667 |  | -85.4599 |  |  | 0 | 0 |
| geonames_time_cat_4d | bitallocz_bo | Unknown |  | BitAllocZ-BO | 3 | 0.0152667 | 0.000449691 | 0.0066 | 2.3047 | 7.13543 | 0.0178 |  | -35.905 |  |  | 0 | 0 |
| geonames_time_cat_4d | hilbert_nd | Unknown |  | Hilbert | 3 | 0.0151 | 0.000668331 | 0.0065 | 2.28127 | 6.375 | 0.0162 |  | -34.4214 |  |  | 0 | 0 |
| geonames_time_cat_4d | lacsfc_h | Tier1 | 0 | LACSFC-H | 3 | 0.0112333 | 0.000478423 | 0.0021 | 2.49567 | 5.5 | 0.0067 |  | 0 |  |  | 0 | 0 |
| geonames_time_cat_4d | lacsfc_meta_refine_tiny | Unknown |  | LACSFC-Meta | 3 | 0.0175333 | 0.00881564 | 0.00306667 | 2.7014 | 5.47917 | 0.0100667 |  | -56.0831 |  |  | 1 | 0 |
| geonames_time_cat_4d | lacsfc_opt_mid | Unknown |  | LACSFC-Opt | 3 | 0.0107667 | 0.000402768 | 0.00196667 | 2.76477 | 5.54167 | 0.00663333 |  | 4.1543 |  |  | 0 | 0 |
| geonames_time_cat_4d | lmsfc_bo | Unknown |  | LMSFC-BO | 3 | 0.0108667 | 0.00063421 | 0.0036 | 2.4045 | 1.53127 | 0.0320667 |  | 3.26409 |  |  | 0 | 0 |
| geonames_time_cat_4d | piecewise | Unknown |  | PiecewiseSFC | 3 | 0.015 | 0.000697615 | 0.00616667 | 3.57117 | 1.40627 | 0.0398333 |  | -33.5312 |  |  | 0 | 0 |
| geonames_time_cat_4d | z | Tier0 | 0 | ZCurve | 3 | 0.0157667 | 0.000377124 | 0.00673333 | 2.33767 | 9.90107 | 0.00326667 |  | -40.3561 |  |  | 0 | 0 |
| osm_time_pop_4d | adacurve | Unknown |  | AdaCurve | 3 | 0.0288667 | 0.00221711 | 0.00266667 | 23.1354 | 1.21353 | 0.0033 |  | -197.595 |  |  | 0 | 0 |
| osm_time_pop_4d | bitallocz_bo | Unknown |  | BitAllocZ-BO | 3 | 0.0154333 | 0.00204342 | 0.00676667 | 2.27607 | 7.2448 | 0.018 |  | -59.1065 |  |  | 0 | 0 |
| osm_time_pop_4d | hilbert_nd | Unknown |  | Hilbert | 3 | 0.0148 | 0.00117757 | 0.00656667 | 2.27083 | 6.72917 | 0.0163333 |  | -52.5773 |  |  | 0 | 0 |
| osm_time_pop_4d | lacsfc_h | Tier1 | 0 | LACSFC-H | 3 | 0.0097 | 0.000748331 | 0.00196667 | 2.53647 | 5.72917 | 0.0064 |  | 0 |  |  | 0 | 0 |
| osm_time_pop_4d | lacsfc_meta_refine_tiny | Unknown |  | LACSFC-Meta | 3 | 0.0100333 | 0.00101434 | 0.002 | 2.58853 | 5.6302 | 0.00776667 |  | -3.43643 |  |  | 1 | 0 |
| osm_time_pop_4d | lacsfc_opt_mid | Unknown |  | LACSFC-Opt | 3 | 0.00973333 | 0.000817856 | 0.00196667 | 2.66667 | 5.59897 | 0.00663333 |  | -0.343643 |  |  | 0 | 0 |
| osm_time_pop_4d | lmsfc_bo | Unknown |  | LMSFC-BO | 3 | 0.0103 | 0.000355903 | 0.00353333 | 2.3594 | 1.5365 | 0.0319667 |  | -6.18557 |  |  | 0 | 0 |
| osm_time_pop_4d | piecewise | Unknown |  | PiecewiseSFC | 3 | 0.0161333 | 0.000754247 | 0.00676667 | 4.83853 | 1.53647 | 0.0397333 |  | -66.323 |  |  | 0 | 0 |
| osm_time_pop_4d | z | Tier0 | 0 | ZCurve | 3 | 0.0159333 | 0.000953357 | 0.0067 | 2.48953 | 10.7396 | 0.00333333 |  | -64.2612 |  |  | 0 | 0 |
| tiger_time_class_4d | adacurve | Unknown |  | AdaCurve | 3 | 0.0226667 | 0.000449691 | 0.00246667 | 17.5156 | 1.2083 | 0.0033 |  | -154.682 |  |  | 0 | 0 |
| tiger_time_class_4d | bitallocz_bo | Unknown |  | BitAllocZ-BO | 3 | 0.0139667 | 0.000410961 | 0.00663333 | 2.11457 | 6.93227 | 0.0178 |  | -56.9288 |  |  | 0 | 0 |
| tiger_time_class_4d | hilbert_nd | Unknown |  | Hilbert | 3 | 0.014 | 0.000355903 | 0.0065 | 2.1198 | 6.76563 | 0.0159 |  | -57.3034 |  |  | 0 | 0 |
| tiger_time_class_4d | lacsfc_h | Tier1 | 0 | LACSFC-H | 3 | 0.0089 | 0.000374166 | 0.00196667 | 2.10937 | 5.60413 | 0.00643333 |  | 0 |  |  | 0 | 0 |
| tiger_time_class_4d | lacsfc_meta_refine_tiny | Unknown |  | LACSFC-Meta | 3 | 0.00923333 | 0.000418994 | 0.00193333 | 2.41143 | 5.72397 | 0.0065 |  | -3.74532 |  |  | 1 | 0 |
| tiger_time_class_4d | lacsfc_opt_mid | Unknown |  | LACSFC-Opt | 3 | 0.0094 | 0.000163299 | 0.00193333 | 2.53643 | 5.6406 | 0.00653333 |  | -5.61798 |  |  | 0 | 0 |
| tiger_time_class_4d | lmsfc_bo | Unknown |  | LMSFC-BO | 3 | 0.009 | 8.16497e-05 | 0.00313333 | 1.98957 | 1.2656 | 0.0327667 |  | -1.1236 |  |  | 0 | 0 |
| tiger_time_class_4d | piecewise | Unknown |  | PiecewiseSFC | 3 | 0.0137 | 0.000282843 | 0.00623333 | 3.16147 | 1.4271 | 0.0396 |  | -53.9326 |  |  | 0 | 0 |
| tiger_time_class_4d | z | Tier0 | 0 | ZCurve | 3 | 0.0142333 | 4.71405e-05 | 0.00653333 | 2.09377 | 10.2813 | 0.0032 |  | -59.9251 |  |  | 0 | 0 |

## Interpretation guide

- `query_s_gain_vs_lacsfc_h_pct > 0` means the mode improves over Tier1.
- `query_s_gain_vs_same_budget_opt_pct > 0` means Tier3 refine beats Tier2 at the same CMA-ES budget.
- `query_s_gap_to_opt_fast_pct <= 0` means the mode matches or beats the stronger Tier2 budget.
- `meta_loaded_rate=1` and `fallback_rate=0` are required for clean Tier3 evidence.

## Raw rows

| workload | mode | tier | budget_evals | seed | curve | query_s | intervals_s | avg_amp | avg_seg | build_s | meta_model_loaded | meta_fallback_used | meta_direct_deploy | meta_error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| geonames_time_cat_4d | adacurve | Unknown |  | 0 | AdaCurve | 0.019 | 0.0023 | 12.3385 | 1.4688 | 0.0033 | False | False | False |  |
| geonames_time_cat_4d | adacurve | Unknown |  | 1 | AdaCurve | 0.0242 | 0.0025 | 16.5599 | 1.0469 | 0.0034 | False | False | False |  |
| geonames_time_cat_4d | adacurve | Unknown |  | 2 | AdaCurve | 0.0193 | 0.0025 | 12.3438 | 1.1719 | 0.0034 | False | False | False |  |
| geonames_time_cat_4d | bitallocz_bo | Unknown |  | 0 | BitAllocZ-BO | 0.015 | 0.0062 | 2.612 | 7.0 | 0.0178 | False | False | False |  |
| geonames_time_cat_4d | bitallocz_bo | Unknown |  | 1 | BitAllocZ-BO | 0.0159 | 0.007 | 2.2396 | 7.5469 | 0.0176 | False | False | False |  |
| geonames_time_cat_4d | bitallocz_bo | Unknown |  | 2 | BitAllocZ-BO | 0.0149 | 0.0066 | 2.0625 | 6.8594 | 0.018 | False | False | False |  |
| geonames_time_cat_4d | hilbert_nd | Unknown |  | 0 | Hilbert | 0.0153 | 0.0063 | 2.6276 | 5.8438 | 0.0163 | False | False | False |  |
| geonames_time_cat_4d | hilbert_nd | Unknown |  | 1 | Hilbert | 0.0158 | 0.0069 | 2.1771 | 6.9062 | 0.0162 | False | False | False |  |
| geonames_time_cat_4d | hilbert_nd | Unknown |  | 2 | Hilbert | 0.0142 | 0.0063 | 2.0391 | 6.375 | 0.0161 | False | False | False |  |
| geonames_time_cat_4d | lacsfc_h | Tier1 | 0 | 0 | LACSFC-H | 0.011 | 0.002 | 2.5807 | 5.4531 | 0.0066 | False | False | False |  |
| geonames_time_cat_4d | lacsfc_h | Tier1 | 0 | 1 | LACSFC-H | 0.0119 | 0.0023 | 2.4688 | 5.9375 | 0.007 | False | False | False |  |
| geonames_time_cat_4d | lacsfc_h | Tier1 | 0 | 2 | LACSFC-H | 0.0108 | 0.002 | 2.4375 | 5.1094 | 0.0065 | False | False | False |  |
| geonames_time_cat_4d | lacsfc_meta_refine_tiny | Unknown |  | 0 | LACSFC-Meta | 0.0112 | 0.0019 | 3.0729 | 5.4844 | 0.0065 | True | False | False |  |
| geonames_time_cat_4d | lacsfc_meta_refine_tiny | Unknown |  | 1 | LACSFC-Meta | 0.03 | 0.0051 | 2.6719 | 5.9531 | 0.0164 | True | False | False |  |
| geonames_time_cat_4d | lacsfc_meta_refine_tiny | Unknown |  | 2 | LACSFC-Meta | 0.0114 | 0.0022 | 2.3594 | 5.0 | 0.0073 | True | False | False |  |
| geonames_time_cat_4d | lacsfc_opt_mid | Unknown |  | 0 | LACSFC-Opt | 0.011 | 0.002 | 3.0391 | 5.5781 | 0.0066 | False | False | False |  |
| geonames_time_cat_4d | lacsfc_opt_mid | Unknown |  | 1 | LACSFC-Opt | 0.0111 | 0.002 | 2.599 | 5.9375 | 0.0068 | False | False | False |  |
| geonames_time_cat_4d | lacsfc_opt_mid | Unknown |  | 2 | LACSFC-Opt | 0.0102 | 0.0019 | 2.6562 | 5.1094 | 0.0065 | False | False | False |  |
| geonames_time_cat_4d | lmsfc_bo | Unknown |  | 0 | LMSFC-BO | 0.0115 | 0.004 | 2.6745 | 1.7344 | 0.0321 | False | False | False |  |
| geonames_time_cat_4d | lmsfc_bo | Unknown |  | 1 | LMSFC-BO | 0.0111 | 0.0035 | 2.3359 | 1.4688 | 0.0316 | False | False | False |  |
| geonames_time_cat_4d | lmsfc_bo | Unknown |  | 2 | LMSFC-BO | 0.01 | 0.0033 | 2.2031 | 1.3906 | 0.0325 | False | False | False |  |
| geonames_time_cat_4d | piecewise | Unknown |  | 0 | PiecewiseSFC | 0.0151 | 0.0062 | 3.7109 | 1.3281 | 0.0395 | False | False | False |  |
| geonames_time_cat_4d | piecewise | Unknown |  | 1 | PiecewiseSFC | 0.0158 | 0.0062 | 3.8542 | 1.4688 | 0.0395 | False | False | False |  |
| geonames_time_cat_4d | piecewise | Unknown |  | 2 | PiecewiseSFC | 0.0141 | 0.0061 | 3.1484 | 1.4219 | 0.0405 | False | False | False |  |
| geonames_time_cat_4d | z | Tier0 | 0 | 0 | ZCurve | 0.0155 | 0.0066 | 2.5286 | 8.9688 | 0.0034 | False | False | False |  |
| geonames_time_cat_4d | z | Tier0 | 0 | 1 | ZCurve | 0.0163 | 0.0068 | 2.3906 | 10.4219 | 0.0032 | False | False | False |  |
| geonames_time_cat_4d | z | Tier0 | 0 | 2 | ZCurve | 0.0155 | 0.0068 | 2.0938 | 10.3125 | 0.0032 | False | False | False |  |
| osm_time_pop_4d | adacurve | Unknown |  | 0 | AdaCurve | 0.0274 | 0.0026 | 22.5469 | 1.1875 | 0.0032 | False | False | False |  |
| osm_time_pop_4d | adacurve | Unknown |  | 1 | AdaCurve | 0.032 | 0.0028 | 25.7344 | 1.3281 | 0.0033 | False | False | False |  |
| osm_time_pop_4d | adacurve | Unknown |  | 2 | AdaCurve | 0.0272 | 0.0026 | 21.125 | 1.125 | 0.0034 | False | False | False |  |
| osm_time_pop_4d | bitallocz_bo | Unknown |  | 0 | BitAllocZ-BO | 0.0153 | 0.0065 | 2.5469 | 8.2188 | 0.0175 | False | False | False |  |
| osm_time_pop_4d | bitallocz_bo | Unknown |  | 1 | BitAllocZ-BO | 0.018 | 0.0076 | 2.2188 | 7.5156 | 0.0186 | False | False | False |  |
| osm_time_pop_4d | bitallocz_bo | Unknown |  | 2 | BitAllocZ-BO | 0.013 | 0.0062 | 2.0625 | 6.0 | 0.0179 | False | False | False |  |
| osm_time_pop_4d | hilbert_nd | Unknown |  | 0 | Hilbert | 0.016 | 0.0065 | 2.5156 | 6.9375 | 0.0168 | False | False | False |  |
| osm_time_pop_4d | hilbert_nd | Unknown |  | 1 | Hilbert | 0.0152 | 0.007 | 2.2969 | 7.0938 | 0.0161 | False | False | False |  |
| osm_time_pop_4d | hilbert_nd | Unknown |  | 2 | Hilbert | 0.0132 | 0.0062 | 2.0 | 6.1562 | 0.0161 | False | False | False |  |
| osm_time_pop_4d | lacsfc_h | Tier1 | 0 | 0 | LACSFC-H | 0.0099 | 0.002 | 2.5156 | 6.3125 | 0.0065 | False | False | False |  |
| osm_time_pop_4d | lacsfc_h | Tier1 | 0 | 1 | LACSFC-H | 0.0105 | 0.002 | 2.75 | 6.2812 | 0.0064 | False | False | False |  |
| osm_time_pop_4d | lacsfc_h | Tier1 | 0 | 2 | LACSFC-H | 0.0087 | 0.0019 | 2.3438 | 4.5938 | 0.0063 | False | False | False |  |
| osm_time_pop_4d | lacsfc_meta_refine_tiny | Unknown |  | 0 | LACSFC-Meta | 0.0108 | 0.0023 | 2.5 | 6.1406 | 0.0104 | True | False | False |  |
| osm_time_pop_4d | lacsfc_meta_refine_tiny | Unknown |  | 1 | LACSFC-Meta | 0.0107 | 0.0019 | 2.7656 | 6.1406 | 0.0064 | True | False | False |  |
| osm_time_pop_4d | lacsfc_meta_refine_tiny | Unknown |  | 2 | LACSFC-Meta | 0.0086 | 0.0018 | 2.5 | 4.6094 | 0.0065 | True | False | False |  |
| osm_time_pop_4d | lacsfc_opt_mid | Unknown |  | 0 | LACSFC-Opt | 0.0101 | 0.002 | 2.75 | 6.0469 | 0.0068 | False | False | False |  |
| osm_time_pop_4d | lacsfc_opt_mid | Unknown |  | 1 | LACSFC-Opt | 0.0105 | 0.002 | 2.8281 | 6.1094 | 0.0066 | False | False | False |  |
| osm_time_pop_4d | lacsfc_opt_mid | Unknown |  | 2 | LACSFC-Opt | 0.0086 | 0.0019 | 2.4219 | 4.6406 | 0.0065 | False | False | False |  |
| osm_time_pop_4d | lmsfc_bo | Unknown |  | 0 | LMSFC-BO | 0.0108 | 0.0039 | 2.5625 | 1.7188 | 0.0316 | False | False | False |  |
| osm_time_pop_4d | lmsfc_bo | Unknown |  | 1 | LMSFC-BO | 0.0101 | 0.0034 | 2.2188 | 1.4688 | 0.0321 | False | False | False |  |
| osm_time_pop_4d | lmsfc_bo | Unknown |  | 2 | LMSFC-BO | 0.01 | 0.0033 | 2.2969 | 1.4219 | 0.0322 | False | False | False |  |
| osm_time_pop_4d | piecewise | Unknown |  | 0 | PiecewiseSFC | 0.0172 | 0.0072 | 5.375 | 1.7344 | 0.0398 | False | False | False |  |
| osm_time_pop_4d | piecewise | Unknown |  | 1 | PiecewiseSFC | 0.0156 | 0.0064 | 4.6562 | 1.3594 | 0.0397 | False | False | False |  |
| osm_time_pop_4d | piecewise | Unknown |  | 2 | PiecewiseSFC | 0.0156 | 0.0067 | 4.4844 | 1.5156 | 0.0397 | False | False | False |  |
| osm_time_pop_4d | z | Tier0 | 0 | 0 | ZCurve | 0.0157 | 0.0065 | 2.7812 | 10.8906 | 0.0032 | False | False | False |  |
| osm_time_pop_4d | z | Tier0 | 0 | 1 | ZCurve | 0.0172 | 0.0068 | 2.4062 | 10.8281 | 0.0035 | False | False | False |  |
| osm_time_pop_4d | z | Tier0 | 0 | 2 | ZCurve | 0.0149 | 0.0068 | 2.2812 | 10.5 | 0.0033 | False | False | False |  |
| tiger_time_class_4d | adacurve | Unknown |  | 0 | AdaCurve | 0.0233 | 0.0026 | 18.0469 | 1.2812 | 0.0033 | False | False | False |  |
| tiger_time_class_4d | adacurve | Unknown |  | 1 | AdaCurve | 0.0223 | 0.0024 | 17.4219 | 1.1406 | 0.0033 | False | False | False |  |
| tiger_time_class_4d | adacurve | Unknown |  | 2 | AdaCurve | 0.0224 | 0.0024 | 17.0781 | 1.2031 | 0.0033 | False | False | False |  |
| tiger_time_class_4d | bitallocz_bo | Unknown |  | 0 | BitAllocZ-BO | 0.0135 | 0.0065 | 2.1406 | 6.5625 | 0.0179 | False | False | False |  |
| tiger_time_class_4d | bitallocz_bo | Unknown |  | 1 | BitAllocZ-BO | 0.0145 | 0.007 | 2.125 | 7.7812 | 0.0175 | False | False | False |  |
| tiger_time_class_4d | bitallocz_bo | Unknown |  | 2 | BitAllocZ-BO | 0.0139 | 0.0064 | 2.0781 | 6.4531 | 0.018 | False | False | False |  |
| tiger_time_class_4d | hilbert_nd | Unknown |  | 0 | Hilbert | 0.0137 | 0.0063 | 2.2344 | 7.125 | 0.0157 | False | False | False |  |
| tiger_time_class_4d | hilbert_nd | Unknown |  | 1 | Hilbert | 0.0145 | 0.0069 | 2.125 | 6.9219 | 0.0158 | False | False | False |  |
| tiger_time_class_4d | hilbert_nd | Unknown |  | 2 | Hilbert | 0.0138 | 0.0063 | 2.0 | 6.25 | 0.0162 | False | False | False |  |
| tiger_time_class_4d | lacsfc_h | Tier1 | 0 | 0 | LACSFC-H | 0.0088 | 0.002 | 2.2656 | 5.6562 | 0.0064 | False | False | False |  |
| tiger_time_class_4d | lacsfc_h | Tier1 | 0 | 1 | LACSFC-H | 0.0094 | 0.002 | 2.1406 | 6.0781 | 0.0065 | False | False | False |  |
| tiger_time_class_4d | lacsfc_h | Tier1 | 0 | 2 | LACSFC-H | 0.0085 | 0.0019 | 1.9219 | 5.0781 | 0.0064 | False | False | False |  |
| tiger_time_class_4d | lacsfc_meta_refine_tiny | Unknown |  | 0 | LACSFC-Meta | 0.0088 | 0.0019 | 2.375 | 5.6875 | 0.0064 | True | False | False |  |
| tiger_time_class_4d | lacsfc_meta_refine_tiny | Unknown |  | 1 | LACSFC-Meta | 0.0098 | 0.002 | 2.5312 | 6.1875 | 0.0066 | True | False | False |  |
| tiger_time_class_4d | lacsfc_meta_refine_tiny | Unknown |  | 2 | LACSFC-Meta | 0.0091 | 0.0019 | 2.3281 | 5.2969 | 0.0065 | True | False | False |  |
| tiger_time_class_4d | lacsfc_opt_mid | Unknown |  | 0 | LACSFC-Opt | 0.0094 | 0.0019 | 2.6562 | 5.5312 | 0.0066 | False | False | False |  |
| tiger_time_class_4d | lacsfc_opt_mid | Unknown |  | 1 | LACSFC-Opt | 0.0096 | 0.002 | 2.5156 | 6.125 | 0.0065 | False | False | False |  |
| tiger_time_class_4d | lacsfc_opt_mid | Unknown |  | 2 | LACSFC-Opt | 0.0092 | 0.0019 | 2.4375 | 5.2656 | 0.0065 | False | False | False |  |
| tiger_time_class_4d | lmsfc_bo | Unknown |  | 0 | LMSFC-BO | 0.0091 | 0.003 | 1.9531 | 1.2031 | 0.0335 | False | False | False |  |
| tiger_time_class_4d | lmsfc_bo | Unknown |  | 1 | LMSFC-BO | 0.0089 | 0.0032 | 2.0156 | 1.3125 | 0.0329 | False | False | False |  |
| tiger_time_class_4d | lmsfc_bo | Unknown |  | 2 | LMSFC-BO | 0.009 | 0.0032 | 2.0 | 1.2812 | 0.0319 | False | False | False |  |
| tiger_time_class_4d | piecewise | Unknown |  | 0 | PiecewiseSFC | 0.0139 | 0.0063 | 3.3594 | 1.4688 | 0.0397 | False | False | False |  |
| tiger_time_class_4d | piecewise | Unknown |  | 1 | PiecewiseSFC | 0.0133 | 0.0061 | 3.0469 | 1.3281 | 0.0397 | False | False | False |  |
| tiger_time_class_4d | piecewise | Unknown |  | 2 | PiecewiseSFC | 0.0139 | 0.0063 | 3.0781 | 1.4844 | 0.0394 | False | False | False |  |
| tiger_time_class_4d | z | Tier0 | 0 | 0 | ZCurve | 0.0142 | 0.0065 | 2.2969 | 9.8594 | 0.0032 | False | False | False |  |
| tiger_time_class_4d | z | Tier0 | 0 | 1 | ZCurve | 0.0143 | 0.0066 | 1.9844 | 10.6719 | 0.0032 | False | False | False |  |
| tiger_time_class_4d | z | Tier0 | 0 | 2 | ZCurve | 0.0142 | 0.0065 | 2.0 | 10.3125 | 0.0032 | False | False | False |  |
