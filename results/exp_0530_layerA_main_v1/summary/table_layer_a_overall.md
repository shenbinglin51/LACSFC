# Layer A Overall All-Curve Comparison

Values are normalized query time relative to Hilbert.

| Dataset | N | d | RowMajor | ColumnMajor | ZCurve | Hilbert | LMSFC-BO | PiecewiseSFC | BitAllocZ-BO | AdaCurve | LACSFC-H | LACSFC-Opt | LACSFC-Meta | Best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gmm_4d | 50000 | 4 | pending | pending | 1.204 | 1 | 0.9895 | 1.121 | 1.09 | 2.105 | 0.8418 | 0.8731 | 0.909 | LACSFC-H |
| osm_3d | 50000 | 3 | pending | pending | 1.073 | 1 | 0.9935 | 1.54 | 0.9857 | 7.476 | 1.26 | 0.883 | 0.9623 | LACSFC-Opt |
| uniform_4d | 50000 | 4 | pending | pending | 0.76 | 1 | 0.6844 | 0.9122 | 1.079 | 7.488 | 0.6134 | 0.7582 | 0.6443 | LACSFC-H |
