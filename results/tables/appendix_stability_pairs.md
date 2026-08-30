**SHAP stability by regime pair (XGBoost). Spearman rank correlation of feature-importance rankings between every pair of regimes, all fifteen combinations of the six regimes.**

| Regime A | Regime B | Spearman rho | Band | Involves small regime |
| --- | --- | --- | --- | --- |
| Pre-GFC Stability | Global Financial Crisis | 1.000 | stable | Yes |
| Pre-GFC Stability | Post-GFC Recovery | 1.000 | stable | No |
| Pre-GFC Stability | Brexit Transition | 1.000 | stable | No |
| Pre-GFC Stability | COVID-19 Shock | 1.000 | stable | Yes |
| Pre-GFC Stability | Post-COVID Recovery | 1.000 | stable | No |
| Global Financial Crisis | Post-GFC Recovery | 1.000 | stable | Yes |
| Global Financial Crisis | Brexit Transition | 1.000 | stable | Yes |
| Global Financial Crisis | COVID-19 Shock | 1.000 | stable | Yes |
| Global Financial Crisis | Post-COVID Recovery | 1.000 | stable | Yes |
| Post-GFC Recovery | Brexit Transition | 1.000 | stable | No |
| Post-GFC Recovery | COVID-19 Shock | 1.000 | stable | Yes |
| Post-GFC Recovery | Post-COVID Recovery | 1.000 | stable | No |
| Brexit Transition | COVID-19 Shock | 1.000 | stable | Yes |
| Brexit Transition | Post-COVID Recovery | 1.000 | stable | No |
| COVID-19 Shock | Post-COVID Recovery | 1.000 | stable | Yes |

"Involves small regime" flags any pair where either regime has fewer than the small-sample threshold of observations (Global Financial Crisis and COVID-19 Shock, six quarters each; see Section 3.8.3).
