# Needs–Design Mapping

## Purpose

This document maps identified user needs to specific design decisions within the project.

The aim is to demonstrate how the forecasting and explainability framework addresses the requirements of policy researchers and machine learning practitioners.

---



## User Groups

The project serves two primary user groups:

1. Policy Researchers
2. Machine Learning Practitioners

---

# User Group 1: Policy Researchers

## Need 1: Accurate GDP Forecasts

### Why it matters

Policy decisions depend on reliable forecasts of economic activity.

### Design Response

- Use multiple forecasting approaches
- Evaluate forecasting performance using MAE, RMSE and MAPE
- Select the best-performing forecasting approach

### Project Component

- Forecasting pipeline
- Model evaluation framework

---

## Need 2: Transparent Model Explanations

### Why it matters

Users must understand the factors driving forecasts.

### Design Response

- Apply SHAP explanations to model outputs
- Generate feature importance rankings
- Visualise key economic drivers

### Project Component

- SHAP analysis module
- Feature importance visualisations

---

## Need 3: Confidence During Economic Crises

### Why it matters

Economic relationships may change during events such as Brexit or COVID-19.

### Design Response

- Analyse explanations separately by regime
- Compare feature rankings across regimes
- Quantify explanation stability

### Project Component

- Regime segmentation framework
- Stability analysis module

---

# User Group 2: Machine Learning Practitioners

## Need 4: Explanation Reliability

### Why it matters

Practitioners need to know whether explanations can be trusted.

### Design Response

- Compute SHAP values for each regime
- Compare rankings between regimes
- Assess consistency using rank correlation

### Project Component

- SHAP stability framework
- Spearman correlation analysis

---

## Need 5: Model Monitoring

### Why it matters

Feature importance may change over time.

### Design Response

- Track regime-level changes in feature importance
- Identify periods of explanation instability

### Project Component

- Regime comparison dashboard
- Stability reporting outputs

---

## Need 6: Quantitative Validation

### Why it matters

Explanation quality should be measurable rather than subjective.

### Design Response

- Define stability metrics
- Establish interpretation thresholds

### Project Component

- Stability scoring framework

---

# Literature-to-Design Mapping

| Literature Finding | Design Decision |
|----------|----------|
| Rudin (2019): Explanations may be unreliable | Evaluate SHAP stability rather than assuming trustworthiness |
| Lundberg & Lee (2017): SHAP provides theoretically grounded attributions | Use SHAP as the primary explanation method |
| Bhatt et al. (2020): Practitioners need monitoring tools | Include stability metrics and regime analysis |
| Bracke et al. (2019): Regulators require explainability | Provide transparent feature attribution outputs |
| ICO & Alan Turing (2020): Explanations should support accountability | Document explanation methodology clearly |
| Slack et al. (2020): SHAP can be misleading | Treat SHAP outputs cautiously and evaluate robustness |
| Mittelstadt et al. (2019): Explanations must be meaningful to users | Focus on interpretable visualisations and stability reporting |

---

# Design Requirements Summary

| Requirement | Implementation |
|------------|----------------|
| Accurate forecasting | ARIMA, Ridge, XGBoost, LightGBM |
| Forecast evaluation | MAE, RMSE, MAPE |
| Explainability | SHAP |
| Explanation reliability | Stability analysis |
| Regime awareness | Economic regime segmentation |
| Quantitative validation | Spearman rank correlation |
| Communication | Visualisations and summary tables |

---

# Key Takeaway

The design of the project is directly derived from identified user needs and literature findings.

Rather than treating explainability as an optional feature, the project incorporates explanation stability analysis as a core design component to support both policy interpretation and machine learning model validation.