# Personas

## Purpose

This document defines the key user groups for the UK GDP forecasting and explainability system.

These personas are used to guide the design of forecasting models, SHAP interpretability outputs, and regime-based stability analysis.

---

# Persona 1: Policy Researcher (Primary User)

## Profile

A macroeconomist working in a policy institution such as:

- Bank of England
- HM Treasury
- Office for National Statistics (ONS)
- Economic research institutes

## Goals

- Produce accurate GDP forecasts
- Understand key drivers of economic growth
- Support monetary and fiscal policy decisions
- Communicate findings to non-technical stakeholders

## Context of Use

- Works with quarterly macroeconomic data
- Operates under time pressure (policy deadlines)
- Requires justifiable and interpretable outputs

## Pain Points

- Black-box models are difficult to justify in policy reports
- Economic relationships change during crises (GFC, Brexit, COVID-19)
- Forecasts without explanations are not actionable
- Uncertainty is hard to communicate

## Needs

- Accurate GDP forecasts
- Transparent explanations of model behaviour
- Stable interpretation across time and regimes
- Clear communication of uncertainty

---

# Persona 2: Machine Learning Practitioner

## Profile

A data scientist or ML engineer working in:

- Central bank analytics teams
- Government data science units
- Financial institutions
- Economic forecasting teams

## Goals

- Build high-performing forecasting models
- Monitor model performance over time
- Ensure model reliability in production
- Understand feature behaviour

## Context of Use

- Works with structured time-series economic data
- Responsible for model deployment and monitoring
- Needs diagnostic tools for model behaviour

## Pain Points

- Model performance alone does not explain behaviour
- Feature importance can change unexpectedly over time
- Difficult to detect when models become unstable
- Lack of tools for explanation monitoring

## Needs

- Reliable model explanations
- Tools to detect explanation drift
- Quantitative stability metrics
- Ability to compare model behaviour across time

---

# Summary

These personas represent the two core stakeholder groups:

- Policy Researchers needs interpretability and trust
- ML Practitioners needs monitoring and stability

Their needs directly inform the design of the forecasting and SHAP stability framework used in this dissertation.