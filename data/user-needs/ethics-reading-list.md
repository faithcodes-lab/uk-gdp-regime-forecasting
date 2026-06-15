# Ethics & XAI Reading List

**Purpose:** Structured reading list for the ethics framing of the dissertation (Chapter 3) and the methodological literature underpinning the SHAP analysis (Chapter 4). Reading these papers helps structures the engagement, summarises why each paper matters, and flags what to focus on for each one.


---


## 1. Rudin (2019): Antagonistic-to-thesis critique

**Full citation (UWE Harvard):**
Rudin, C. (2019) 'Stop explaining black box machine learning models for high-stakes decisions and use interpretable models instead', *Nature Machine Intelligence*, 1(5), pp. 206–215.


**Key argument:** Post-hoc explanations of black-box models for high-stakes decisions are unreliable, and the field should pivot to inherently interpretable models instead.

**Why this paper matters for the project:**

#### 1. Rudin’s Definition of “High-Stakes” and Whether Macroeconomic Forecasting Fits

**What Rudin Means by High-Stakes:**

Rudin does not provide a formal mathematical definition of “high-stakes.” Instead, she characterises high-stakes decision contexts as situations where model outputs can have serious consequences for individuals or society. Throughout the paper, she repeatedly uses examples from healthcare, criminal justice, environmental safety, finance, and public policy, where incorrect predictions can lead to substantial harm.  ￼

Examples provided include:

* Incorrect parole decisions.
* Poor bail decisions.
* Unsafe air-quality recommendations.
* Healthcare decision-making.
* Financial and credit decisions.  ￼

The underlying principle is that:

When decisions materially affect human welfare, transparency and accountability become more important than marginal gains in predictive accuracy.

Rudin therefore argues that black-box models should be avoided whenever an interpretable alternative with similar predictive performance exists.  ￼

---

#### Does UK Macroeconomic Forecasting Fit This Definition?

This is not explicitly discussed in the paper, but there is a strong argument that it does.

GDP forecasts influence:

* Monetary policy decisions by the Bank of England.
* Interest rate setting.
* Government fiscal planning.
* Public spending decisions.
* Economic policy responses during crises.

Unlike Rudin’s examples, GDP forecasts do not directly determine outcomes for individual people. However, they influence decisions that affect millions of people indirectly through inflation, employment, taxation, borrowing costs, and government spending.

Therefore, UK GDP forecasting can reasonably be viewed as a high-stakes policy-support context, even if it is not as directly consequential as healthcare or criminal justice decisions.

**Implication for the Dissertation**

This creates a tension at the heart of the project:

* The dissertation uses XGBoost and LightGBM.
* Rudin argues against relying on black-box models in high-stakes settings.
* The dissertation therefore needs to justify why post-hoc explanations are being studied rather than assuming they are sufficient.

This makes Rudin one of the strongest critical papers relevant to the project.

---

#### 2. Interpretable-by-Design vs Post-Hoc Explained Models

This distinction is the central argument of the paper.

---

#### a. Interpretable-by-Design Models

An interpretable model is one whose reasoning process is directly visible and understandable without requiring an additional explanation layer.

Examples discussed by Rudin include:

* Sparse linear models.
* Rule lists.
* Decision rules.
* Additive models.
* Prototype-based neural networks.  ￼  ￼

The key property is:

The explanation is the model itself.

The user can inspect the model and directly observe how predictions are generated.

---

#### b. Post-Hoc Explained Models

Post-hoc explanation methods attempt to explain a model after it has already been trained.

Examples include:

* SHAP.
* LIME.
* Counterfactual explanations.
* Feature attribution methods.

Rudin argues that these explanations are fundamentally different from interpretable models because they constitute a second model attempting to approximate or explain the first model.  ￼

The concern is that:

The explanation may not faithfully represent what the original model is actually doing.

As a result, explanations may be:

* unreliable,
* misleading,
* unstable,
* or incomplete.  ￼

---

#### Why This Matters for the Dissertation

The dissertation uses:

* XGBoost
* LightGBM

These are black-box models.

The project then applies:

* SHAP

which is a post-hoc explanation method.

Therefore, the dissertation sits precisely in the category that Rudin criticises.

Rather than ignoring this criticism, the project can be positioned as:

An empirical investigation into whether SHAP explanations remain reliable across structural economic regimes.

---

#### 3. Rashomon Effect

The Rashomon Effect is one of the most important concepts in the paper.

---

#### Definition

Rudin defines the Rashomon Set as:

The collection of reasonably accurate models for a given prediction problem.  ￼

The key insight is:

Many different models may achieve similar predictive accuracy while relying on different internal structures or decision rules.

--- 

Why This Happens

Because real-world data are noisy and finite, there is often no single uniquely optimal model.

Different algorithms can therefore produce:

* different feature relationships,
* different parameter estimates,
* different decision boundaries,

while still achieving similar predictive performance.  ￼

---

Connection to Stability

Rudin argues that large Rashomon Sets naturally create instability:

Small changes in data or modelling choices may lead to a different model that performs equally well.  ￼

Rudin later links this directly to algorithmic instability, noting that highly correlated features can produce multiple equally accurate models with very different structures.  ￼

---

#### Why This Matters for SHAP

SHAP implicitly assumes that feature importance rankings provide meaningful explanations.

However, the Rashomon Effect raises a challenge:

If multiple equally accurate models exist, then:

* Model A may rank unemployment highest.
* Model B may rank inflation highest.

Both models may forecast GDP equally well.

This means there may not be a single uniquely correct explanation.

The consequence is that:

Explanation instability may be a natural consequence of model multiplicity rather than a failure of SHAP itself.

This is highly relevant to the dissertation’s focus on explanation stability across economic regimes.

---

#### Gaps Identified in this reading

Gap 1: Lack of Empirical Evaluation of Explanation Stability Across Structural Regimes

Rudin repeatedly argues that post-hoc explanations may be unreliable or unstable.  ￼

However, the paper does not provide a framework for measuring how explanation reliability changes over time.

Specifically, Rudin does not investigate:

* structural breaks,
* economic regime changes,
* temporal evolution of feature importance.

For macroeconomic forecasting, this is important because relationships between variables can change dramatically during:

* the Global Financial Crisis,
* Brexit,
* COVID-19.

Gap Statement

While Rudin (2019) argues that post-hoc explanations may be unreliable, limited research evaluates whether feature attribution methods such as SHAP remain stable across structural economic regimes.

---

Gap 2: No Quantitative Framework for Measuring Explanation Reliability

Rudin raises concerns regarding:

* instability,
* model multiplicity,
* and explanation reliability.

However, she does not propose a quantitative framework for assessing these issues.

The paper identifies the problem but does not provide:

* stability metrics,
* thresholds,
* benchmarking procedures,
* regime-level comparison methods.

Gap Statement

Although Rudin (2019) highlights the possibility that explanations may be unstable due to the Rashomon Effect, no quantitative methodology is proposed for measuring explanation stability or determining whether explanations are sufficiently consistent for decision support.

This gap is especially powerful because it leads directly to the proposed use of:

* SHAP rankings,
* Spearman rank correlation,
* regime-by-regime stability analysis.

---

Strategic Takeaway for the Dissertation

I should not position the dissertation as disproving Rudin.

A stronger position is:

Rudin argues that post-hoc explanations may be unreliable. This dissertation investigates that concern empirically by evaluating whether SHAP explanations remain stable across major UK economic regimes.

That framing turns one of the strongest critiques of SHAP into one of the strongest motivations for the research.



---

## 2. Lundberg & Lee (2017): SHAP foundational paper 

**Full citation (UWE Harvard):**
Lundberg, S.M. and Lee, S.-I. (2017) 'A unified approach to interpreting model predictions', in *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, Long Beach, CA, USA, 4–9 December, pp. 4765–4774.

**Key argument:** SHAP values, derived from cooperative game theory (Shapley values), provide a unique additive feature attribution method satisfying three desirable properties (local accuracy, missingness, consistency).



#### 1. Purpose of the Paper

Before SHAP, multiple explanation methods existed (e.g., LIME, DeepLIFT, Layer-wise Relevance Propagation), often producing different explanations for the same model. Lundberg and Lee (2017) sought to develop a unified explanation framework with desirable theoretical properties.

The paper demonstrates that many existing explanation approaches can be expressed within a common additive feature attribution framework and shows that SHAP (SHapley Additive Explanations) is the unique method satisfying a set of theoretically desirable axioms.

--- 

#### 2. Key Contribution

The paper introduces SHAP, a model explanation framework based on Shapley values from cooperative game theory.

A SHAP value measures the contribution of an individual feature towards moving a prediction away from a baseline prediction.

Example:

| Component | Value |
|----------|------|
| Baseline GDP forecast | 1.5% |
| Unemployment contribution | +0.6% |
| PMI contribution | +0.5% |
| Interest rate contribution | +0.2% |
| Oil price contribution | +0.2% |
| **Final forecast** | **3.0%** |

The final prediction is decomposed into additive feature contributions, allowing individual predictions to be interpreted.

---

#### 3. Theoretical Guarantees (Axioms)

The paper proves that SHAP is the unique additive feature attribution method satisfying three properties.

#### 3.1 Local Accuracy

Definition

The sum of all SHAP values equals the model prediction.

$f(x)=\phi_0+\sum_{i=1}^{M}\phi_i$

where:

* f(x) = model prediction
* φ₀ = baseline prediction  
* φᵢ = contribution of feature i

What it guarantees

* Nothing is left unexplained.
* The explanation exactly reconstructs the prediction.

Limitation

Local accuracy guarantees that the explanation adds up correctly, but does not guarantee:

* temporal stability,
* causal interpretation,
* robustness across structural breaks,
* consistency across economic regimes.

These issues fall outside the scope of the original paper.

---

#### 3.2 Missingness

Definition

If a feature is absent from the model, its SHAP value is zero.

What it guarantees

* Unused variables cannot receive attribution.
* Prevents arbitrary assignment of importance.

Limitation

Missingness does not address:

* multicollinearity,
* proxy variables,
* redundant predictors.

This is particularly relevant in macroeconomics where variables such as inflation, unemployment, GDP growth and PMI are often highly correlated.

---

#### 3.3 Consistency

Definition

If a model changes such that a feature contributes more strongly to predictions, its SHAP value should not decrease.

What it guarantees

Feature importance behaves logically.

Limitation

Consistency applies within a given model and dataset. It does not guarantee consistency:

* across time,
* across datasets,
* across structural breaks,
* across economic regimes.

---

### 4. TreeSHAP

Definition

TreeSHAP is an exact algorithm for calculating SHAP values for tree-based machine learning models.

Examples include:

* Random Forest
* XGBoost
* LightGBM

Unlike sampling-based approaches, TreeSHAP exploits tree structures to calculate exact Shap values efficiently.

Relevance to the Dissertation

The dissertation’s primary forecasting models are:

* XGBoost
* LightGBM

Therefore TreeSHAP will be the explanation method used throughout the analysis.

---

#### 5. Additivity Assumption

A central assumption of SHAP is additive decomposition:

Prediction = Baseline + Feature Contributions

Even when the underlying machine learning model captures highly complex interactions, SHAP represents those effects through additive feature contributions.

Potential Concern

Economic systems are highly interconnected.

Examples include:

* Interest rates affecting inflation.
* Inflation affecting consumption.
* Consumption affecting GDP growth.

These relationships are often non-linear and interactive.

SHAP simplifies these interactions into additive contributions, potentially obscuring complex economic relationships.

---

6. Where SHAP’s Guarantees Stop

The paper guarantees:

- Local Accuracy

- Missingness

- Consistency

However, the paper does not guarantee:

- Temporal stability

- Regime stability

- Causal interpretation

- Economic validity

- Stable feature rankings across structural breaks

Consequently, while SHAP explanations are theoretically coherent, the paper does not establish whether those explanations remain reliable when the underlying data-generating process changes substantially.

---

#### 7. Relevance to the Dissertation

This paper provides the theoretical foundation for the dissertation’s interpretability framework.

The study relies on SHAP to explain GDP forecasts generated by XGBoost and LightGBM models.

However, the dissertation investigates an issue that lies beyond the scope of Lundberg and Lee (2017):

Whether SHAP feature attributions remain stable across major structural economic regimes such as the Global Financial Crisis, Brexit, and COVID-19.

The dissertation therefore extends the discussion from explanation validity within a model to explanation stability across changing economic environments.

---

#### 8. Literature Gaps Identified

Gap 1: No Evaluation of SHAP Stability Across Structural Breaks

Lundberg and Lee (2017) establish theoretical guarantees for SHAP explanations within a model but do not investigate whether those explanations remain stable when the underlying data-generating process changes.

This leaves open the question of whether feature attribution rankings remain consistent across major economic disruptions such as the GFC, Brexit, and COVID-19.

---

Gap 2: No Framework for Measuring Explanation Reliability

While SHAP produces feature attributions, the paper does not provide a methodology for evaluating:

* explanation stability,
* explanation reliability,
* regime-level consistency,
* acceptable levels of explanation change.

As a result, users receive explanations without a formal mechanism for assessing whether those explanations remain sufficiently stable for decision support.

---

9. Key Takeaway for the Dissertation

Lundberg and Lee (2017) establish why SHAP explanations are theoretically justified through the properties of local accuracy, missingness, and consistency.

However, these guarantees do not address whether SHAP explanations remain stable when economic relationships change during major structural breaks.

This creates an open research question that directly motivates the dissertation:

- Do SHAP explanations remain sufficiently stable across the Global Financial Crisis, Brexit, and COVID-19 to support economic forecasting and policy interpretation?



---

## 3. Bhatt et al. (2020): Practitioner needs evidence

**Full citation (UWE Harvard):**
Bhatt, U., Xiang, A., Sharma, S., Weller, A., Taly, A., Jia, Y., Ghosh, J., Puri, R., Moura, J.M.F. and Eckersley, P. (2020) 'Explainable machine learning in deployment', in *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency (FAT* '20)*, Barcelona, Spain, 27–30 January, pp. 648–657.


**Key argument :** Practitioners deploying ML explanations report a sustained gap between research-oriented explanation methods and the operational needs of real-world ML teams.


---

#### Key argument

The paper investigates how machine learning practitioners actually use explanations in real deployed systems, showing that explanation methods are not used primarily for theory or interpretability research, but for practical operational tasks such as debugging, monitoring, compliance, and communication.

---

#### 1. Taxonomy of explanation use-cases

Bhatt et al. identify four main real-world uses of explanations:

| Use-case      | Meaning                                   | How practitioners use it |
|---------------|-------------------------------------------|---------------------------|
| Debugging     | Understanding model errors                | Identify why predictions are wrong or unstable |
| Monitoring    | Tracking model behaviour in production    | Detect drift or unexpected feature reliance |
| Compliance    | Meeting regulatory requirements           | Justify automated decisions to auditors/regulators |
| Communication | Explaining outputs to stakeholders        | Translate model output into human-understandable reasoning |

#### Key insight for the disseration:

Explanations are not just for interpretability — they are operational tools in ML systems.

---

#### 2. Gap between research methods and deployment needs

Bhatt et al. show a consistent mismatch between:

Research focus:

* benchmark accuracy
* theoretical properties of explainability methods
* static evaluation of explanations

Practitioner needs:

* stability of explanations over time
* robustness under real-world data drift
* explanations that support decision-making under uncertainty
* interpretability during system failures

Key takeaway

There is a translation gap between academic XAI research and real-world deployment requirements.

---

Why this matters for the dissertation

The project directly sits in this gap because it is testing:

* SHAP explanation stability
* across regimes (GFC, Brexit, COVID)

which is exactly a deployment stress-test question, not a theoretical one.

---

3. Practitioner interview evidence (what you should extract)

The paper reports qualitative findings from interviews with machine learning practitioners across different industries. Rather than focusing on formal evaluation metrics, practitioners describe explanation use in operational contexts.

Key reported themes include:

- Explanations used for diagnosing model errors and unexpected behaviour in production environments.

- Explanations used for monitoring model performance and detecting distribution shift.

- Explanations used to support communication between technical teams and non-technical stakeholders.

- Explanations viewed as insufficient when accuracy metrics alone do not provide insight into model behaviour.

These findings are presented as thematic summaries rather than structured quantitative results, highlighting the practical role of explainability in deployed machine learning systems.

---

Key gaps in Bhatt et al. (2020)

Gap 1: No temporal or regime-based analysis

The study assumes explanation use is stable across time, but does not examine:

* structural breaks
* distribution shifts
* crisis periods (e.g. COVID, financial shocks)
* changing economic regimes


There is no analysis of whether explanation utility or reliability changes when the underlying system enters a different regime.

---

Gap 2:  No quantitative framework for explanation quality

The paper provides:

* taxonomy (qualitative categories)

but no:

* numerical metrics
* thresholds for “good explanations”
* stability measures
* evaluation benchmarks

---

Gap 3: Limited domain specificity (macroeconomics not covered)

Findings are based on:

* general ML systems
* industry deployment contexts

Not:

* macroeconomic forecasting
* central banking
* GDP prediction systems

So transferability is assumed but not validated.

---

Gap 4: No link between explanation use-case and stability requirements

Different use-cases likely require different robustness levels:

* debugging:  can tolerate noise
* compliance: needs consistency
* policy forecasting: requires high stability

But Bhatt et al. do not formalise this.

---

In summary: 


Bhatt et al. (2020) examine how machine learning practitioners use explanations in deployed systems and identify four primary use-cases: debugging, monitoring, compliance, and communication. 

The study highlights a persistent gap between academic research on explainability and practitioner needs, particularly regarding robustness, usability, and operational relevance.

However, the analysis is largely qualitative and does not provide quantitative measures of explanation quality or stability. In addition, the study does not consider how explanation utility may change under temporal distribution shifts or structural breaks, nor does it focus on macroeconomic forecasting contexts where model behaviour may vary significantly across economic regimes.

As a result, the question of whether explanation methods remain reliable under changing system dynamics remains open.

---

Key takeaway for the dissertation

Bhatt et al. tells us:

“Here is how explanations are used in real-world ML systems.”

The dissertation extends this to:

“Do these explanations remain reliable when the system enters fundamentally different economic regimes?”


---

## 4. Bracke et al. (BoE Staff Working Paper, 2019): Central bank practitioner paper

**Full citation (UWE Harvard):**
Bracke, P., Datta, A., Jung, C. and Sen, S. (2019) *Machine learning explainability in finance: an application to default risk analysis*. Bank of England Staff Working Paper No. 816. London: Bank of England.



**Key argument :** Bank of England researchers apply SHAP to a default-risk model and document both the value (transparency, supervisory monitoring) and the limitations (instability, interpretability gaps) of post-hoc explanations in a regulated financial context.


This is one of the most important central bank–specific applications of SHAP and provides direct evidence that:

* SHAP is already used in regulatory and financial supervision contexts
* Explainability is treated as a governance and accountability tool, not just a technical add-on
* Model explanations are expected to support auditability and trust, not only prediction accuracy

For this dissertation, it strengthens the argument that:

SHAP-based explanations are not abstract research outputs but are increasingly embedded in policy-relevant decision systems, including those similar in structure to macroeconomic forecasting.

It therefore directly supports the relevance of evaluating whether SHAP remains reliable under changing economic regimes.

---

#### 1. How BoE researchers frame the trade-off between performance and explainability

The paper highlights a practical tension:

* More complex models (e.g. ensemble methods) improve predictive performance
* But they reduce transparency for supervisors and decision-makers

Bracke et al. show that SHAP is used as a bridge solution, allowing institutions to:

* retain high-performing models
* while recovering interpretable feature-level explanations post-hoc

However, they also note that this is not a perfect substitute for inherently interpretable models in all contexts.

Key implication:
Explainability is treated as a necessary compromise rather than a fully solved problem.

---

#### 2. SHAP limitations identified from a supervisory perspective

From a Bank of England regulatory viewpoint, the paper highlights several concerns:

(a) Instability of explanations

Feature attributions can change when:

* data distribution shifts
* model retraining occurs
* or correlated variables behave differently

This is especially problematic in sensitive scenarios, where interpretability is most needed.

---

(b) Correlation and ambiguity in economic variables

Highly correlated financial and macroeconomic indicators make it difficult to:

* uniquely assign importance to a single variable
* interpret causal meaning from SHAP values

This creates ambiguity in supervisory interpretation.

---

(c) Risk of over-trust in explanations

The paper notes that stakeholders may:

* treat SHAP outputs as “truth-like”
* without recognising their dependence on model structure and data context

This is particularly important in regulated environments.

---

#### 3. How explanations are validated against domain expectations

Bracke et al. do not validate SHAP using statistical metrics alone. Instead, they rely on:

(a) Domain-consistency checks

They assess whether:

* the most important features align with economic intuition
* known risk drivers (e.g. income, credit history) appear prominently

---

(b) Expert judgement (human validation)

Domain experts evaluate whether:

* explanations are plausible
* feature rankings make sense in a credit risk context

---

(c) Case-based reasoning

They inspect individual predictions to:

* confirm whether SHAP attributions align with expected borrower behaviour patterns

---

#### Key methodological insight

This establishes a hybrid validation approach:

SHAP explanations are not validated purely statistically, but through a combination of economic reasoning and expert interpretability checks.

This is directly relevant to the dissertation because SHAP stability analysis can be framed as an extension of this idea:

* Bracke et al. validate plausibility
* The project evaluates consistency over time and regimes

---

4. Strong gaps identified from the paper

Gap 1: No temporal or regime-based evaluation of explanations

The paper evaluates SHAP in a static credit risk setting, but does not assess:

* economic cycles
* financial crises
* structural breaks
* time-varying relationships between variables

Implication for the dissertation:

There is no evidence that SHAP explanations remain stable under macroeconomic regime shifts such as:

* GFC (2008)
* Brexit transition
* COVID-19 shock

---

Gap 2: No formal framework for explanation stability

Although the paper discusses concerns around stability, it does not provide:

* quantitative stability metrics
* ranking consistency measures
* cross-period comparison methods
* thresholds for acceptable explanation variation

Result:
Explainability is discussed qualitatively, but not operationalised as a measurable property.

---

Gap 3: Domain limitation (credit risk vs macroeconomics)

The study focuses on:

* default risk prediction (micro-level financial data)

It does not extend to:

* macroeconomic forecasting
* GDP prediction
* policy-relevant time series models

Implication:
Transferability of conclusions to macroeconomic regimes is assumed, not tested.

---

Key takeaway for the dissertation

Bracke et al. (2019) demonstrate that SHAP is already used in central banking contexts and valued for its interpretability in regulated environments. 

However, the paper treats explanation validity as a static property validated through expert judgement, rather than a dynamic property that may vary across time or economic regimes.

This leaves an open question:

Do SHAP explanations remain consistent and reliable when applied to macroeconomic forecasting problems that involve structural breaks and regime shifts?


---

## 5. ICO & Alan Turing Institute (2020): UK regulatory guidance

**Full citation (UWE Harvard):**
Information Commissioner's Office and Alan Turing Institute (2020) *Explaining decisions made with AI*. London: Information Commissioner's Office.



**Key argument :** Joint UK regulatory and research-body guidance setting out what counts as an adequate explanation of an AI-assisted decision and the procedural obligations on organisations deploying such systems.


#### 1. Explanation framework 

The guidance defines six required explanation types:

| Type                     | Description |
|--------------------------|-------------|
| Rationale explanation    | Why a decision was made |
| Responsibility explanation | Who is accountable for the decision |
| Data explanation         | What data was used in the decision |
| Fairness explanation     | How fairness was considered |
| Safety and performance explanation | How reliable and accurate the system is |
| Impact explanation       | What effects the decision may have |

---

#### 2. Key insight for the dissertation

The document frames explanation as a multi-dimensional requirement, not a purely technical output. This implies that feature attribution methods such as SHAP only cover part of what is required for real-world explainability in policy contexts.

---

#### 3. Relevance to SHAP-based modelling

SHAP can contribute to:

* rationale explanations
* data-driven feature influence explanations

However, it does not directly address:

* responsibility allocation
* fairness justification
* broader societal impact assessment

---

#### 4. Key limitation

The guidance does not provide quantitative methods for measuring explanation quality or stability. It is also not specific to macroeconomic forecasting or time-series models, leaving open how such principles translate to GDP prediction systems.

---

#### Key takeaway

The ICO & Turing guidance establishes that explainability is a regulatory and accountability requirement, not just a modelling feature. 

This reinforces the importance of evaluating whether SHAP-based explanations are sufficiently reliable and meaningful in high-stakes policy environments such as economic forecasting.


---

## 6. Slack et al. (2020): SHAP robustness critique 

**Full citation (UWE Harvard):**
Slack, D., Hilgard, S., Jia, E., Singh, S. and Lakkaraju, H. (2020) 'Fooling LIME and SHAP: adversarial attacks on post hoc explanation methods', in *Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society (AIES '20)*, New York, NY, USA, 7–9 February, pp. 180–186.

**Key argument:** Both LIME and SHAP can be fooled by adversarial classifiers that produce different feature importances on the perturbation samples used by the explainer than on the actual data, undermining the trustworthiness of these explanation methods.



SHAP explanations can appear valid while not reflecting the true internal decision logic of the model.

---

Why this paper matters for the project

This paper provides a direct robustness challenge to SHAP, which is central to this dissertation’s interpretability framework.

It introduces an additional dimension of explanation failure beyond statistical instability:

Type of instability	Source
Regime instability	Structural economic changes (GFC, Brexit, COVID)
Adversarial instability	Intentional manipulation of model-explainer interaction

This strengthens the dissertation’s core argument that:

SHAP explanations must be evaluated for robustness, not assumed to be inherently trustworthy.

---

#### 1. Mechanism of adversarial manipulation

The authors show that adversarial models can exploit the SHAP pipeline by:

* altering how feature perturbations are interpreted
* ensuring SHAP’s local approximations receive misleading signals
* maintaining predictive accuracy while distorting explanation outputs

This leads to a key failure mode:

Explanations can be decoupled from the true decision boundary of the model.

---

#### 2. Implications for trust in SHAP

The paper highlights several practical risks:

* SHAP explanations may be visually consistent but semantically misleading
* Users may over-trust explanation outputs
* Explanation fidelity is not guaranteed even when predictions are accurate

---

Key implication for this dissertation

Even if SHAP explanations appear stable across economic regimes:

stability does not guarantee faithfulness to the underlying model logic.

This introduces an important distinction:

* Stable explanations is not equal to truthful explanations

---

#### 3. Connection to Rudin (2019)

Slack et al. extend Rudin’s critique of post-hoc explanations:

Rudin (2019)Slack et al. (2020)

Post-hoc explanations may be unreliable	Explanations can be intentionally manipulated
Black-box explanations may mislead users	Explanation outputs can be detached from model logic

This reinforces the argument that:

explanation systems require empirical validation, not just theoretical justification.

---

#### 4. Limitations of the study

No macroeconomic or time-series application

The paper does not evaluate:

* GDP forecasting models
* economic time series
* structural breaks or regime shifts
* real-world policy contexts

This limits direct transferability to the dissertation context.

---

5. Research gaps relevant to this dissertation

Gap 1: No evaluation under natural (non-adversarial) distribution shifts

The study focuses on adversarial manipulation but does not consider:

* economic regime changes
* financial crises
* structural breaks in macroeconomic data

---

Gap 2: No time-series or forecasting context

The experiments are conducted on general ML classification tasks, not:

* XGBoost/LightGBM forecasting models
* GDP prediction systems
* macroeconomic datasets

---

Gap 3: No framework for measuring explanation robustness

The paper demonstrates vulnerability but does not provide:

* stability metrics
* robustness scoring methods
* temporal consistency evaluation
* regime-based comparison frameworks

---

#### 6. Explanation robustness framework (summary table)

| Instability Type           | Description | Relevance to Dissertation |
|----------------------------|-------------|----------------------------|
| Adversarial manipulation   | Model intentionally distorts SHAP outputs while maintaining accuracy | Theoretical worst-case failure of SHAP |
| Regime shift (natural)     | Structural economic changes (GFC, Brexit, COVID) | Core focus of dissertation |
| Model variability         | Changes in learned parameters across retraining | Affects SHAP feature rankings over time |

---

Key takeaway

Slack et al. (2020) show that SHAP explanations can be decoupled from the true decision process through adversarial design, even when predictive accuracy remains high.

While this paper focuses on worst-case scenarios, it raises a broader concern directly relevant to this dissertation:

If SHAP explanations can be manipulated without affecting accuracy, their reliability under real-world structural economic changes must be empirically tested.

This provides a strong methodological justification for evaluating:

* SHAP stability across economic regimes
* robustness of feature attribution rankings over time
* reliability of post-hoc explanations in policy-relevant forecasting systems



---

## 7. Croushore & Stark (2001): Data revision bias

**Full citation (UWE Harvard):**
Croushore, D. and Stark, T. (2001) 'A real-time data set for macroeconomists', *Journal of Econometrics*, 105(1), pp. 111–130.

**Key argument (single sentence):** Macroeconomic time series are heavily revised after first publication, and using revised "final" data to evaluate forecasting models overstates predictive performance relative to what would have been achievable in real time.


This leads to a critical insight:

Forecasting models evaluated on revised data may appear more accurate than they would have been in real-world conditions.

---

Why this paper matters for the project

This paper is central to the data realism and evaluation integrity of the dissertation.

It supports the argument that:

* GDP forecasting studies can suffer from look-ahead bias
* model performance may be overstated when using revised datasets
* real-time forecasting is more difficult than back-tested evaluation suggests

For this dissertation, it provides the justification for a key methodological limitation:

The study uses revised macroeconomic data due to the unavailability of consistent real-time data across all UK indicators (2000–2025).

This ensures transparency about the constraints of empirical evaluation.

---

### 1. Nature of data revisions

The paper shows that macroeconomic datasets are not static:

* initial GDP estimates are released quickly
* subsequent revisions adjust values as more information becomes available
* historical economic series can change materially over time

This affects both:

* model training
* model evaluation

---

#### 2. Impact on forecasting accuracy

A key implication is that:

Forecast evaluation using final revised data does not reflect true real-time predictive performance.

This creates an optimism bias in empirical results, because:

* models are trained on “cleaner” information than was originally available
* evaluation uses corrected outcomes rather than initial releases

---

#### 3. Methodological recommendation

Croushore & Stark recommend:

* using real-time data  where available
* evaluating models under information constraints that mimic real-time forecasting

This is considered the gold standard for macroeconomic forecasting research.

---

#### 4. Relevance to this dissertation

This study directly informs the methodological design of the project:

* the analysis uses revised ONS and macroeconomic series
* real-time vintages are not consistently available for all variables in the 2000–2025 window
* therefore, results reflect back-tested performance rather than true real-time forecasting accuracy

----

5. Key limitation acknowledged

While widely used, this dataset framework introduces an unavoidable limitation:

* results may be optimistically biased
* real-world forecasting performance may differ from reported metrics
* some features may not reflect what was known at the time of prediction

---

6. Key takeaway

Croushore & Stark (2001) establish that macroeconomic forecasting is fundamentally affected by data revision processes, meaning that evaluation using final datasets can overstate predictive accuracy.

For this dissertation, the implication is clear:

Forecasting performance should be interpreted with caution, as it may not fully represent real-time predictive conditions due to the use of revised macroeconomic data.



---

## 8. Mittelstadt, Russell & Wachter (2019):Taxonomy of explanations

**Full citation (UWE Harvard):**
Mittelstadt, B., Russell, C. and Wachter, S. (2019) 'Explaining explanations in AI', in *Proceedings of the 2019 Conference on Fairness, Accountability, and Transparency (FAT* '19)*, Atlanta, GA, USA, 29–31 January, pp. 279–288.


**Key argument (single sentence):** ML explanations as currently constructed do not match the contrastive, selective, and socially-embedded structure of human explanation, which limits their practical interpretability for non-expert audiences.


---

#### 1. Contrastive vs Absolute Explanations
- Human explanations are typically contrastive (e.g., “Why this instead of that?”)
- Most ML explanation methods, including SHAP, provide absolute feature attributions
- This mismatch can reduce interpretability in real decision settings

---

#### 2. Selectivity of Human Explanation
- Humans do not expect full decomposition of all contributing factors
- Instead, they prefer selective, simplified reasoning
- ML explanations may overwhelm users with too much information

---

#### 3. Social and Contextual Nature of Explanation
- Explanations are not purely technical
- They depend on audience, intent, and institutional context
- A “good explanation” for a regulator may differ from one for a data scientist

---

#### 4. Trust and Over-Reliance Risk
- Users may over-trust explanations when presented with structured outputs (e.g., SHAP plots)
- This can lead to misplaced confidence in model reasoning
- Important implication for policy-facing forecasting systems

---

#### Key Relevance to This Dissertation

This paper directly supports the argument that:

Even if SHAP is mathematically valid, it may still fail as a practical explanation tool if it does not align with human interpretability needs.

It strengthens the dissertation’s motivation for:

- Evaluating explanation stability across economic regimes
- Moving beyond accuracy to usability and interpretability consistency
- Questioning whether feature attributions remain meaningful in policy contexts

---

#### Potential Gap Identified

### Gap: No Operational Framework for “Human-Meaningful Stability”
While the paper clearly critiques current ML explanations, it does not:

- Define measurable criteria for explanation usability
- Provide metrics for evaluating interpretability quality
- Propose methods for testing explanation usefulness over time or across contexts

### Dissertation Link
This directly motivates the need for:

- Regime-based SHAP stability testing
- Quantitative ranking consistency measures (e.g., Spearman correlation)
- A framework linking explanation stability to real-world interpretability

---

#### Key Takeaway

The paper shifts the focus from:

“Are explanations mathematically correct?”

to

“Are explanations meaningful to humans in context?”

This aligns strongly with the dissertation’s aim of evaluating whether SHAP explanations remain useful and stable across changing economic regimes.

---





