# User Testing Protocol: `regime-shap` Sprint 5 Informal User Testing


**Purpose:** Document the informal user testing of the open-source `regime-shap` package conducted in Sprint 5. The protocol is intentionally lightweight (to fit the realistic timeline) but rigorous (per UWE research-ethics guidance and supervisor direction). It produces evidence for the dissertation's Evaluation chapter  and validates the user needs documented in `personas.md`.


---

## 1. Scope and objectives

### 1.1 What this user testing is

A small-scale (atleast 10 participant), 30-minute-per-participant, voluntary user testing exercise on the `regime-shap` Python package. The exercise asks each participant to: install the package, work through one provided example notebook, optionally attempt the method on their own data, then complete a short structured feedback form.

### 1.2 What this user testing is **not**

- It is not a controlled experiment with hypotheses about cognitive load or learnability.
- It is not a quantitative usability benchmark.
- It is not user testing of the research findings themselves; the research findings are evaluated separately.

### 1.3 Why this user testing exists

Three reasons:

1. **Evidence base for the dissertation's Evaluation chapter.** The chapter would otherwise rest entirely on automated tests and benchmark datasets. Direct usability evidence from testers materially strengthens the evidence base.
2. **Validation of the documented user needs.** `personas.md` and `needs-design-mapping.md` derive user needs from published literature; the testing exercise validates whether those needs hold for real practitioners and whether the design decisions that follow from them actually meet those needs.
3. **Improvement of the package before publication.** Findings feed back into `regime-shap` before the v1.0 PyPI release.

---

## 2. Participants

### 2.1 Inclusion criteria

Participants must be:

- Working data scientists/ students, ML engineers/practitiners, or quantitative researchers with hands-on experience in Python using common data-science libraries (pandas, scikit-learn).
- Familiar with the concept of feature importance, at the level of being able to read a SHAP summary plot.
- Voluntary participants.

### 2.2 Exclusion criteria

The exercise should not be offered to:

- Individuals whose participation would conflict with employer or academic assessment responsibilities
- Individuals with prior detailed exposure to the regime-shap codebase

### 2.3 Sample size and rationale

At least 10 participants.

This sample size is sufficient for qualitative usability testing and thematic analysis, and is feasible within the Sprint 5 timeframe.

### 2.4 Recruitment

Participants will be recruited through:

- Direct outreach to peers and fellow students in data science, economics, or machine learning-related programmes
- Personal academic and professional networks

Participation is voluntary and no incentives are provided.

Candidates who agree are sent the consent form to sign and return before any testing material is shared.

---

## 3. Procedure

### 3.1 Pre-test (5 minutes; the tester does this on their own)

1. Tester reads `participant-information-sheet.md` (sent by email or via shared link).
2. Tester signs `consent-template.md` and returns it the researchers (signed PDF, scanned image, or e-signature; whichever is convenient for the tester).
3. Researcher confirms receipt and provides:
   - A link to the `regime-shap` package on PyPI and the GitHub repository.
   - A link to the example notebook to work through.
   - The feedback form (`feedback-form-template.md`): sent as a Microsoft Form.

### 3.2 Test (≈25 minutes; the tester does this on their own machine)

The tester is asked to:

1. **Install the package** in a fresh Python environment of their choosing (`pip install regime-shap`). Time-cap: 5 minutes; if the installation has not succeeded by 5 minutes, the tester is asked to stop and record the failure in the feedback form.
2. **Run the example notebook** end-to-end  Time-cap: 15 minutes.
3. **(Optional)** Attempt the method on one of their own datasets. Time-cap: the tester decides whether to attempt and how long to spend.
4. **Complete the feedback form**. Time-cap: 5 minutes.

### 3.3 Post-test (Researcher work)

1. Stores the returned consent form and feedback form responses securely on UWE OneDrive (per UWE Data Protection guidance).
2. Codes each tester as "Tester 1", "Tester 2", etc.; no identifying information appears anywhere except in the consent form, which is held separately and is not used in any analysis.
3. Analyses the feedback responses thematically and writes up the findings(user-testing subsection).

---

## 4. Data handling and protection

The exercise is governed by the UK GDPR, the Data Protection Act 2018, and UWE's Data Protection Guidance for UWE Bristol students.
Specifically:

- **Lawful basis:** explicit, written consent obtained via `consent-template.md` before any testing material is shared.
- **Data minimisation:** no name, no employer, no email, and no demographic data are collected on the feedback form. The consent form (which does collect a name and signature) is held separately and never linked to feedback responses in any analysis.
- **Storage:** signed consent forms and feedback responses are stored on UWE OneDrive in a folder accessible only to the researcher. 
- **Pseudonymisation:** testers are referred to as "Tester 1", "Tester 2", etc., in all subsequent reporting (dissertation, research paper, presentation).
- **Retention:** consent forms and feedback responses are retained until the researcher receives official confirmation of her award (anticipated late 2026). After confirmation, all personal data is securely deleted from UWE OneDrive.
- **Withdrawal:** participants may withdraw at any point up to the point of anonymisation in the feedback analysis (which occurs immediately upon receipt of the feedback form). After analysis, the data has been pseudonymised and aggregated and is no longer traceable to any individual; withdrawal after this point is not technically possible.
- **Breach handling:** if any incident affects participant data, researcher reports to her supervisor immediately and follows UWE's incident reporting procedures.

---

## 5. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Tester's identity becomes guessable from feedback content (e.g. tester mentions a unique dataset they own) | Researcher reviews each feedback response before any quotation in the dissertation; quotations are paraphrased rather than verbatim where any identifying detail could be inferred. |
| Tester's employer would object to participation | The recruitment message and PIS make voluntariness explicit; the tester is responsible for deciding whether their employer relationship permits participation. |
| Software installation fails on the tester's machine | The protocol time-caps installation at 5 minutes and treats installation failure as itself a useful finding (recorded in the feedback form). The tester is not asked to debug. |
| Bias from social closeness to the researcher (testers wanting to be kind in feedback) | The feedback form has a free-text field for "what would put you off adopting this in your day job", which surfaces critique that polite social feedback would suppress. |
| Small sample size limits generalisability of findings| Findings will be presented as qualitative feedback rather than representative evidence of all practitioners.|

---

## 6. Outputs

The testing exercise produces:

1. **Signed consent forms**: held on UWE OneDrive, not in this repository, not shared.
2. **Feedback responses**: held on UWE OneDrive in a pseudonymised form (Tester 1, Tester 2, …).
3. **Synthesised findings**:  written up in `report/chapters/05-evaluation.md` and `report/appendices/user-testing-findings.md`. Quotations are paraphrased; no name, employer, or other identifying detail appears.
4. **Package improvements**:  issues filed against the `regime-shap` GitHub repository before the v1.0 release; the issues reference "Tester N" rather than any individual.
