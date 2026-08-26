# Replicability Module — Cross-Paper Claim Comparator Pilot Report

**Date:** 2026-08-26
**Description:** Pilot validation of the Replicability module's cross-paper claim comparator (Steps 3a+3b), using Vandewiele et al. 2021 as ground truth.

## 1. Setup

**task_description** (verbatim):
> predicting preterm birth using SMOTE-balanced XGBoost classifiers on electrohysterogram (EHG) signals

**target_claim** (verbatim):
> Applying SMOTE oversampling to EHG-based preterm birth prediction improves classifier AUC compared to no oversampling

**distilled_query** (Step 3a output, verbatim):
> preterm birth prediction using electrohysterogram signals

**Why Vandewiele et al. 2021 as ground truth.** Vandewiele et al., *Artif Intell Med* 111:101987, is a methodologically confirmed case of SMOTE-before-split data leakage inflating AUC in EHG-based preterm birth prediction, with both an inflated and a corrected AUC value independently documented, and a rich citing literature. This gives the comparator's taxonomy real direct/conceptual/not_comparable and agree/conflict/partially_agree cases to be tested against, rather than a synthetic scenario constructed to exercise the taxonomy artificially.

**This is the second run.** The first run surfaced two issues: title-only candidates causing inconsistent `comparison_type` judgments between equally-uninformative candidates, and the target paper's own prior work being retrieved as if it were an independent comparator. Both were fixed in code (`has_body_text` short-circuit in `_judge_candidate`; `exclude_corpus_id` filtering in `distill_and_retrieve`) before this run.

## 2. Results

| corpus_id | title | comparison_type | agreement | confidence |
|---|---|---|---|---|
| 255151451 | Predicting preterm births from electrohysterogram recordings... | direct | conflict | high |
| 272452849 | Classification of Term and Preterm Birth Data from Elektrohi... | not_comparable | no_comparable_evidence | high |
| 229685414 | Assessing Velocity and Directionality of Uterine Electrical ... | not_comparable | no_comparable_evidence | high |
| 221914217 | Application of Artificial Intelligence in Early Diagnosis of... | not_comparable | no_comparable_evidence | high |
| 8011525 | Prediction of Preterm Deliveries from EHG Signals Using Mach... | direct | agree | medium |
| 246908656 | Prediction of Preterm Delivery from Unbalanced EHG Database | direct | no_comparable_evidence | low |
| 255591419 | Enhancing classification of preterm-term birth using continu... | not_comparable | no_comparable_evidence | high |
| 278073696 | Electrohysterogram Data Augmentation Using Generative Advers... | conceptual | partially_agree | low |

## 3. Summary Counts

**By comparison_type:**
- direct: 3
- not_comparable: 4
- conceptual: 1

**By agreement:**
- no_comparable_evidence: 5
- conflict: 1
- agree: 1
- partially_agree: 1

## 4. Notable Finding

Candidate corpus_id **8011525** (Fergus et al., "Prediction of Preterm Deliveries from EHG Signals Using Machine Learning") was judged `direct`/`agree`, reporting 95% AUC.

Cross-referencing Vandewiele et al.'s own published reproduction results (github.com/GillesVandewiele/EHG-Oversampling), this paper corresponds to their "fergus2013" reproduced study, where they measured an `incorrect_oversampling_auc` (SMOTE applied before train/test split) of **0.967** and a corrected `with_oversampling_auc` (SMOTE applied after split) of only **0.597**. The 95% AUC this candidate reports is itself very likely inflated by the same leakage mechanism the target claim is being tested against — not an independent confirming result.

This is **not a flaw in the taxonomy judgment itself**: `comparison_type`/`agreement` correctly reflect what the snippet states, which contains no information about split methodology one way or the other. It demonstrates a structural limitation instead — cross-paper agreement signals can be confounded when a comparator study shares the same undetected methodological flaw as the target study.

This is exactly the rationale for the project's composite Research Rigor score design (Reproducibility + Replicability + Applicability triangulation): within-paper robustness checks (seed/split leakage detection) and cross-paper claim comparison are complementary, not substitutable; neither alone is sufficient.

## 5. Other Observations

- **corpus_id 278073696** was judged `conceptual`/`partially_agree` because it uses ADASYN, not SMOTE specifically — a precise distinction that shows the taxonomy can separate near-miss methodological variants from true direct comparators.
- **corpus_id 246908656** was judged `comparison_type=direct` but `agreement=no_comparable_evidence` — this demonstrates the code-enforced rule (empty `extracted_claim` forces `no_comparable_evidence` regardless of `comparison_type`) firing correctly on real data, not just in unit tests.
- The **first run** (pre-fix) had 5 of 8 candidates as title-only snippets with no body text, producing an inconsistent `comparison_type` judgment between two equally-uninformative candidates. The `has_body_text` fix and `exclude_corpus_id` fix together produced a fuller, more substantive candidate set in this second run: 8 of 8 candidates had real body/abstract text, and the target paper's own prior work was correctly excluded.

## 6. Caveats

- This is a single-case feasibility demonstration, not a validated precision/recall benchmark. No claim is made about the taxonomy's accuracy rate across a broader corpus.
- Real recall against Asta's corpus cannot be measured — no ground truth coverage exists for what Asta should have retrieved.
- The tool judges directional agreement as stated in retrieved snippets; it does not and cannot independently verify whether a comparator study's own reported metric is itself methodologically sound — see the Fergus/8011525 finding above.
