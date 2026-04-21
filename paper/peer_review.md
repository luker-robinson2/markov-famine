# Peer Review — *Predicting Food Security Transitions in the Horn of Africa: A Non-Homogeneous Markov Chain Approach with Regularized Ensemble Machine Learning*

**Reviewer:** Claude (Opus 4.7) · **Date:** 2026-04-20 · **Recommendation:** Major revision

---

## Summary

The paper wraps an XGBoost IPC-phase classifier in a non-homogeneous Markov chain (NHMC) formalism, trains it on a CHIRPS/MODIS/ERA5 + FEWS-NET panel for 37 admin-1 regions across Kenya/Ethiopia/Somalia (2015–2024), and reports R² = 0.865 on a 2024 test year with a <1% train-test gap. A complementary DeltaPredictor models phase *changes* and detects 52% of the 21 transitions in the test set. The codebase is open-source, the temporal split is clean, and the feature engineering (VCI, SPI-3, cumulative deficits) is consistent with the applied climate-drought literature.

The paper is **well-written, technically tidy, and reproducible**, but two framing choices — (a) headlining R² on a near-persistent ordinal target and (b) motivating a 3–6-month operational lead time while evaluating only one-step-ahead — seriously undercut the stated impact. The actually interesting result (the DeltaPredictor) is buried. The recommended revision is largely a **reframing and re-evaluation** exercise, not new modeling.

---

## Major Concerns

### 1. R² on an ordinal 4-class target is the wrong metric, and "zero overfit" is a near-tautology under persistence

The paper's headline numbers — **Persistence R² = 0.921, PhasePredictor R² = 0.865, train-test gap < 1%** — are mechanically driven by the fact that ~95% of region-months are unchanged. When the target barely moves, predicting `S_{t+1} = S_t` scores near-perfectly on any interval-scale metric, and any model regularized toward persistence inherits both the high R² and, by construction, a vanishing generalization gap. A Phase-2-heavy, low-variance target cannot meaningfully overfit when the model has been explicitly tuned (depth 3, α = 1, λ = 5, subsample 0.6) to approach the persistence predictor.

This means the "zero overfitting" claim in §5.2 and the abstract is **technically correct but uninformative** — it describes a property of the regularization schedule, not of the model's forecast skill. More importantly, R² treats the IPC scale as interval (distance from Minimal to Stressed ≡ distance from Crisis to Emergency), which the IPC manual itself does not support.

The food-security-forecasting literature has converged on **chance- and persistence-corrected skill scores**:

- **Quadratic Weighted Kappa** (already reported — should be foregrounded).
- **F1-macro** — the Andrée et al. (World Bank, 2022 *Science Advances*) and Busker et al. (2024 *Earth's Future*) standard; handles class imbalance honestly.
- **Ranked Probability Skill Score (RPSS)** vs. a persistence or climatology reference (Weigel et al. 2007 *MWR*; Zhou et al. 2023 *Sci Reports* argues this explicitly for food insecurity).
- **Heidke Skill Score** (Mason & Stephenson 2008) for categorical forecasts.

**Action:** Replace R² as the lead metric with F1-macro and RPSS-vs-persistence, and report QWK alongside. Expect RPSS to be near zero or negative on the phase-level task — and that is the correct finding. The DeltaPredictor's 52% transition recall is where positive skill actually lives; let the numbers tell that story.

### 2. Bait-and-switch on lead time

The introduction (§1, ¶2) motivates the problem with a **3–6-month operational lead time** required by humanitarian logistics, then benchmarks against FEWS-NET's 78% accuracy at the 3-month horizon (Bertetti et al. 2024). But every result in §5 is **one-step-ahead** (monthly). The iterative multi-step construction in Eq. 4 is described but never evaluated.

This is the most common failure mode in this literature, and reviewers outside the ML community — the humanitarian practitioners you cite — will catch it immediately. Comparable papers evaluate across horizons and report honest degradation:

- Busker et al. 2024 evaluates 1–12 months and shows R² collapses past 3 months.
- Andrée et al. 2022 evaluates 4, 8, 12 months with F1-macro.
- Westerveld et al. 2021 evaluates 1–12 months.
- Martini et al. 2023 (*Nature Food*) is explicitly a *nowcast* and should not be cited as a lead-time result.

**Action:** Report t+1, t+3, t+6 forecasts on the 2024 test year using the NHMC multi-step product (Eq. 4), with Monte Carlo CIs as already described in §4.1. If t+6 R² collapses (it will), that *is* the result, and you have a principled explanation from the product-of-stochastic-matrices structure.

### 3. The DeltaPredictor is the contribution — foreground it

The DeltaPredictor detects 52% of phase changes and 67% of worsening transitions against a persistence baseline of 0%. For an early-warning system, **this is the paper's result**. Yet it is introduced third after the PhasePredictor, framed as auxiliary, and then folded into a blended Hybrid that dilutes its strongest feature (transition recall 0.19 vs 0.52).

The Hybrid also shows a moderate overfit gap (+6.5% accuracy, +9.9% R²) — not a triumph to feature.

**Action:** Restructure the paper around the transition-detection contribution:
- Abstract: lead with transition recall, not R².
- §4: present DeltaPredictor first, justify the δ ∈ {−2,…,+2} framing up front.
- §5: introduce Table 2 with transition detection as column 1 (not column 5) and drop R² to a supporting column.
- Drop or relegate the Hybrid unless you can show it Pareto-dominates on a skill-score basis.

### 4. Row-wise per-origin-state XGBoost is statistically shaky

Equation 3 trains a separate classifier per origin state. With 37 regions × 120 months × ~95% persistence in states 1–2, **the Phase-4 (Emergency) classifier is fit on a handful of rows** — effectively unidentified. This is a known issue in the multi-state / covariate-dependent Markov literature (Jackson's `msm`; Meira-Machado 2009 *Stat Methods Med Res*; Bartolucci & Farcomeni 2019 *Stat Methods Appl*), which generally recommends parameter sharing via a **multinomial logit with state × covariate interactions** or a **sequential ordinal model with shared base learners**.

The paper does not report row-wise sample sizes for the four origin-state classifiers, nor per-origin calibration. Readers cannot tell whether the Emergency row of **P**_t is a genuine estimate or essentially a prior.

**Action:** Either (a) add a short robustness check — report n-per-origin, and fit a single multinomial model with origin-state as a feature as a sensitivity comparison — or (b) acknowledge this as a named limitation citing the survival-model literature.

### 5. Single test year limits the claim

n = 407 region-months in a single (recovery) year is thin evidence, particularly when 2024 was relatively favorable (crisis fraction 48.4%, down from 2022's 61%). The model's behavior under a worsening year is unobserved. The 2020–2023 drought — the most informative regime — sits in the training set.

**Action:** Rolling-origin evaluation (train through Dec 2020, test 2021; train through 2021, test 2022; …) with metrics aggregated across the four most recent years. The existing pipeline already supports this; it is a few hours of compute, not a re-design. This strengthens every claim in the paper at very low cost.

---

## Minor Concerns

- **Abstract, line 3:** "demonstrating zero overfitting" — delete. Replace with transition-recall headline.
- **§1, ¶3 (contributions):** Contribution (ii) ("train-test gap < 1%") is a modeling artifact, not a contribution. Merge into (iii) and make the contribution "transition-focused evaluation against a persistence baseline."
- **§4.2:** The 15× upweight on non-zero deltas is introduced without justification. The imbalanced-classification literature (He & Garcia 2009; Johnson & Khoshgoftaar 2019) recommends tuning the weight to maximize a target metric (QWK, F1-macro) on a rolling validation fold. State how 15 was chosen — if it was eyeballed, say so and add a sensitivity sweep (weights 1, 5, 15, 50) in an appendix.
- **§4.1, Eq. 4:** π_t is called "a one-hot encoding of the current observed phase." For genuine probabilistic forecasting, this should be initialized to the model's *predicted* distribution over the current phase (or the empirical conditional). The one-hot start understates uncertainty.
- **§4.3:** "Overfit Gap" is defined relative to accuracy; the text later uses R² gap. Pick one and be consistent, or report both explicitly in Table 3 headers.
- **§5.5, transition matrix **P̂**:** 4×4 matrix rows do not sum exactly to 1 as printed (row 3: 0.000+0.054+0.912+0.033 = 0.999; row 4: 0.000+0.007+0.094+0.899 = 1.000). Minor but fix.
- **§5.6:** Report country-level transition recall, not just accuracy/R². Somalia's R² = 0.713 with 90.9% accuracy is another case of the R²-vs-accuracy axis misleading the reader.
- **§5.7 "Temporal Patterns":** This subsection is descriptive and belongs in §3 (Data) or §2.4 (Climate Drivers). It does not report model results.
- **References:** Busker 2024's full citation is incomplete (`et al.` in author list). Fix.
- **Figures:** The paper contains **no figures**. A phase time series per country, a confusion matrix, and a SHAP summary plot would each carry 500+ words of text. Add at least (i) a mean-phase time series panel per country, (ii) a normalized confusion matrix for the DeltaPredictor, (iii) a SHAP bar plot. This is the single largest readability gain available.

---

## Readability & Writing

Strengths: clear prose, consistent notation, strong topic sentences, the Markov theory in §2.2–§2.3 is presented cleanly without padding.

Weaknesses:
- The **abstract is 290 words**; target 150–200. Cut the codebase URL, the drought scenario sentence, and "demonstrating zero overfitting."
- **§2.2 (Chapman-Kolmogorov)** and the stationary-distribution paragraph are standard and can be cut to three sentences or moved to an appendix. You never use `P^(n) = P^n` in Results; the NHMC product form in Eq. 2 is what matters.
- **§4.2** uses "essentially zero overfitting" / "slightly refined version of persistence" / "aggressive regularization successfully prevents memorization" — three phrasings of the same thing in one subsection. Pick one.
- **Passive voice** dominates §4–5 ("is assembled," "are evaluated," "are obtained"). Active voice tightens these significantly.

---

## What to Keep

- The NHMC formalism (Eq. 1–2) is an honest contribution — most ML-for-food-security papers drop the Markov structure entirely.
- Strict temporal split with fitted statistics (VCI bounds, SPI gamma params) computed from training data only — cleanly done and not universal in this literature.
- The "ratchet effect" observation in §5.5 is genuinely interesting and could be a separate paragraph motivating the DeltaPredictor (asymmetric transition probabilities justify modeling Δ explicitly).
- Open-sourced code and data pipeline.

---

## Recommended Revision Path (in order)

1. **Re-evaluate.** Add F1-macro, QWK (already present), and RPSS-vs-persistence as primary metrics. Run t+1, t+3, t+6 multi-step forecasts via Eq. 4. Run rolling-origin CV over 2021–2024.
2. **Reframe.** Lead with transition detection; demote R² and the "zero overfit" claim; restructure §5 around skill-vs-persistence.
3. **Shorten.** Cut §2.2 redundancy, abstract by 30%, and consolidate §4.2's three restatements of regularization.
4. **Add figures.** Time series, confusion matrix, SHAP plot.
5. **Discuss the row-wise formulation** limitation explicitly, with per-origin sample sizes.

With these changes the paper goes from "an XGBoost model close to persistence" (the current reading) to "a calibrated transition-detection system with honest multi-horizon skill scores" — which is both a stronger statistical claim and a more useful humanitarian one.

---

## Key References for the Revision

- Busker, T. et al. (2024). Predicting food-security crises in the Horn of Africa using ML. *Earth's Future* 12(8). — benchmark at matching horizons.
- Andrée, B. P. J. et al. (2022). Predicting food crises. *Science Advances* 8(40). — F1-macro at 4/8/12-month horizons, cross-country.
- Zhou, Y. et al. (2023). On the forecastability of food insecurity. *Scientific Reports* 13. — argues explicitly for RPSS vs. persistence in this domain.
- Weigel, A. P. et al. (2007). The discrete Brier and ranked probability skill scores. *Monthly Weather Review* 135. — canonical RPSS reference.
- Mason, S. J. & Stephenson, D. B. (2008). *Forecast Verification: A Practitioner's Guide*, 2nd ed., Wiley. — HSS and categorical verification.
- Foini, P. et al. (2024). Forecasting trends in food security with real-time data. *Nature Communications Earth & Environment* 5. — reservoir-computing baseline at 60-day horizon.
- Bartolucci, F. & Farcomeni, A. (2019). A shared-parameter continuous-time hidden Markov and survival model. *Statistical Methods & Applications* 28. — covariate-dependent Markov chains with shared parameters (alternative to row-wise XGBoost).
- Johnson, J. M. & Khoshgoftaar, T. M. (2019). Survey on deep learning with class imbalance. *J. Big Data* 6. — justifies tuning δ-weighting rather than picking 15× by inspection.
