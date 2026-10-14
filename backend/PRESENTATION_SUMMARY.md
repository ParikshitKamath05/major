# DocuMind Presentation Summary — Evaluation & Calibration Analysis

**Evaluation Dataset:** SQuAD 2.0 Balanced Sample (Seed=42, N=148 evaluated, 2 failed due to rate-limit exhaustion, 98.67% success rate)  
**Evaluation File:** [`eval_results_squad2_20261011_163354.json`](file:///C:/major/backend/eval_results_squad2_20261011_163354.json)  
**Detailed Breakdown:** [`breakdown_answerable.json`](file:///C:/major/backend/breakdown_answerable.json)  

---

## 1. Headline Metrics Table

| Metric | Measured Value | Self-RAG LLaMA-7B | Self-RAG LLaMA-13B | Notes |
|:---|:---:|:---:|:---:|:---|
| **Accuracy** | **0.5000** (50.0%) | 0.7000 (70.0%) | 0.7500 (75.0%) | Self-RAG from Asai et al. 2023 (full val set) |
| **ECE (10 bins)** | **0.2581** (25.8%) | *NR* | *NR* | Expected Calibration Error |
| **Brier Score** | **0.3122** | *NR* | *NR* | Mean squared error of probabilities |
| **AUROC (+conf)** | **0.4545** | *NR* | *NR* | Below random (0.50) → anti-informative |
| **AUROC (-conf)** | **0.5455** | *NR* | *NR* | Higher confidence correlates with incorrectness |

*NR = Not Reported by Self-RAG authors (Self-RAG does not expose calibration infrastructure).*

---

## 2. Answerable vs. Unanswerable Breakdown

| Cohort | N | Correct | Incorrect | Accuracy | Mean Confidence |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Answerable Questions** | 74 | 0 | 74 | **0.0000** (0.0%) | 0.6188 |
| **Unanswerable Questions** | 74 | 74 | 0 | **1.0000** (100.0%) | 0.6002 |
| **Overall Combined** | **148** | **74** | **74** | **0.5000** (50.0%) | **0.6095** |

### Unanswerable Question Analysis
- **Non-empty predicted answer count:** **74 / 74 (100%)**
- **Abstention Phrase Match:** 100% of responses contained valid abstention indicators (`"I cannot answer this question reliably. The provided documents do not contain sufficient information..."`).
- **Substantive Hallucinations:** **0 / 74 (0.0%)**. The system never fabricated plausible facts when information was missing.

---

## 3. Bimodality Verification Result

- **Count with confidence in `[0.4, 0.7)`:** **0 examples (CONFIRMED)**
- **Count with confidence `< 0.4`:** **51 examples (34.46%)** (Cluster around ~0.35–0.38)
- **Count with confidence `≥ 0.7`:** **97 examples (65.54%)** (Cluster around ~0.73–0.78)

**Mechanism:** This confirms the exact reason for the flat abstention table between thresholds 0.4 and 0.7 (coverage remains strictly flat at 65.54% and accuracy at 47.42%). The evidential fusion Dirichlet distribution concentrates evidence into two discrete regimes: one where semantic entropy is high/candidate agreement is absent (mass near ~0.36), and one where self-consistency saturates (mass near ~0.75).

---

## 4. Honest Interpretation (3 Core Takeaways)

1. **Anti-Informative Confidence Signal Out-of-Domain (AUROC = 0.4545):**  
   The confidence score is negatively correlated with true accuracy. The mean predictive confidence for *incorrect* predictions (0.6188) is slightly higher than for *correct* predictions (0.6002). In an out-of-domain evaluation setting where the target knowledge base does not match the retrieved document index, the system's confidence cannot be treated as an indicator of factual correctness.

2. **Self-Consistency Overweighting as the Suspected Mechanism:**  
   When candidates are generated from an LLM asked to answer from limited or irrelevant context, the candidates often agree on similar phrasing or generic fallback reasoning. This causes high candidate similarity (self-consistency $\approx 0.85\text{--}1.0$), which EDL rewards with high belief mass ($b_1$) despite low retrieval similarity, artificially elevating predictive confidence to $\approx 0.73\text{--}0.78$.

3. **Current Abstention Policy Would Hurt Accuracy if Applied Naively:**  
   At threshold 0.0, overall accuracy is 50.0% (100% coverage). At threshold 0.5, coverage drops to 65.5% while accuracy drops to 47.42%. Filtering out "low-confidence" predictions removes a higher proportion of correct unanswerable queries (28 removed) than incorrect answerable queries (23 removed), degrading selective accuracy below random guessing.

---

## 5. Limitations

- **Sample Size ($N=148$):** Capped at 150 balanced queries due to free-tier API rate pacing constraints; 2 queries failed from backoff limit exhaustion.
- **Out-of-Domain (OOD) Target Corpus:** SQuAD 2.0 evaluation queries span diverse Wikipedia topics, whereas the indexed vector store contains a single reference document (`bb34c886fe7f`). Hence, retrieval relevance scores were consistently low, triggering the escalation and abstention cascade for all queries.
- **Conformal Calibration Fit Prior to Backend Swap:** The 42 calibration points in `logs/conformal_calibration.jsonl` were collected under an earlier run configuration. Because conformal prediction requires exchangeability, testing on OOD SQuAD queries introduces distribution shift.
- **Free-Tier API Rate Limits:** Pacing constraints (6.0s sleep per query) limited sample throughput to ~3.5 queries/minute.

---

## 6. Generated Presentation Artifacts

- [`confidence_histogram.png`](file:///C:/major/backend/confidence_histogram.png): Overlaid histograms for correct vs. incorrect answers with the 0.5 threshold and the `[0.4, 0.7)` bimodal gap.
- [`risk_coverage.png`](file:///C:/major/backend/risk_coverage.png): Risk-coverage sweep from threshold 0 to 1, showing anti-informative degradation below the 50% baseline.
- [`reliability_diagram.png`](file:///C:/major/backend/reliability_diagram.png): 10-bin calibration curve showing ECE = 0.2581 with shaded calibration error gap.
- [`documind_vs_selfrag.png`](file:///C:/major/backend/documind_vs_selfrag.png): Bar comparison of DocuMind (50.0%) vs Self-RAG 7B (70%) and 13B (75%).
- [`documind_vs_selfrag_metrics.png`](file:///C:/major/backend/documind_vs_selfrag_metrics.png): 4-panel comparison highlighting that DocuMind is the only system reporting calibration metrics.
