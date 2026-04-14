---
name: full-verification-team
description: "Verifies all factual claims, citations, and cross-references in the drought manuscript."
agents:
  - citation-verifier
  - pdf-reader
  - drought-reviewer
---

# Full Verification Team

## Purpose
Verify every verifiable claim in the drought manuscript.

## Verification Targets
1. **DOIs** (55 extracted, 37 confirmed, 17 pending alternate verification)
2. **Quantitative claims** (62 extracted — percentages, trends, statistics)
3. **Figure cross-references** (ED3 cited but no caption; Figs 5–9, ED1, ED2, ED6 never cited)
4. **Attributed findings** (claims attributed to specific papers — verify they appear in source)
5. **Statistical thresholds** (Cohen's d, p-values, significance levels)
6. **Period definitions** (1980–1999, 2000–2023 — check consistency across sections)

## Workflow
1. **citation-verifier** — DOI batch verification against Crossref
2. **pdf-reader** — Download and read key cited papers, verify attributed claims
3. **drought-reviewer** — Cross-check methodology claims against source datasets
