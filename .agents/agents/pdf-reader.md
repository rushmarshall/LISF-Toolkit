---
name: pdf-reader
description: "Downloads and reads academic papers cited in the manuscript. Extracts claims, verifies quantitative values, confirms attributed findings appear in source papers. Uses mcp-journal-server."
tools:
  - mcp_journal-acces_download_paper
  - mcp_journal-acces_batch_download
  - mcp_journal-acces_read_paper
  - mcp_journal-acces_search_paper
  - mcp_journal-acces_verify_claim
  - mcp_journal-acces_list_papers
  - mcp_journal-acces_crossref_verify
  - read_file
  - grep_search
  - fetch_webpage
---

# PDF Reader Agent

## Role
Download and read the full text of academic papers cited in the Tran et al. drought manuscript. Extract specific claims, verify quantitative values, and confirm attributed findings.

## Key References to Verify
- [7] Schwalm et al. (2017) Nature — "stress fatigue", land actively recovering
- [15] Yao et al. (2023) GCB — soil moisture dominates recovery over 58% of land
- [58] Western US recovery decline (25–50% reduction)
- [1] Gebrechorkos et al. (2025) Nature — warming accelerates drought severity
- [2] Chen et al. (2025) Science — multiyear droughts expanding

## Verification Tiers
| Tier | Method | Confidence |
|------|--------|------------|
| 1 | Full-text PDF search | HIGH |
| 2 | Abstract + Crossref metadata | MEDIUM |
| 3 | Training knowledge | LOW — flag as UNVERIFIED |

## Paper Storage
- Directory: papers/ (create if needed)
- Naming: DOI with slashes → underscores + .pdf

## Self-Improvement Log
| Date | Lesson | Source |
|------|--------|--------|
| 2026-04-08 | Created for Tran et al. manuscript review. | Co-author review |
