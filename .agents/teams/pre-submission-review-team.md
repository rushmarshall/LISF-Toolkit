---
name: pre-submission-review-team
description: "Full manuscript review pipeline for co-author review of Nature drought manuscript. Runs drought-reviewer → writing-reviewer → citation-verifier → ultra-thinker sequentially."
agents:
  - drought-reviewer
  - writing-reviewer
  - citation-verifier
  - ultra-thinker
  - manuscript-editor
---

# Pre-Submission Review Team

## Purpose
Execute a complete co-author review of Tran et al. "Accelerating recurrence weakens drought recovery across 59% of global land" prior to Nature submission.

## Workflow
1. **drought-reviewer** — Domain accuracy: verify claims, check methodology, flag inconsistencies
2. **writing-reviewer** — Language quality: AI markers, grammar, flow, Nature style compliance
3. **citation-verifier** — Reference integrity: DOI verification, author/year match, sequential numbering
4. **ultra-thinker** — Logical stress-test: argument gaps, hidden assumptions, reviewer anticipation
5. **manuscript-editor** — Apply all approved edits to review notebook

## Deliverables
- Annotated review in review.ipynb with [EDIT], [COMMENT], [QUESTION], [PRAISE] tags
- DOI verification report
- Priority action items ranked by severity

## Exit Criteria
- All sections reviewed
- All quantitative claims cross-checked
- DOIs verified
- No unresolved internal contradictions
