# Triple-Check Protocol
> Quality assurance gate applied before any major deliverable is finalized.
> Owner: QA Department | Version: 1.0

## Overview
Every significant deliverable must pass three independent reviews before approval.
Each review checks from a different vantage point: self, peer, and cross-department.
This protocol is invoked by other workflows at their quality gate stages.

**Estimated Duration:** 0.5 session
**Applies To:** Draft sections, literature reviews, data analyses, final manuscripts

---

## Check 1: Self-Review
**Performed by:** The original agent that produced the work

### Process
1. Re-read the deliverable against the original task criteria
2. Run all applicable automated checks:
   - AI vocabulary scan (flag: utilize, facilitate, leverage, delve, etc.)
   - Sentence length check (flag any sentence > 40 words)
   - DOI format validation (if references present)
   - Acronym usage (defined on first use?)
   - Consistency check (terminology, formatting)
3. Compare output against the expected deliverable specification
4. Document any known weaknesses or uncertainties

### Output
- Self-review checklist (all items pass/fail)
- Known issues list with severity
- Confidence assessment (High / Medium / Low)

---

## Check 2: Peer Review
**Performed by:** A different agent from the same department

### Assignment
- Manuscript Writer ↔ Technical Editor
- Research Scout ↔ Research Director
- Data Analyst ↔ Methodology Auditor
- Scientific Reviewer ↔ Citation Verifier

### Process
1. Independent review without seeing the self-review results
2. Focus areas:
   - **Completeness:** Does the deliverable address all requirements?
   - **Accuracy:** Are facts, figures, and citations correct?
   - **Standards adherence:** Does it follow institute conventions?
   - **Clarity:** Is the content clear and well-organized?
3. Note any issues not caught by self-review

### Output
- Peer review report
- Issue list with severity (Critical / Major / Minor)
- Recommendation: APPROVE / REVISE / REJECT

---

## Check 3: Cross-Department Review
**Performed by:** QA Department (Scientific Reviewer or Methodology Auditor)

### Process
1. Review from an external quality perspective
2. Focus areas:
   - **Reasoning quality:** Are arguments logically sound?
   - **Evidence strength:** Do claims have adequate support?
   - **Methodology:** Are methods appropriate and correctly applied?
   - **Consistency:** Does this deliverable align with other project outputs?
3. Apply the relevant quality gate from `protocols/quality-gates.md`

### Output
- QA review report
- Issue list with severity
- Recommendation: APPROVE / REVISE / REJECT

---

## Resolution Protocol

### Decision Matrix

| Check 1 | Check 2 | Check 3 | Action |
|---------|---------|---------|--------|
| PASS    | APPROVE | APPROVE | **Approve** — deliverable is finalized |
| PASS    | APPROVE | REVISE  | **Fix & re-check** Track 3 only |
| PASS    | REVISE  | APPROVE | **Fix & re-check** Track 2 only |
| PASS    | REVISE  | REVISE  | **Return** to originator with consolidated feedback |
| FAIL    | any     | any     | **Return** to originator — self-review must pass first |
| any     | REJECT  | any     | **Return** with major revisions required |
| any     | any     | REJECT  | **Return** with major revisions required |
| any     | REJECT  | REJECT  | **Escalate** to Research Director |

### Escalation
- If a deliverable fails the triple-check **twice**, escalate to Research Director
- Research Director may reassign the task, adjust scope, or convene a brainstorm session
- Critical issues (fabricated data, logical fallacies, phantom citations) trigger
  immediate escalation regardless of other check results

### Iteration Limits
- Maximum 3 revision cycles through triple-check
- After 3 cycles, Research Director intervenes to diagnose the root cause
