# Verification Pipeline
> Orthogonal triple-check verification — each track checks a different dimension.
> Owner: QA Department | Version: 1.0

## Overview
This pipeline provides independent, parallel verification across three orthogonal
dimensions: reasoning quality, evidence integrity, and methodological soundness.
Each track is staffed by different agents to avoid confirmation bias. Tracks run
concurrently and converge at the Research Director for final adjudication.

**Estimated Duration:** 1 session
**Required Agents:** Scientific Reviewer, Citation Verifier, PDF Reader,
  Methodology Auditor, Data Analyst, Research Director

---

## Track A: Reasoning Review
**Lead:** Scientific Reviewer

### Purpose
Evaluate the logical and intellectual architecture of the document, independent
of whether the evidence or methodology is correct.

### Checklist
1. **Logical consistency**
   - [ ] Arguments follow from premises to conclusions
   - [ ] No internal contradictions between sections
   - [ ] Claims are proportional to evidence presented

2. **Circular reasoning detection**
   - [ ] Conclusions are not assumed in premises
   - [ ] Definitions are not self-referential
   - [ ] Supporting arguments are independent of the thesis

3. **Narrative coherence**
   - [ ] Story flows logically from introduction to conclusion
   - [ ] Each section builds on the previous one
   - [ ] Transitions between ideas are justified, not just grammatical

4. **Intellectual architecture**
   - [ ] The argument structure is sound (deductive, inductive, or abductive)
   - [ ] Alternative interpretations are acknowledged
   - [ ] Limitations are genuinely engaged with, not token gestures
   - [ ] The contribution is clearly articulated

### Output
- Reasoning score: PASS / CONDITIONAL PASS / FAIL
- Issue list with severity (Critical / Major / Minor)
- Specific locations of each issue

---

## Track B: Evidence Review
**Lead:** Citation Verifier + PDF Reader

### Purpose
Verify that every piece of evidence cited actually exists, says what the
document claims it says, and is used appropriately.

### Checklist
1. **DOI resolution**
   - [ ] Every DOI in the reference list resolves via CrossRef API
   - [ ] Resolved metadata matches what's cited (authors, year, title)
   - [ ] No DOIs point to retracted papers (check Retraction Watch)

2. **Claim-source alignment**
   - [ ] Each in-text citation supports the specific claim it's attached to
   - [ ] Cited passages actually say what the document claims
   - [ ] Context of the original source is preserved (not cherry-picked)

3. **Temporal consistency**
   - [ ] No claims about "current" state based on outdated sources (>5 years for fast-moving fields)
   - [ ] Chronological claims match actual publication dates
   - [ ] Superseded findings are not presented as current consensus

4. **Phantom citation detection**
   - [ ] No references that appear real but don't exist
   - [ ] No references with slightly altered titles or author names
   - [ ] No self-citations disguised as independent support

### Output
- Evidence score: PASS / CONDITIONAL PASS / FAIL
- DOI verification log (resolved / failed / flagged)
- Claim-alignment exceptions
- Phantom citation alerts

---

## Track C: Methodology Review
**Lead:** Methodology Auditor + Data Analyst

### Purpose
Evaluate whether the methods described are appropriate, correctly executed,
and accurately reported.

### Checklist
1. **Statistical method appropriateness**
   - [ ] Statistical tests match the data type and research design
   - [ ] Sample size adequate for the claims made
   - [ ] Assumptions of statistical tests are met or acknowledged
   - [ ] Multiple comparison corrections applied where needed

2. **Effect size interpretation**
   - [ ] Effect sizes reported alongside p-values
   - [ ] Confidence intervals provided
   - [ ] Effect sizes interpreted in practical (not just statistical) terms
   - [ ] Comparisons to established benchmarks where available

3. **Reproducibility assessment**
   - [ ] Methods described in sufficient detail to replicate
   - [ ] Data sources and access methods documented
   - [ ] Analysis pipeline described step-by-step
   - [ ] Software and version information provided

4. **Uncertainty quantification**
   - [ ] Sources of uncertainty identified
   - [ ] Error propagation handled appropriately
   - [ ] Sensitivity analyses conducted for key parameters
   - [ ] Results not presented with false precision

### Output
- Methodology score: PASS / CONDITIONAL PASS / FAIL
- Issue list with severity and recommendations
- Reproducibility rating (High / Medium / Low)

---

## Convergence
**Lead:** Research Director

### Process
1. Collect reports from all three tracks
2. Classify the overall result:

| Track A | Track B | Track C | Decision |
|---------|---------|---------|----------|
| PASS    | PASS    | PASS    | **APPROVE** — proceed to next pipeline stage |
| PASS    | PASS    | COND    | **APPROVE WITH FIXES** — minor methodology fixes |
| COND    | PASS    | PASS    | **APPROVE WITH FIXES** — minor reasoning fixes |
| FAIL    | any     | any     | **REJECT** — return to writing/editing |
| any     | FAIL    | any     | **REJECT** — return to citation/evidence review |
| any     | any     | FAIL    | **REJECT** — return to methodology/analysis |
| COND    | COND    | COND    | **ESCALATE** — Research Director deep review |

3. For rejections, compile specific fix list from relevant track report
4. For escalations, Research Director reviews all three reports and decides path

### Output
- **Verification Report**
  - Overall status: APPROVED / APPROVED WITH FIXES / REJECTED / ESCALATED
  - Track A summary and score
  - Track B summary and score
  - Track C summary and score
  - Required actions (if any)
  - Assigned agents for fixes (if any)
