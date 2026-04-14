# Quality Gates
> Checkpoints enforced at each stage of research and manuscript production.
> Owner: QA Department | Version: 1.0

## Overview
Quality gates are non-negotiable checkpoints that must be passed before work
advances to the next pipeline stage. Each gate is owned by a specific agent
who is responsible for verification. Gates are designed to catch issues early,
when they are cheapest to fix.

---

## Gate 1: Research Completeness
**Owner:** Research Scout | **Applied after:** Deep Research Pipeline, Stage 2
**Blocks:** Document retrieval and synthesis

### Checklist
- [ ] Literature search covers 3+ databases (CrossRef, Semantic Scholar, Google Scholar minimum)
- [ ] Minimum 10 relevant papers identified (adjust threshold for niche topics)
- [ ] Key competing studies found (papers arguing against the thesis)
- [ ] Key supporting studies found (papers arguing for the thesis)
- [ ] Citation chaining performed on top 3–5 seed papers
- [ ] All DOIs verified via CrossRef API
- [ ] No fabricated or placeholder references
- [ ] Gap analysis completed — all sub-questions have at least some coverage
- [ ] Search strategy documented (keywords, filters, databases, date range)
- [ ] Duplicate papers removed across databases

### Failure Response
- Missing databases → expand search
- Insufficient papers → broaden search terms or relax inclusion criteria
- DOI failures → replace with verified alternatives
- Gaps identified → targeted follow-up search for specific sub-questions

---

## Gate 2: Draft Quality
**Owner:** Technical Editor | **Applied after:** Manuscript Pipeline, Phase 2
**Blocks:** Editing passes

### Checklist
- [ ] All required sections present per target venue format
- [ ] Every quantitative claim has a supporting citation
- [ ] Every qualitative claim attributed or marked as author interpretation
- [ ] All acronyms defined on first use
- [ ] No sentences exceeding 40 words
- [ ] No banned AI vocabulary (utilize, facilitate, leverage, delve, underscore,
      comprehensive, cutting-edge, groundbreaking, pivotal, noteworthy, etc.)
- [ ] All figures and tables referenced in the text
- [ ] All figures and tables have descriptive captions
- [ ] No [CITE_NEEDED] or [FIGURE_NEEDED] placeholders remain
- [ ] Word count within ±10% of target
- [ ] Consistent terminology throughout (no synonym cycling)
- [ ] Reference list complete and formatted per style guide

### Failure Response
- Missing sections → return to Manuscript Writer with specific gaps
- Citation gaps → route to Research Scout
- AI vocabulary → Technical Editor performs a targeted prose pass
- Placeholder remaining → route to responsible agent

---

## Gate 3: Scientific Rigor
**Owner:** Scientific Reviewer | **Applied after:** Manuscript Pipeline, Phase 3
**Blocks:** Verification phase

### Checklist
- [ ] All claims logically consistent (no internal contradictions)
- [ ] No circular reasoning (conclusions not assumed in premises)
- [ ] Effect sizes reported with confidence intervals where applicable
- [ ] P-values accompanied by effect size measures (not standalone)
- [ ] Statistical tests appropriate for the data type and design
- [ ] Sample sizes reported and adequate for claims made
- [ ] Limitations section genuinely engages with weaknesses
- [ ] Alternative explanations acknowledged where appropriate
- [ ] Causal language matches study design (correlation ≠ causation)
- [ ] Temporal consistency verified (no outdated claims presented as current)
- [ ] Generalizability claims appropriate for the sample/scope

### Failure Response
- Logical issues → return to Manuscript Writer with specific locations
- Statistical issues → route to Methodology Auditor
- Missing limitations → Domain Expert and Scientific Reviewer collaborate on additions
- Overclaimed results → Technical Editor and Scientific Reviewer revise language

---

## Gate 4: Pre-Submission
**Owner:** Research Director | **Applied after:** Manuscript Pipeline, Phase 4
**Blocks:** Final polish and submission

### Checklist
- [ ] All 3 verification tracks pass (see `workflows/verification-pipeline.md`)
- [ ] Formatting matches target journal/venue style guide exactly
- [ ] Author list and affiliations finalized and accurate
- [ ] Author contributions statement included (CRediT taxonomy if applicable)
- [ ] Data availability statement included
- [ ] Code availability statement included (if applicable)
- [ ] Conflict of interest declaration included
- [ ] Funding acknowledgment included
- [ ] Ethics approval statement included (if applicable)
- [ ] Word count within venue limits
- [ ] Abstract word count within venue limits
- [ ] Keywords selected per venue guidelines
- [ ] Supplementary materials complete and referenced
- [ ] Cover letter drafted (if required by venue)
- [ ] All co-authors have reviewed and approved the manuscript

### Failure Response
- Verification track failure → return to relevant pipeline stage
- Formatting issues → Technical Editor applies fixes
- Missing statements → Research Director drafts or assigns
- Co-author approval pending → hold submission until obtained

---

## Gate Application Rules

1. **Gates are sequential** — Gate N must pass before Gate N+1 is evaluated
2. **Gates are mandatory** — no gate may be skipped, even under time pressure
3. **Gates are documented** — pass/fail results logged in the project record
4. **Partial passes** — if most items pass but 1–2 minor items fail, the gate
   owner may grant a conditional pass with specific fix-by requirements
5. **Repeated failures** — if a gate fails 3 times, escalate per
   `protocols/escalation.md` Level 4
