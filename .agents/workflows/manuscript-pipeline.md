# Manuscript Production Pipeline
> End-to-end manuscript production from planning through submission-ready draft.
> Owner: Writing Department | Version: 1.0

## Overview
This pipeline manages the full lifecycle of producing an academic manuscript.
It coordinates across all departments: research, writing, data science, QA, and
domain expertise. The pipeline enforces three editing passes and a verification
phase before sign-off.

**Estimated Duration:** 3–8 sessions depending on manuscript complexity
**Required Agents:** Research Director, Manuscript Writer, Technical Editor, Scientific Reviewer,
  Domain Expert, Data Analyst, Research Scout, Citation Verifier, Methodology Auditor

---

## Phase 1: Planning
**Duration:** 1 session

### Step 1.1: Section Assignment
**Lead:** Research Director
- Review the research question, findings, and target venue
- Determine manuscript type (original research, review, methods paper, etc.)
- Assign sections to responsible agents
- Define target word count per section
- Set timeline with milestones

### Step 1.2: Outline Creation
**Lead:** Manuscript Writer
- Create detailed outline with:
  - Section headings and subheadings
  - Key points per section (3–5 bullet points each)
  - Planned figures and tables with descriptions
  - Target citations per section
- Map evidence from deep-research-pipeline outputs to outline sections

### Step 1.3: Structure Validation
**Lead:** Domain Expert
- Validate the outline against target venue requirements
- Confirm logical flow of the argument
- Verify that all necessary sections are present
- Check that the narrative structure serves the research question

### Phase 1 Output
- Approved outline document
- Section assignments with responsible agents
- Figure/table plan
- Timeline with milestones

### Phase 1 Gate
- [ ] Outline approved by Research Director and Domain Expert
- [ ] All sections assigned
- [ ] Target venue requirements documented

---

## Phase 2: Drafting
**Duration:** 2–4 sessions

### Step 2.1: First Draft
**Lead:** Manuscript Writer
- Write each section following the approved outline
- Insert placeholder citations where needed (marked as [CITE_NEEDED])
- Flag sections needing data/figures (marked as [FIGURE_NEEDED])
- Maintain consistent voice and terminology throughout
- Target 80% of final word count

### Step 2.2: Figures and Tables
**Lead:** Data Analyst
- Generate all planned figures and tables
- Ensure figures are publication-quality
- Write descriptive captions
- Create any supplementary materials
- Verify all data representations are accurate

### Step 2.3: Citation Gap Fill
**Lead:** Research Scout
- Resolve all [CITE_NEEDED] placeholders
- Find supporting references for under-cited claims
- Verify citation relevance and recency
- Update reference list

### Phase 2 Output
- Complete first draft with all sections
- All figures and tables
- Complete reference list (no placeholders remaining)

### Phase 2 Gate
- [ ] All sections drafted
- [ ] No [CITE_NEEDED] or [FIGURE_NEEDED] placeholders remain
- [ ] All figures/tables generated and captioned
- [ ] Reference list complete

---

## Phase 3: Editing (3-Pass)
**Duration:** 1–2 sessions

### Pass 1: Structural Review
**Lead:** Technical Editor
**Focus:** Document architecture and flow

Checklist:
- [ ] All required sections present and in correct order
- [ ] Headings follow target venue hierarchy
- [ ] Logical flow between paragraphs and sections
- [ ] Introduction sets up the research question effectively
- [ ] Discussion addresses all results
- [ ] Conclusion does not introduce new information
- [ ] Abstract accurately summarizes the paper
- [ ] Word count within target range

**Output:** Structural edit report with specific change requests

### Pass 2: Prose Review
**Lead:** Technical Editor
**Focus:** Language quality and readability

Checklist:
- [ ] No AI vocabulary markers (utilize → use, facilitate → help, etc.)
- [ ] No sentences exceeding 40 words
- [ ] All acronyms defined on first use
- [ ] Consistent terminology throughout
- [ ] Active voice preferred over passive
- [ ] Paragraph length appropriate (3–7 sentences)
- [ ] Transitions between ideas are smooth
- [ ] No hedge-stacking (slightly somewhat possibly)

**Output:** Line-edit report with specific corrections

### Pass 3: Accuracy Review
**Lead:** Scientific Reviewer
**Focus:** Scientific claims and reasoning

Checklist:
- [ ] Every quantitative claim has a supporting citation or data reference
- [ ] Statistical results correctly reported (test, df, p-value, effect size)
- [ ] Claims do not overstate the evidence
- [ ] Limitations properly acknowledged
- [ ] No logical fallacies or circular reasoning
- [ ] Cause-effect language appropriate for study design
- [ ] Competing explanations addressed

**Output:** Accuracy review report with required corrections

### Phase 3 Gate
- [ ] All three editing passes complete
- [ ] All critical and major issues resolved
- [ ] Minor issues logged and addressed

---

## Phase 4: Verification
**Duration:** 1 session

### Track 4.1: Citation Verification
**Lead:** Citation Verifier
- Verify every DOI resolves correctly
- Confirm author names, years, and titles match CrossRef metadata
- Check that in-text citations match reference list entries
- Detect any fabricated or phantom references

### Track 4.2: Methodology Audit
**Lead:** Methodology Auditor
- Verify statistical claims are valid
- Check that methods described match results reported
- Confirm effect sizes and confidence intervals are correct
- Validate any mathematical formulas or derivations

### Track 4.3: Domain Accuracy Check
**Lead:** Domain Expert
- Verify domain-specific claims and terminology
- Confirm the paper's position relative to current field consensus
- Check that domain conventions are followed
- Validate any domain-specific methodological choices

### Phase 4 Gate
- [ ] All DOIs verified
- [ ] No fabricated references
- [ ] Statistical claims validated
- [ ] Domain accuracy confirmed

---

## Phase 5: Polish
**Duration:** 0.5–1 session

### Step 5.1: Final Formatting
**Lead:** Technical Editor
- Apply target venue formatting requirements
- Verify reference format matches style guide (APA, Vancouver, etc.)
- Check figure/table numbering and cross-references
- Ensure supplementary materials are complete
- Verify author information and affiliations
- Check acknowledgments and funding statements

### Step 5.2: Sign-Off
**Lead:** Research Director
- Final read-through of complete manuscript
- Verify all Phase 4 issues resolved
- Confirm manuscript meets original research objectives
- Approve for submission

### Phase 5 Output
- Submission-ready manuscript
- Cover letter (if required)
- Supplementary materials
- Submission checklist completed

---

## Iteration Protocol
If any check in Phase 3 or Phase 4 fails:

1. **Single track failure:** Return to Phase 2 with specific fix list.
   Re-run only the failed check after fixes.
2. **Multiple track failures:** Return to Phase 2 with consolidated feedback.
   Re-run all Phase 3 passes after fixes.
3. **Structural failure (Phase 3, Pass 1):** May require return to Phase 1
   for outline revision. Research Director decides.
4. **Maximum iterations:** 3 full cycles. If still failing, escalate to
   Research Director for scope/approach reconsideration.
