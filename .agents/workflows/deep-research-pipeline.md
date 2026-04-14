# Deep Research Pipeline
> Full research discovery pipeline — from question to literature review draft.
> Owner: Research Department | Version: 1.0

## Overview
This pipeline takes a research question from initial decomposition through literature
discovery, retrieval, validation, and synthesis. It produces a verified literature
review section with all citations confirmed.

**Estimated Duration:** 1–3 sessions depending on scope
**Required Agents:** Research Director, Research Scout, PDF Reader, Domain Expert, Citation Verifier

---

## Stage 1: Problem Decomposition
**Lead:** Research Director

### Inputs
- Research question or topic statement
- Scope constraints (time period, domains, geographic focus)
- Target output format (literature review, background section, scoping review)

### Process
1. Break the primary research question into 3–7 sub-questions
2. Identify existing knowledge vs. knowledge gaps for each sub-question
3. Define the search strategy:
   - Primary keywords and Boolean combinations
   - Target databases (CrossRef, Semantic Scholar, Google Scholar)
   - Inclusion/exclusion criteria (date range, study type, language)
   - Expected number of papers per sub-question
4. Prioritize sub-questions by importance to the overall question

### Outputs
- Sub-question list with priority rankings
- Search strategy document
- Knowledge gap map

### Quality Check
- [ ] Sub-questions are collectively exhaustive for the main question
- [ ] Sub-questions are mutually exclusive (minimal overlap)
- [ ] Search terms cover synonyms and related terminology

---

## Stage 2: Literature Discovery
**Lead:** Research Scout

### Inputs
- Sub-question list and search strategy from Stage 1

### Process
1. **Multi-database search**
   - Execute searches across CrossRef, Semantic Scholar, and Google Scholar
   - Apply inclusion/exclusion criteria
   - Deduplicate results across databases
2. **Citation chaining**
   - Forward citation search from key seed papers
   - Backward citation search (reference lists of relevant papers)
   - Identify highly-cited foundational works
3. **Author tracking**
   - Identify prolific contributors in the space
   - Check recent publications from key authors
   - Note research group affiliations
4. **Gap analysis**
   - Compare discovered literature against sub-questions
   - Identify sub-questions with insufficient coverage
   - Flag areas needing expanded or alternative search terms

### Outputs
- Paper list with metadata (title, authors, year, DOI, abstract)
- Citation network summary
- Key author list
- Gap analysis report

### Quality Check
- [ ] At least 3 databases searched
- [ ] Minimum 10 relevant papers identified
- [ ] Citation chaining performed on top 3–5 seed papers
- [ ] Gap analysis completed for all sub-questions

---

## Stage 3: Document Retrieval
**Lead:** PDF Reader

### Inputs
- Paper list with DOIs from Stage 2

### Process
1. **Batch DOI resolution** — resolve all DOIs via CrossRef API
2. **Full-text retrieval** — download available PDFs or extract from open-access sources
3. **Full-text extraction** — parse PDF content into structured text
4. **Claim-evidence mapping** — extract key claims and the evidence supporting each
5. Flag papers that could not be retrieved for manual follow-up

### Outputs
- Full-text extracts organized by sub-question
- Claim-evidence table (claim → source → page/section → evidence type)
- Retrieval failure list

### Quality Check
- [ ] All DOIs resolve successfully
- [ ] Full-text available for ≥80% of target papers
- [ ] Claims mapped to specific evidence passages

---

## Stage 4: Domain Validation
**Lead:** Domain Expert

### Inputs
- Claim-evidence table from Stage 3
- Full-text extracts

### Process
1. Verify extracted claims against domain knowledge
2. Flag inconsistencies between sources
3. Identify outdated information (superseded by newer findings)
4. Assess methodological soundness of key studies
5. Note domain-specific nuances that may affect interpretation

### Outputs
- Validated claim list (confirmed / disputed / uncertain)
- Methodological quality notes per study
- Domain-specific context annotations

### Quality Check
- [ ] All major claims validated or flagged
- [ ] Methodological concerns documented
- [ ] Temporal consistency verified (newer studies not contradicted by relying on outdated ones)

---

## Stage 5: Synthesis
**Lead:** Research Director

### Inputs
- All outputs from Stages 1–4

### Process
1. Compile findings across sources, organized by sub-question
2. Identify areas of consensus across studies
3. Identify areas of disagreement and characterize the debate
4. Note the strength of evidence for each major finding
5. Draft the literature review section following target format
6. Ensure narrative flows logically between sub-questions

### Outputs
- Literature review draft
- Evidence strength summary table
- Outstanding questions / future research directions

---

## Quality Gate: Final Verification
**Lead:** Citation Verifier

Before the literature review is considered complete:
- [ ] Every DOI in the document resolves to the correct paper
- [ ] Every in-text citation matches an entry in the reference list
- [ ] No fabricated or phantom references
- [ ] Metadata (authors, year, title) matches CrossRef records
- [ ] All claims trace back to a verified source

**Pass → Deliver to requesting workflow**
**Fail → Return to relevant stage with specific issues listed**
