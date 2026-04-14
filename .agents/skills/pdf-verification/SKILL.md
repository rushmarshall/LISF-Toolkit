---
name: pdf-verification
description: "Source verification, DOI checking, and claim cross-referencing procedures for academic manuscripts. Ensures every cited reference is real, accessible, and accurately represented."
version: 1.0.0
author: Sebastian R.O. Marshall
tags: [Citation Verification, DOI, CrossRef, Academic Integrity]
---

# PDF Verification Skill

Verify that every reference in a manuscript is real, correctly cited, and faithfully represented. This skill covers DOI resolution, metadata validation, PDF retrieval, and claim cross-checking.

---

## 1. Single DOI Verification

### 1.1 CrossRef API Command
```bash
# Verify a single DOI and extract key metadata
curl -s "https://api.crossref.org/works/DOI_HERE" | python3 -c "
import sys, json
d = json.load(sys.stdin)['message']
print(f\"Title: {d.get('title',[''])[0][:80]}\")
print(f\"Author: {d.get('author',[{}])[0].get('family','?')}\")
print(f\"Year: {d.get('published',{}).get('date-parts',[['']])[0][0]}\")
print(f\"Journal: {d.get('container-title',[''])[0]}\")
print(f\"Volume: {d.get('volume','')}\")
print(f\"Pages: {d.get('page','')}\")
print(f\"Type: {d.get('type','')}\")
print(f\"URL: {d.get('URL','')}\")
"
```

### 1.2 What to Check
Compare CrossRef metadata against the manuscript's reference entry:
- [ ] **Title**: Does the cited title match CrossRef exactly (or closely)?
- [ ] **Authors**: First author surname match? Author count plausible?
- [ ] **Year**: Publication year match?
- [ ] **Journal**: Journal name match?
- [ ] **Volume/Pages**: Volume and page numbers correct?
- [ ] **DOI format**: Properly formatted (starts with `10.`)?

---

## 2. Batch Verification Workflow

### 2.1 Extract DOIs from Reference List
```bash
# Extract all DOIs from a text file (references, .bib, or manuscript)
grep -oE '10\.[0-9]{4,}/[^ ,;}\n]+' references.txt | sort -u > dois.txt
echo "Found $(wc -l < dois.txt) unique DOIs"
```

### 2.2 Batch Verify via CrossRef
```bash
# Verify each DOI and log results
while IFS= read -r doi; do
  response=$(curl -s -o /dev/null -w "%{http_code}" "https://api.crossref.org/works/$doi")
  if [ "$response" = "200" ]; then
    echo "✅ $doi"
  else
    echo "❌ $doi (HTTP $response)"
  fi
  sleep 1  # Rate limit: be polite to CrossRef API
done < dois.txt
```

### 2.3 Rate Limiting
- CrossRef allows ~50 requests/second for polite clients.
- Always include a delay (`sleep 1`) for batch operations.
- For large batches (100+ DOIs), consider using the CrossRef `mailto` parameter:
  ```
  curl -s "https://api.crossref.org/works/DOI?mailto=your@email.com"
  ```
  This moves you to the polite pool with higher rate limits.

---

## 3. Common DOI Issues and Solutions

### 3.1 DOI Not Found (HTTP 404)
| Cause | Solution |
|-------|----------|
| Typo in DOI | Check for missing characters, extra spaces, or encoding issues |
| Very recent paper (2024–2025) | May not be indexed yet; verify via publisher website |
| Preprint (arXiv, bioRxiv) | Preprints may lack DOIs; use arXiv ID instead |
| Legacy paper (pre-2000) | May not have been retroactively assigned a DOI; verify manually |
| Book chapter or report | May use a different DOI prefix; check publisher directly |

### 3.2 DOI Resolves but Metadata Mismatch
| Issue | Action |
|-------|--------|
| Title differs slightly | May be a subtitle issue or encoding difference — acceptable if close |
| Author name differs | Check for transliteration variants or name changes |
| Year off by one | Check online-first vs print publication dates |
| Journal name differs | Check for abbreviation vs full name |
| Wrong paper entirely | **Red flag** — possible fabricated or misattributed reference |

### 3.3 Retracted Papers
- Check CrossRef metadata for `update-to` field indicating retraction.
- Cross-reference with [Retraction Watch](https://retractionwatch.com/) database.
- A cited retracted paper is not automatically a problem — but it must be acknowledged.

---

## 4. PDF Retrieval and Access

### 4.1 Accessing Full Text
- Use DOI URL (`https://doi.org/DOI_HERE`) to reach the publisher page.
- Check institutional access, open access, or preprint versions.
- Unpaywall API can locate free legal copies:
  ```bash
  curl -s "https://api.unpaywall.org/v2/DOI_HERE?email=your@email.com"
  ```

### 4.2 When PDF Is Unavailable
- Check the authors' institutional pages or personal websites.
- Search for preprint versions on arXiv, bioRxiv, SSRN, etc.
- Check Google Scholar for cached or alternative versions.
- Document that the paper could not be independently verified.

---

## 5. Claim Cross-Checking

### 5.1 Process
For each key claim in the manuscript that cites a specific reference:
1. Locate the cited paper (via DOI or search).
2. Find the specific result or statement being cited.
3. Verify that the manuscript's characterisation is **accurate and fair**.
4. Flag any of these issues:
   - **Misquote**: The cited paper says something different.
   - **Overclaim**: The manuscript overstates the cited finding.
   - **Context loss**: The cited finding is presented without important caveats.
   - **Wrong reference**: The claim should cite a different paper.

### 5.2 Verification Log Template
```markdown
| # | Manuscript Claim | Cited Reference | Verified? | Notes |
|---|-----------------|-----------------|-----------|-------|
| 1 | "[Claim text]"  | Author et al. Year | ✅/❌/⚠️ | [Details] |
```

---

## 6. Verification Summary Report

After completing verification, produce a summary:

```markdown
## Reference Verification Summary
- **Total references**: [N]
- **DOIs extracted**: [N]
- **Confirmed valid via CrossRef**: [N] ([%])
- **Not indexed / DOI not found**: [N] (list reasons)
- **Metadata mismatches**: [N] (list details)
- **Claims cross-checked**: [N]
- **Issues found**: [N] (list each)
- **Retracted papers cited**: [N]
```

---

## 7. Integration with Other Skills

- **doi-verification**: Detailed DOI resolution and metadata validation.
- **manuscript-writing**: Reference formatting and citation order checks.
- **decision-tracking**: Log verification decisions for co-author review.

---

*Apply this skill to every manuscript before submission. A single fabricated or misattributed reference can undermine an entire paper's credibility.*
