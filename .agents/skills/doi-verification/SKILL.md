---
name: doi-verification
description: "Automated DOI resolution and metadata validation via CrossRef API. Use for reference list verification in any manuscript."
version: 1.0.0
author: Sebastian R.O. Marshall
tags: [DOI, CrossRef, Citation Verification, Reference Management]
---

# DOI Verification Skill

Resolve and validate Digital Object Identifiers (DOIs) against the CrossRef API. This skill provides commands, workflows, and failure-mode handling for verifying reference lists in academic manuscripts.

---

## 1. Single DOI Verification

### 1.1 Full Metadata Retrieval
```bash
# Verify a single DOI and extract all key metadata fields
DOI="10.1038/s41586-023-06626-3"
curl -s "https://api.crossref.org/works/$DOI" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)['message']
    print('=== DOI Verification Result ===')
    print(f\"DOI:     {d.get('DOI', 'N/A')}\")
    print(f\"Title:   {d.get('title',['N/A'])[0]}\")
    authors = d.get('author', [])
    auth_str = ', '.join(a.get('family','?') for a in authors[:3])
    if len(authors) > 3: auth_str += ' et al.'
    print(f\"Authors: {auth_str}\")
    pub = d.get('published', d.get('issued', {}))
    year = pub.get('date-parts', [['']])[0][0]
    print(f\"Year:    {year}\")
    print(f\"Journal: {d.get('container-title',['N/A'])[0]}\")
    print(f\"Volume:  {d.get('volume','N/A')}\")
    print(f\"Pages:   {d.get('page','N/A')}\")
    print(f\"Type:    {d.get('type','N/A')}\")
    print(f\"URL:     {d.get('URL','N/A')}\")
    # Check for retractions
    updates = d.get('update-to', [])
    if updates:
        for u in updates:
            print(f\"⚠️  UPDATE: {u.get('label','')} — {u.get('DOI','')}\")
    print('✅ DOI resolved successfully')
except (json.JSONDecodeError, KeyError):
    print('❌ DOI not found or invalid response')
"
```

### 1.2 Quick Existence Check
```bash
# Fast check — just verify the DOI exists (HTTP status only)
DOI="10.1038/s41586-023-06626-3"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "https://api.crossref.org/works/$DOI")
if [ "$STATUS" = "200" ]; then echo "✅ $DOI exists"; else echo "❌ $DOI not found (HTTP $STATUS)"; fi
```

---

## 2. Batch Verification Workflow

### 2.1 Step 1: Extract DOIs
```bash
# From a plain text reference list or .bib file
grep -oE '10\.[0-9]{4,}/[^ ,;}\"\n]+' references.txt | \
  sed 's/[.)]*$//' | sort -u > dois_to_verify.txt
echo "Extracted $(wc -l < dois_to_verify.txt) unique DOIs"
```

### 2.2 Step 2: Batch Verify
```bash
# Verify all DOIs with rate limiting and detailed output
VALID=0; INVALID=0; TOTAL=0
while IFS= read -r doi; do
  TOTAL=$((TOTAL + 1))
  response=$(curl -s -o /dev/null -w "%{http_code}" \
    "https://api.crossref.org/works/$doi?mailto=your@email.com")
  if [ "$response" = "200" ]; then
    echo "✅ [$TOTAL] $doi"
    VALID=$((VALID + 1))
  else
    echo "❌ [$TOTAL] $doi (HTTP $response)"
    INVALID=$((INVALID + 1))
  fi
  sleep 0.5  # Rate limiting — be polite to CrossRef
done < dois_to_verify.txt
echo ""
echo "=== Summary: $VALID valid, $INVALID invalid, $TOTAL total ==="
```

### 2.3 Step 3: Detailed Metadata Dump
```bash
# For each DOI, extract metadata into a TSV for comparison
echo -e "DOI\tTitle\tFirst_Author\tYear\tJournal\tVolume\tPages" > doi_metadata.tsv
while IFS= read -r doi; do
  curl -s "https://api.crossref.org/works/$doi?mailto=your@email.com" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)['message']
    title = d.get('title',[''])[0].replace('\t',' ')[:100]
    auth = d.get('author',[{}])[0].get('family','?')
    year = d.get('published',d.get('issued',{})).get('date-parts',[['']])[0][0]
    journal = d.get('container-title',[''])[0]
    vol = d.get('volume','')
    pages = d.get('page','')
    print(f'$doi\t{title}\t{auth}\t{year}\t{journal}\t{vol}\t{pages}')
except: print(f'$doi\tERROR\t\t\t\t\t')
" >> doi_metadata.tsv
  sleep 0.5
done < dois_to_verify.txt
echo "Metadata saved to doi_metadata.tsv"
```

---

## 3. Verification Checklist

For each reference, compare the manuscript entry against CrossRef metadata:

| Field | Check | Common Issues |
|-------|-------|---------------|
| **Title** | Exact or near-exact match | Subtitles, encoding, capitalisation |
| **First author** | Surname match | Transliteration, hyphenated names |
| **Year** | Exact match | Online-first vs print year |
| **Journal** | Name match (full or abbreviation) | Abbreviation inconsistencies |
| **Volume** | Exact match | Missing for online-first articles |
| **Pages** | Match or article number | Page ranges vs article IDs |
| **DOI format** | Starts with `10.` | Extra characters, URL encoding |

---

## 4. Common Failure Modes

### 4.1 Fabricated DOIs
- **Symptom**: DOI returns HTTP 404 and no Google Scholar results.
- **Red flag**: DOI format looks plausible but doesn't resolve.
- **Action**: Search for the paper by title + author. If found with a different DOI, correct it. If the paper doesn't exist, flag as potentially fabricated.

### 4.2 Wrong Metadata
- **Symptom**: DOI resolves but title/author/year don't match the manuscript.
- **Causes**: Wrong DOI copied; two papers confused; DOI from a different version.
- **Action**: Verify the correct DOI by searching CrossRef by title:
  ```bash
  curl -s "https://api.crossref.org/works?query.title=TITLE_HERE&rows=3" | python3 -c "
  import sys, json
  items = json.load(sys.stdin)['message']['items']
  for i, item in enumerate(items):
      print(f\"{i+1}. {item.get('DOI','')} — {item.get('title',[''])[0][:80]}\")
  "
  ```

### 4.3 Retracted Papers
- **Symptom**: CrossRef metadata includes an `update-to` field with type "retraction".
- **Action**: Check [Retraction Watch](https://retractionwatch.com/). Citing a retracted paper is acceptable if the retraction is acknowledged, but citing it as valid evidence is not.

### 4.4 Preprints and Non-Indexed Sources
- **Symptom**: DOI returns 404 but paper exists on arXiv, bioRxiv, SSRN, etc.
- **Cause**: Preprint DOIs use different registries (DataCite, not CrossRef).
- **Action**: For arXiv, use `https://api.semanticscholar.org/v1/paper/arXiv:ARXIV_ID`.

### 4.5 Very Recent Papers (Last 6 Months)
- **Symptom**: DOI not yet indexed in CrossRef.
- **Cause**: Publisher delay in registering metadata.
- **Action**: Verify via publisher website directly. Re-check in 2–4 weeks.

---

## 5. DOI Format Reference

### Valid DOI Patterns
```
10.1038/s41586-023-06626-3          (Nature)
10.1126/science.abm8668             (Science)
10.1029/2022GL098432                (AGU journals)
10.1016/j.scitotenv.2023.161234    (Elsevier)
10.5194/hess-27-1243-2023           (Copernicus)
10.3390/rs15040932                  (MDPI)
```

### DOI Regex for Extraction
```regex
10\.[0-9]{4,}/[^\s,;}\"\]]+
```
Note: Strip trailing punctuation (periods, parentheses) after extraction.

---

## 6. Integration with Citation Verification Agent

When used within the ai-research-agents framework:

1. **Input**: Reference list (plain text, BibTeX, or extracted from manuscript).
2. **Process**: Extract DOIs → Batch verify → Flag issues → Cross-check claims.
3. **Output**: Verification report with status for each reference.

### Verification Report Template
```markdown
## DOI Verification Report
**Date**: YYYY-MM-DD
**Manuscript**: [Title]

### Summary
| Metric | Count |
|--------|-------|
| Total references | [N] |
| DOIs extracted | [N] |
| Verified via CrossRef | [N] (✅) |
| Not indexed | [N] (⚠️) |
| Metadata mismatch | [N] (❌) |
| Retracted | [N] (🚫) |

### Issues Requiring Attention
| # | DOI | Issue | Suggested Action |
|---|-----|-------|-----------------|
| 1 | 10.xxxx/... | [Description] | [Action] |
```

---

## 7. Best Practices

- **Always use `mailto` parameter** in CrossRef API calls for higher rate limits.
- **Cache responses** — don't re-verify DOIs you've already checked in the same session.
- **Verify early** — run DOI verification before detailed claim cross-checking.
- **Document unresolvable DOIs** — not every DOI will be in CrossRef; document why and verify manually.
- **Re-run before submission** — metadata can change (corrections, retractions).

---

*A verified reference list is the foundation of academic integrity. Run this workflow on every manuscript before submission.*
