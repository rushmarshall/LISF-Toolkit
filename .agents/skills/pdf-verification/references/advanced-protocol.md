# PDF-Based Citation Verification Skill

## Purpose
Verify citation claims by reading the actual full-text PDF, not just metadata or abstracts.

## Verification Tiers
| Tier | Method | Confidence | When to use |
|------|--------|------------|-------------|
| 1 | Full-text PDF search | HIGH | Paper downloaded, text extractable |
| 2 | Abstract + Crossref metadata | MEDIUM | Paper paywalled, can't download |
| 3 | Training knowledge | LOW | No access at all; flag as UNVERIFIED |

## Verification Protocol

### Pre-check: DOI Resolution
Use `batch_crossref_verify` to verify all DOIs at once:
```
batch_crossref_verify(dois="doi1, doi2, doi3, ...")
```
If a DOI is fabricated (wrong title), STOP and flag immediately.

### Download
Use `batch_download` to fetch all papers concurrently:
```
batch_download(dois="doi1, doi2, doi3, ...")
```
Processes 8 DOIs concurrently. Sources: Semantic Scholar, OpenAlex, Unpaywall, PMC, UVA proxy.

### Quantitative Claim Verification
For each claim like "54 km³/yr TWS loss (Tiwari et al., 2009)":

1. **Search for the number**: `search_paper(doi, "54")` and `search_paper(doi, "re:5[34]\s*±?\s*\d*\s*km")`
2. **Search for related terms**: `search_paper(doi, "TWS")`, `search_paper(doi, "loss")`
3. **Read the Results section**: `read_paper(doi, start_page=X, end_page=Y)` where X-Y is the Results section
4. **Use verify_claim** for a quick automated check
5. **Manual review**: Read the evidence passages carefully

### What to Check
- **Exact value**: Does the paper say 54 or 54±9 or ~54?
- **Units**: km³/yr vs km³ total vs mm/yr vs Gt/yr
- **Time period**: 2002-2008 vs 2002-2006 vs 2003-2008
- **Geographic scope**: "northern India" vs "3 NW Indian states" vs "Indo-Gangetic Plain"
- **Quantity type**: TWS vs groundwater vs total storage
- **Statistical context**: Mean value vs peak vs minimum
- **Where in paper**: Abstract, Results, Discussion, Supplementary?

### Output Format
```
CITATION: Author et al. (Year)
DOI: 10.xxxx/xxxxx
CLAIM: "[what our chapter says]"
STATUS: CONFIRMED / MODIFIED / NOT_FOUND / WRONG
EVIDENCE: "[exact text from paper, page X]"
CORRECTION: "[if MODIFIED/WRONG, what the paper actually says]"
CONFIDENCE: HIGH (full-text) / MEDIUM (abstract only) / LOW (metadata only)
```

## Batch Verification Workflow
When verifying all citations in a section:
1. Extract all (Author, Year, claim) triples from the text
2. Crossref-verify all DOIs in parallel
3. Download all available PDFs (Unpaywall for open access)
4. For each paper: search → verify → report
5. Compile results into a verification table

## Known Publisher Access Patterns
| Publisher | Open Access | API Download | Notes |
|-----------|-------------|-------------|-------|
| Springer/Nature | Some OA | YES (Semantic Scholar/OpenAlex) | OA papers downloadable via httpx |
| Wiley (AGU) | Some OA | YES (Unpaywall) | WRR, GRL, JGR papers often paywalled |
| AAAS (Science) | Rarely | Blocked (403) | Even OA papers block automated access |
| Elsevier | Rarely | Sometimes | Strong bot detection |
| MDPI | All OA | YES | Easiest to download |
| Copernicus | All OA | YES | HESS, TC, etc. |
| AMS | Some OA | YES | BAMS, J. Hydrometeorol |
| PMC | All OA | YES | Via PMC ID from Semantic Scholar |

## Handling Paywalled Papers
1. The v2 server automatically tries UVA proxy (JOURNAL_PROXY_PREFIX env var)
2. If proxy fails, download PDFs manually and place in `papers/` folder
3. Filename format: DOI with slashes→underscores, e.g., `10.1029_2021WR031614.pdf`
4. Use `search_all_papers` to search a term across all downloaded PDFs at once

## Batch Verification Workflow
When verifying all citations in a section:
1. Extract all (Author, Year, claim) triples from the text
2. `batch_crossref_verify` all DOIs in one call (10 concurrent)
3. `batch_download` all DOIs in one call (8 concurrent)
4. For each paper: `verify_claim` or `search_paper` for specific numbers
5. Use `search_all_papers` for cross-paper term searches
6. Compile results into a verification table
