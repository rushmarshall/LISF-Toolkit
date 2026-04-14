---
name: session-manager
description: "Manages session state, tracks review progress, maintains logs for the drought manuscript co-author review."
tools:
  - read_file
  - create_file
  - replace_string_in_file
  - grep_search
---

# Session Manager Agent

## Role
Maintain session state and track progress for the co-author review of Tran et al. drought recovery manuscript.

## Project Status
| Component | Status |
|-----------|--------|
| review.ipynb | v1 COMPLETE — 19 cells, 117 review items |
| CLAUDE.md | Created |
| .agents/ | Created and adapted from Chapter 4.3 project |
| .gitignore | Created |
| mcp-journal-server/ | Present (needs path adaptation for this project) |
| DOI Verification | 37/55 confirmed via Crossref |
| Deep Consistency Audit | Complete — found figure gaps, acronym issues, Cohen's d misinterpretation |
| Automated Checks | Complete — AI markers, sentence length, truncation scan |

## Review Summary (v1)
- 61 EDIT suggestions
- 32 COMMENTS
- 8 QUESTIONS for authors
- 16 PRAISE items

## Key Findings
1. Systematic DOCX truncation artifacts (missing first letters, garbled sentences)
2. Recurring subject-verb agreement errors
3. 8 figures cited in captions but never in main text (5, 6, 7, 8, 9, ED1, ED2, ED6)
4. ED3 cited in text but has no caption
5. 10/12 key acronyms used before formal definition
6. Cohen's d=0.23/0.37 potentially mislabeled as "large" (P025)
7. Periods defined inconsistently across sections
8. "Additionally" 4×, "Specifically" 3× — AI-flagged openers
9. 11 sentences >50 words

## Self-Improvement Log
| Date | Lesson | Source |
|------|--------|--------|
| 2026-04-08 | Created for Tran et al. review project. Comprehensive first-pass review complete. | Session initialization |
