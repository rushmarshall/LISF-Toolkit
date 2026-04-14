---
name: decision-tracking
description: "Structured tracking of review decisions, editorial changes, and deferred items. Provides an audit trail for manuscript revisions and collaborative reviews."
version: 1.0.0
author: Sebastian R.O. Marshall
tags: [Decision Tracking, Review Management, Audit Trail]
---

# Decision Tracking Skill

Track every decision made during manuscript review, revision, and collaboration. Maintain a clear audit trail so that all changes are documented, justified, and recoverable.

---

## 1. Decision Record Format

Every decision must follow this template:

```markdown
## DECISION: [Short descriptive title]
- **ID**: D-[NNN]
- **Date**: YYYY-MM-DD
- **Status**: OPEN | RESOLVED | DEFERRED
- **Tag**: EDIT | COMMENT | QUESTION | PRAISE
- **Context**: [Why this decision came up — what triggered it]
- **Options considered**:
  1. [Option A — description]
  2. [Option B — description]
  3. [Option C — description, if applicable]
- **Chosen**: [Which option was selected]
- **Rationale**: [Why this option was chosen over alternatives]
- **Assigned to**: [Person responsible, if applicable]
- **Resolved date**: [YYYY-MM-DD, when closed]
```

---

## 2. Status Workflow

Decisions follow a simple state machine:

```
OPEN ──→ RESOLVED    (decision made and implemented)
  │
  └──→ DEFERRED     (needs more information or author input)
         │
         └──→ OPEN  (re-opened when information arrives)
               │
               └──→ RESOLVED
```

### Status Definitions
| Status | Meaning |
|--------|---------|
| **OPEN** | Decision is pending — awaiting review, discussion, or input |
| **RESOLVED** | Decision has been made and implemented |
| **DEFERRED** | Cannot decide now — requires additional information, data, or author input |

### Rules
- Every decision starts as **OPEN**.
- A decision can only move to **RESOLVED** when the chosen action has been **implemented** (not just decided).
- **DEFERRED** items must include a note on what is needed to unblock them.
- **DEFERRED** items should be reviewed weekly to avoid stale backlogs.

---

## 3. Tagging System

Every decision receives exactly one primary tag:

| Tag | Use When | Example |
|-----|----------|---------|
| **EDIT** | A change to the manuscript text, figures, or data | "Replace 'crucial' with 'important' in paragraph 3" |
| **COMMENT** | An observation or note that does not require a change | "The introduction effectively sets up the research gap" |
| **QUESTION** | Clarification needed from an author or collaborator | "Which time period does Figure 3 cover?" |
| **PRAISE** | Positive feedback on a specific aspect | "Excellent use of effect sizes throughout Results" |

### Tag Guidelines
- **EDIT** decisions must specify the exact change (old text → new text) and location.
- **QUESTION** decisions must specify who needs to answer and the deadline.
- **PRAISE** items are always immediately **RESOLVED** — they exist for the audit trail only.
- **COMMENT** items that later require action should be re-tagged as **EDIT** or **QUESTION**.

---

## 4. Decision Categories

For review workflows, decisions typically fall into these categories:

| Category | Description | Action |
|----------|-------------|--------|
| **APPROVE EDIT** | Suggested edit accepted | Implement in manuscript |
| **REJECT EDIT** | Suggested change is not appropriate | Document rationale for rejection |
| **DEFER** | Needs author or expert input | Assign to responsible party |
| **ESCALATE** | Major concern requiring group discussion | Flag for meeting/email thread |
| **MERGE** | Multiple related decisions consolidated | Link to parent decision |

---

## 5. Decision Log Template

Maintain a running log in a structured file:

```markdown
# Decision Log — [Manuscript Title or Project Name]

## Summary
- **Total decisions**: [N]
- **Open**: [N]
- **Resolved**: [N]
- **Deferred**: [N]

---

## DECISION: [Title]
- **ID**: D-001
- **Date**: YYYY-MM-DD
- **Status**: OPEN
- **Tag**: EDIT
- **Context**: [...]
- **Options considered**: [...]
- **Chosen**: [...]
- **Rationale**: [...]

---

## DECISION: [Title]
- **ID**: D-002
...
```

---

## 6. Summary Report Template

Generate a summary report at key milestones (e.g., end of review round, before submission):

```markdown
# Review Decision Summary
**Date**: YYYY-MM-DD
**Manuscript**: [Title]
**Review round**: [1 / 2 / Final]

## Statistics
| Metric | Count |
|--------|-------|
| Total decisions | [N] |
| Resolved | [N] |
| Open | [N] |
| Deferred | [N] |

## By Tag
| Tag | Count | Resolved | Open | Deferred |
|-----|-------|----------|------|----------|
| EDIT | [N] | [N] | [N] | [N] |
| COMMENT | [N] | [N] | [N] | [N] |
| QUESTION | [N] | [N] | [N] | [N] |
| PRAISE | [N] | [N] | [N] | [N] |

## Open Items Requiring Action
| ID | Title | Tag | Assigned To | Blocking? |
|----|-------|-----|-------------|-----------|
| D-XXX | [...] | [...] | [...] | Yes/No |

## Deferred Items Awaiting Input
| ID | Title | What Is Needed | From Whom | Deadline |
|----|-------|---------------|-----------|----------|
| D-XXX | [...] | [...] | [...] | YYYY-MM-DD |

## Key Decisions This Round
1. **D-XXX**: [Brief description of significant decision and rationale]
2. **D-XXX**: [Brief description]
3. ...
```

---

## 7. Best Practices

### Documentation
- Record decisions **as they happen**, not after the fact.
- Include enough context that someone reading the log 6 months later can understand the reasoning.
- Link to specific manuscript locations (section, paragraph, line) when possible.

### Review Workflow
1. Read through the manuscript/review comments.
2. Create an **OPEN** decision for each item requiring action.
3. Tag each decision (EDIT, COMMENT, QUESTION, PRAISE).
4. Work through OPEN items, resolving or deferring each.
5. Generate a summary report before the next review round.

### Avoiding Decision Debt
- Set a weekly review cadence for DEFERRED items.
- If a deferred item has been waiting >2 weeks, escalate it.
- Close the loop: every QUESTION must eventually become RESOLVED (even if the answer is "not applicable").

---

## 8. Integration with Other Skills

- **manuscript-writing**: EDIT decisions feed directly into manuscript revisions.
- **pdf-verification**: QUESTION decisions may require source verification.
- **academic-phrasebank**: EDIT decisions on wording can reference the phrasebank.

---

*Every decision deserves a record. Undocumented changes lead to confusion, duplicated work, and lost rationale. Track decisions consistently from the first draft to final submission.*
