# Learning Protocol
> How the Marshall Research Institute learns, improves, and evolves over time.
> Owner: Administration (Session Manager / Curator) | Version: 1.0

## Overview
The institute improves through a structured learning loop: capture observations
during sessions, review them in post-mortems, promote durable lessons to the
knowledge base, and evolve agent definitions and workflows accordingly. This
protocol ensures that mistakes are learned from and successes are replicated.

---

## During Sessions

### Real-Time Capture
As agents work, they record observations in real time:

1. **Decision logging** → `memory/decision-log.md`
   - Log every non-trivial decision (approach chosen, tool selected, scope adjusted)
   - Include rationale — why this choice over alternatives
   - Tag with project name and date

2. **Surprise logging**
   - Note unexpected findings (results that contradict expectations)
   - Note unexpected difficulties (tools failing, data unavailable)
   - Note unexpected successes (approaches that worked better than expected)
   - Capture these in session notes for the post-mortem

3. **Error pattern logging**
   - When an error occurs, note:
     - What happened
     - What caused it
     - How it was resolved
     - Whether it's a new pattern or a recurrence

### Tagging Convention
All observations should be tagged for later retrieval:
- `[LESSON]` — actionable insight
- `[ERROR]` — mistake or failure
- `[SURPRISE]` — unexpected finding
- `[PROCESS]` — observation about workflow effectiveness
- `[DOMAIN]` — domain-specific knowledge gained

---

## After Sessions (Post-Mortem)

### Step 1: Run Post-Mortem
- Follow `workflows/post-mortem.md` template
- Archive in `memory/postmortems/YYYY-MM-DD-project-name.md`

### Step 2: Extract Lessons
- Review all tagged observations from the session
- Identify patterns across multiple observations
- Draft candidate lessons in clear, actionable language

### Step 3: Triage Lessons
The Session Manager (Curator) reviews each candidate lesson and assigns a disposition:

| Disposition | Meaning | Action |
|-------------|---------|--------|
| **PROMOTE** | Durable, broadly applicable | Add to `memory/knowledge-base.md` |
| **LOCAL** | Project-specific, not generalizable | Keep in post-mortem only |
| **INVESTIGATE** | Needs more data before deciding | Track; revisit after next project |

---

## Promotion Criteria

A lesson is promoted to `memory/knowledge-base.md` if it meets **any** of these criteria:

1. **Cross-project applicability** — it applies to 2+ projects or research domains
2. **Error prevention** — it prevents a class of errors, not just a single instance
3. **Protocol change** — it leads to a change in a protocol or workflow
4. **Domain knowledge** — it captures verified domain knowledge not documented elsewhere
5. **Tool knowledge** — it documents tool behavior, API quirks, or workarounds

### Promotion Format
When adding to the knowledge base, include:
- The lesson in clear, concise language
- Source: which post-mortem or session it came from
- Date added
- Applicable domain or category

---

## Knowledge Base Management

### Size Cap
- Maximum **500 lines** in `memory/knowledge-base.md`
- When approaching the cap, prune stale or less-relevant entries

### Tagging
Every knowledge base entry is tagged with:
- **Domain:** (e.g., statistics, ecology, writing, tools)
- **Date added:** YYYY-MM-DD
- **Source:** Post-mortem reference

### Quarterly Review
Every ~3 months (or every 10 projects, whichever comes first):
1. Review all entries older than 6 months
2. For each entry, assess:
   - Is this still accurate?
   - Is this still relevant?
   - Has it been superseded by a newer lesson?
3. Dispositions:
   - **Keep** — still valuable
   - **Update** — needs revision
   - **Archive** — move to a separate archive file
   - **Delete** — no longer relevant

---

## Agent Evolution

### Review Trigger
After every 3 completed projects, review each agent definition:

### Review Process
1. **Capability audit**
   - Remove capabilities that were never used across 3+ projects
   - Add capabilities that were repeatedly needed but missing
   - Update skill descriptions to match actual usage patterns

2. **Protocol audit**
   - Update protocols based on post-mortem findings
   - Remove protocol steps that consistently add no value
   - Add protocol steps that would have prevented recurring issues

3. **Workflow audit**
   - Adjust workflow stages based on actual performance
   - Optimize stages that consistently take longer than expected
   - Remove bottleneck stages or add parallel tracks

4. **Version increment**
   - Update the version number in the agent definition header
   - Document what changed and why in a changelog comment

### Evolution Principles
- **Evidence-based:** Changes must be supported by post-mortem data
- **Conservative:** Don't change what's working; focus on what's failing
- **Reversible:** Keep old definitions available in case changes need rollback
- **Documented:** Every change has a rationale traced to a specific lesson
