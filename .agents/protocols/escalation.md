# Escalation Protocol
> When and how to escalate issues beyond the current agent's scope.
> Owner: Administration | Version: 1.0

## Overview
Not every problem can or should be solved by the agent that discovers it.
This protocol defines clear escalation levels, triggers, and procedures to
ensure issues reach the right decision-maker efficiently.

---

## Escalation Levels

### Level 1: Self-Resolution
**Who:** The agent that encountered the issue

**Process:**
1. Attempt to resolve independently using available skills
2. Consult relevant reference materials:
   - `memory/knowledge-base.md` for known patterns
   - `memory/decision-log.md` for prior decisions on similar issues
   - Relevant protocol or workflow documentation
3. Apply best judgment within the agent's defined scope

**Escalate to Level 2 if:**
- Issue persists after self-resolution attempt
- Resolution requires information the agent doesn't have
- The agent's confidence in their resolution is LOW

---

### Level 2: Peer Consultation
**Who:** A collaborating agent in the same or adjacent department

**Process:**
1. Send a QUESTION message (per `protocols/communication.md`)
2. Include:
   - The specific issue encountered
   - What was already tried
   - What information or input is needed
3. Peer provides input or co-resolves the issue

**Escalate to Level 3 if:**
- Peer consultation does not resolve the issue
- The issue crosses department boundaries
- Resolution requires authority the peer doesn't have

---

### Level 3: Department Lead
**Who:** The lead agent of the relevant department

**Department Leads:**
- Research Department → Research Director
- Writing Department → Manuscript Writer (with Technical Editor)
- Data Science Department → Data Analyst
- QA Department → Scientific Reviewer
- Domain Expertise → Domain Expert

**Process:**
1. Send an ESCALATION message with:
   - Issue description
   - Impact on current work
   - Attempted solutions and their outcomes
   - Recommended resolution (if any)
2. Department lead reviews and either resolves or escalates further

**Escalate to Level 4 if:**
- Issue affects multiple departments
- Priority conflicts between tasks
- Scope change is needed
- Critical quality failure detected

---

### Level 4: Research Director
**Who:** Research Director (institute-wide authority)

**Handles:**
- Cross-department coordination issues
- Priority conflicts between competing tasks
- Scope changes to ongoing projects
- Critical quality gate failures (after 2 revision cycles)
- Resource allocation decisions
- Final adjudication on verification disagreements

**Process:**
1. Send an ESCALATION message with CRITICAL or HIGH priority
2. Include a complete briefing:
   - Issue timeline and history
   - All attempted resolutions
   - Impact assessment
   - Recommended options (at least 2)
3. Research Director makes a binding decision
4. Decision is logged in `memory/decision-log.md`

---

## Escalation Triggers

### Immediate Escalation (Skip to Level 4)
These situations bypass normal escalation levels:
- **Fabricated content detected** — phantom citations, invented data
- **Ethical concern** — about content, methodology, or claims
- **Critical contradiction** — key finding contradicted by strong evidence
- **Systematic failure** — a pipeline stage failing repeatedly

### Standard Escalation Triggers
- Contradictory evidence from multiple trusted sources → Level 3
- DOI verification failure on a critical reference → Level 2 (then Level 3 if unresolved)
- Statistical result that contradicts domain knowledge → Level 3
- Scope creep detected → Level 4
- Quality gate failure after 2 revision cycles → Level 4
- Unknown domain requiring new expertise → Level 4
- Agent blocked for more than 15 minutes on a single issue → Level 2

---

## Escalation Response Requirements

Every escalation response must include:
1. **Decision:** What to do next
2. **Rationale:** Why this decision was made
3. **Assignment:** Who is responsible for executing
4. **Timeline:** When it should be completed
5. **Follow-up:** How and when to verify resolution

---

## Anti-Patterns
- **Over-escalation:** Sending routine questions to Research Director. Use self-resolution and peer consultation first.
- **Under-escalation:** Sitting on critical issues hoping they resolve. When in doubt, escalate.
- **Escalation ping-pong:** Passing an issue back and forth between levels. The escalation recipient owns the issue until resolved.
- **Vague escalation:** Sending "I'm stuck" without context. Always include what was tried and what's needed.
