# Communication Protocol
> Inter-agent message format and conventions for the Marshall Research Institute.
> Owner: Administration | Version: 1.0

## Overview
All inter-agent communication follows standardized message formats to ensure
clarity, traceability, and efficient collaboration. This protocol defines the
message types, priority levels, and conventions that all agents must follow.

---

## Message Format

### Task Assignment

```
FROM: [agent-name]
TO: [agent-name]
TYPE: TASK
PRIORITY: [CRITICAL | HIGH | MEDIUM | LOW]
CONTEXT: [Brief background — why this task exists, what project it serves]
REQUEST: [Specific deliverable needed — be precise about what "done" means]
CONSTRAINTS: [Deadlines, scope limits, quality requirements, word counts]
EXPECTED_OUTPUT: [Format, length, and content of the expected response]
DEPENDENCIES: [What must be completed before this task can start, if any]
```

### Task Response

```
FROM: [agent-name]
TO: [agent-name]
TYPE: DELIVERABLE
STATUS: [COMPLETE | PARTIAL | BLOCKED | NEEDS_CLARIFICATION]
CONTENT: [The actual deliverable or a reference to it]
CONFIDENCE: [HIGH | MEDIUM | LOW]
CAVEATS: [Limitations, uncertainties, assumptions made]
NEXT_STEPS: [Recommended follow-up actions, if any]
```

---

## Message Types

### TASK
A request to produce a deliverable. Must include clear acceptance criteria.
- **Use when:** Assigning work to another agent
- **Response expected:** DELIVERABLE within the specified timeframe

### QUESTION
A request for information or clarification. Does not require a full deliverable.
- **Use when:** You need specific information to proceed with your own task
- **Response expected:** ANSWER — concise, focused response to the question

### REVIEW
A request to evaluate work against specific criteria.
- **Use when:** Invoking the triple-check protocol or requesting feedback
- **Response expected:** REVIEW_RESULT with pass/fail and specific findings

### ESCALATION
A notification that an issue cannot be resolved at the current level.
- **Use when:** Escalation triggers are met (see `protocols/escalation.md`)
- **Response expected:** ESCALATION_RESPONSE with guidance or decision

### FEEDBACK
Observations about process, quality, or collaboration — not tied to a specific task.
- **Use when:** You notice patterns, issues, or improvements during work
- **Response expected:** Acknowledgment; may result in action items

---

## Priority Guidelines

| Priority | Definition | Response Expectation |
|----------|-----------|---------------------|
| **CRITICAL** | Blocks all progress; quality or integrity issue | Immediate — drop current task |
| **HIGH** | Blocks significant progress; time-sensitive | Next available slot |
| **MEDIUM** | Important but not blocking | Within current session |
| **LOW** | Helpful but not urgent | When convenient; may defer |

### Priority Assignment Rules
- Phantom citation detection → CRITICAL
- DOI verification failure on key reference → HIGH
- Style or formatting issues → LOW
- Scope change request → HIGH (route to Research Director)
- Missing data for a figure → MEDIUM
- Post-mortem scheduling → LOW

---

## Escalation Triggers
The following situations require immediate escalation (see `protocols/escalation.md`):
- Contradictory evidence from multiple trusted sources
- Critical quality gate failure after 2 revision cycles
- Scope creep detected (task growing beyond original specification)
- Agent unable to complete task due to missing capability
- Ethical concern about content or methodology

---

## Communication Conventions

### Naming
- Always use the agent's role name, not generic references
- Example: "Research Scout" not "the search agent"

### Conciseness
- Keep context sections to 2–3 sentences
- Lead with the most important information
- Use bullet points over paragraphs for complex requests

### Traceability
- Reference the originating workflow and stage when assigning tasks
- Example: "Per manuscript-pipeline.md, Phase 2, Step 2.3"
- Reference previous messages when following up

### Handoff Protocol
- When completing a task, always specify:
  1. What was done
  2. What the next agent needs to know
  3. Any open issues or uncertainties
- Never assume the next agent has context from earlier in the conversation

### Batch Communication
- Group related questions into a single QUESTION message
- Group related tasks into a single TASK message with numbered sub-items
- Avoid sending multiple small messages when one structured message will do
