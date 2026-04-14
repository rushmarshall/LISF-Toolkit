# Post-Mortem Protocol
> After-action review conducted at the end of every project or significant task.
> Owner: Research Director | Version: 1.0

## Overview
Post-mortems capture what happened, why it happened, and what to change going
forward. They are the primary input to the institute's learning protocol
(`protocols/learning.md`). Every completed project gets a post-mortem. Archive
completed reviews in `memory/postmortems/`.

**Estimated Duration:** 15–30 minutes
**Required Participants:** All agents involved in the project
**File Naming:** `memory/postmortems/YYYY-MM-DD-project-name.md`

---

## Post-Mortem Template

```markdown
## Post-Mortem: [Project/Task Name]
Date: [DATE]
Duration: [Total sessions / time spent]
Participants: [List of agents involved]
Status: [Completed / Partially Completed / Abandoned]

### Summary
[1–2 sentence description of the project goal and outcome]

### What Went Well
- [Successes and effective processes]
- [Tools or techniques that worked particularly well]
- [Efficient collaborations between agents]
- [Quality gates that caught real issues]

### What Went Wrong
- [Issues, delays, errors encountered]
- [Root cause for each issue]
- [Impact of each issue on the project]

### Lessons Learned
- [Actionable insights from this project]
- [Which of these should be promoted to knowledge-base.md?]
- [Tag each lesson: PROMOTE / LOCAL / INVESTIGATE]

### Process Improvements
- [Recommended changes to agent definitions]
- [Recommended changes to workflows]
- [Recommended changes to protocols]
- [New skills or capabilities needed]
- [Tools that should be added or modified]

### Metrics
- Papers reviewed: [X]
- DOIs verified: [Y / Z] ([percentage]% success rate)
- Issues found in verification: [N] (Critical: [_], Major: [_], Minor: [_])
- Revision cycles: [N]
- Triple-check pass rate: [first-pass / total attempts]
- Final quality score: [_/10]
- Estimated vs. actual duration: [estimated] → [actual]

### Follow-Up Actions
| Action | Owner | Deadline | Status |
|--------|-------|----------|--------|
| [Action item] | [Agent] | [Date] | PENDING |
```

---

## Post-Mortem Process

### Step 1: Data Collection
- Gather all artifacts from the project (drafts, reviews, verification reports)
- Note any in-session observations or surprises
- Collect metrics from each pipeline stage

### Step 2: Facilitated Review
- Research Director facilitates the review
- Each participating agent shares their perspective
- Focus on systemic issues, not individual blame

### Step 3: Action Items
- Convert lessons into specific, actionable changes
- Assign owners and deadlines for each action
- Determine which lessons to promote to knowledge base

### Step 4: Archive
- Save completed post-mortem to `memory/postmortems/`
- Update `memory/decision-log.md` with any decisions made
- Submit promotion candidates to Session Manager (Curator)
