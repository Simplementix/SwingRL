# Phase 17: Doc Housekeeping - Context

**Gathered:** 2026-03-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix stale counters, descriptions, and plan counts in REQUIREMENTS.md and ROADMAP.md so planning documents accurately reflect actual project state. Documentation-only phase — no code changes.

</domain>

<decisions>
## Implementation Decisions

### REQUIREMENTS.md fixes
- Update coverage counter to match actual state after Phase 15-16 completion (all 74 complete → "Complete: 74, Pending: 0")
- Fix descriptions for FEAT-09 and FEAT-10 to reflect Phase 14 gap closure work (correlation pruning CLI + sentiment integration)
- Fix descriptions for PAPER-02 and PAPER-09 to reflect Phase 13 gap closure work (model path fix + reconciliation scheduling)
- Note: Phase 15-16 will reset some requirements to Pending during execution — this phase runs after 15-16, so final counts reflect completed state

### ROADMAP.md fixes
- Update progress table plan counts for Phase 5 (show 5/5 not 4/5)
- Update progress table plan counts for Phase 6 (show 3/3 not 0/3)
- Update progress table plan counts for Phase 7 (show 3/3 not 2/3)
- All phases should show correct completed plan counts and "Complete" status

### Claude's Discretion
- Exact wording for updated requirement descriptions
- Whether to update REQUIREMENTS.md "Last updated" date footer
- Whether to also fix STATE.md stale fields (progress counts, current position)

</decisions>

<specifics>
## Specific Ideas

No specific requirements — all changes are factual corrections enumerated in the v1.0 re-audit tech debt section.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- None — documentation-only phase

### Established Patterns
- REQUIREMENTS.md coverage counter at bottom of file
- ROADMAP.md progress table with `| Phase | Plans Complete | Status | Completed |` format

### Integration Points
- `.planning/REQUIREMENTS.md` — coverage counter and traceability table
- `.planning/ROADMAP.md` — progress table

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 17-doc-housekeeping*
*Context gathered: 2026-03-10*
