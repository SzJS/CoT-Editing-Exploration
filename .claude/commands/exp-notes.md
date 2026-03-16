---
description: Start or continue an ML experiment discussion
allowed-tools: Read, Write, Edit, Glob, Bash(date:*), AskUserQuestion
argument-hint: [exp-X.Y.md]
---

Facilitate an ML experiment discussion while maintaining a structured document.

## Setup

1. **Get filename:** Use `$ARGUMENTS` or ask user for experiment number → create filename `exp-X.Y.md`
2. **Check if file exists:**
   - **New:** Create from TEMPLATE below (auto-fill date). Then run Pre-flight.
   - **Resuming:** Read file, run Discrepancy Check, summarize state (filled/missing sections, recent notes), ask where to continue.

## Template

~~~
# Experiment X.Y – YYYY-MM-DD

## Context & Reproducibility

### Parent Experiment
### Working Commit + Dependencies
### Output Artefacts
### Other Dependencies / Links

---

## Summary

---

## Motivation
**Fill before starting.**

---

## Design
**Fill before starting.**

---

## Results

---

## Notes / Log / Scratchpad

~~~

## Pre-flight

**Must fill before proceeding:** Motivation, Design
**Prompt once (don't block):** Parent Experiment, Working Commit

## During Discussion

- Add timestamped notes to Notes section: `[YYYY-MM-DD HH:MM] content`
- Update formal sections when relevant info emerges
- Remove italicized placeholder text when filling sections
- Ask clarifying questions to strengthen experimental design

## Discrepancy Check (on resume)

Compare Notes against Motivation/Design/Results. Flag inconsistencies, ask user whether to update formal section or clarify notes.

## Wrap-up

When user indicates done: prompt for Summary (max 200 words) and Results, check completeness.
