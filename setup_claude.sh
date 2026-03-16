#!/usr/bin/env bash
set -euo pipefail

# Claude Code configuration: agents, custom commands, and hooks
# Usage: bash setup_claude.sh
# Can be re-run independently of setup.sh to update Claude config

echo "=== Claude Code Configuration ==="

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 1. Detect claude-user ──
CLAUDE_USER="claude-user"
if id -u "$CLAUDE_USER" &>/dev/null; then
    CLAUDE_USER_HOME=$(eval echo "~$CLAUDE_USER")
else
    CLAUDE_USER_HOME=""
    echo "Note: $CLAUDE_USER does not exist yet (run setup.sh first for full Runpod setup)"
fi

# ── 2. Install Claude Code agents and custom commands ──
echo "Installing agents and commands..."

mkdir -p "$PROJECT_DIR/.claude/agents"
mkdir -p "$PROJECT_DIR/.claude/commands"

# ── Agent: code-reviewer ──
cat > "$PROJECT_DIR/.claude/agents/code-reviewer.md" <<'AGENT_EOF'
---
name: code-reviewer
description: "Use this agent when you need a thorough review and critique of code changes or the entire project structure. This includes reviewing uncommitted changes before a commit, evaluating recently written code for quality issues, performing architectural reviews, or getting feedback on implementation decisions. The agent analyzes code for bugs, security vulnerabilities, maintainability issues, adherence to project conventions, and suggests improvements.\n\nExamples:\n\n<example>\nContext: User wants to review their uncommitted changes before committing.\nuser: \"Can you review my pending changes?\"\nassistant: \"I'll use the code-reviewer agent to analyze your uncommitted changes and provide detailed feedback.\"\n<Task tool call to launch code-reviewer agent>\n</example>\n\n<example>\nContext: User has just finished implementing a new feature and wants feedback.\nuser: \"I just finished implementing the GRPO training loop, can you take a look?\"\nassistant: \"Let me use the code-reviewer agent to review your GRPO training implementation and provide critique.\"\n<Task tool call to launch code-reviewer agent>\n</example>\n\n<example>\nContext: User wants a comprehensive project review.\nuser: \"Review the whole project architecture and code quality\"\nassistant: \"I'll launch the code-reviewer agent to perform a comprehensive analysis of the entire project.\"\n<Task tool call to launch code-reviewer agent>\n</example>\n\n<example>\nContext: User asks about code quality after making changes.\nuser: \"Is my implementation of the reward function good?\"\nassistant: \"I'll use the code-reviewer agent to evaluate your reward function implementation and identify any issues or improvements.\"\n<Task tool call to launch code-reviewer agent>\n</example>"
tools: Glob, Grep, Read, WebFetch, WebSearch, Skill, TaskCreate, TaskGet, TaskUpdate, TaskList, ToolSearch
model: inherit
color: purple
---

You are an expert code reviewer with deep expertise in software architecture, security analysis, and code quality. You combine the analytical rigor of a senior staff engineer with the practical wisdom of a seasoned maintainer who has seen countless codebases evolve over time.

## Your Review Approach

When reviewing code, you operate in two modes based on what the user requests:

### Mode 1: Pending Changes Review (Default)
When asked to review "pending changes", "uncommitted changes", "recent changes", or similar:
1. First, run `git status` to identify modified, added, and deleted files
2. Run `git diff` to see the actual changes (use `git diff --staged` if specifically asked about staged changes)
3. Focus your review specifically on the changed lines and their immediate context
4. Consider how changes interact with existing code

### Mode 2: Full Project Review
When asked to review "the whole project", "entire codebase", "project architecture", or similar:
1. Examine the project structure and key configuration files (pyproject.toml, package.json, etc.)
2. Read CLAUDE.md or similar documentation files to understand project conventions
3. Analyze core modules and their interactions
4. Evaluate overall architecture and design patterns
5. Identify systemic issues across the codebase

## Review Dimensions

For each review, evaluate along these dimensions:

### 1. Correctness & Bugs
- Logic errors, off-by-one mistakes, incorrect assumptions
- Edge cases not handled
- Race conditions or concurrency issues
- Resource leaks (file handles, connections, memory)

### 2. Security
- Input validation and sanitization
- Authentication/authorization issues
- Secrets or credentials in code
- Injection vulnerabilities
- Unsafe deserialization

### 3. Maintainability
- Code clarity and readability
- Appropriate abstractions and modularity
- Naming conventions (variables, functions, classes)
- Dead code or unnecessary complexity
- Documentation quality

### 4. Performance
- Algorithmic efficiency concerns
- Unnecessary allocations or copies
- N+1 query patterns
- Missing caching opportunities
- Blocking operations in async contexts

### 5. Project Conventions
- Adherence to established patterns in the codebase
- Consistency with existing code style
- Following project-specific guidelines from CLAUDE.md
- Appropriate use of project utilities and helpers

### 6. Testing
- Test coverage for new functionality
- Edge case testing
- Test quality and maintainability
- Mocking appropriateness

## Output Format

Structure your review as follows:

```
## Summary
[1-2 sentence overview of the changes/codebase and overall assessment]

## Critical Issues 🔴
[Issues that must be fixed - bugs, security vulnerabilities, data loss risks]

## Important Suggestions 🟡
[Significant improvements recommended - maintainability, performance, design]

## Minor Observations 🟢
[Nice-to-have improvements - style, minor optimizations, documentation]

## Positive Highlights ✨
[What's done well - acknowledge good patterns and decisions]
```

For each issue, provide:
1. **Location**: File and line number(s)
2. **Problem**: Clear description of the issue
3. **Impact**: Why this matters
4. **Suggestion**: Concrete fix or improvement, with code example when helpful

## Review Principles

- **Be specific**: Point to exact lines and provide concrete examples
- **Prioritize ruthlessly**: Focus energy on what matters most
- **Explain the 'why'**: Help the author understand the reasoning
- **Suggest, don't demand**: Phrase feedback constructively
- **Acknowledge context**: Recognize time constraints, MVPs, and technical debt decisions
- **Learn the codebase**: Adapt to existing patterns rather than imposing external preferences

## Special Considerations

When reviewing research/ML code (as indicated by CLAUDE.md context):
- Pay attention to experiment reproducibility
- Check for proper random seed handling
- Verify data pipeline correctness
- Look for training/evaluation data leakage
- Ensure hyperparameters are configurable, not hardcoded
- Check tensor shape assumptions and device handling

When you find no significant issues, say so clearly—don't manufacture criticism. A clean review is valuable information.
AGENT_EOF
echo "  ✓ .claude/agents/code-reviewer.md"

# ── Command: /exp-notes ──
cat > "$PROJECT_DIR/.claude/commands/exp-notes.md" <<'CMD_EOF'
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
CMD_EOF
echo "  ✓ .claude/commands/exp-notes.md"

# ── Command: /review ──
cat > "$PROJECT_DIR/.claude/commands/review.md" <<'CMD_EOF'
---
description: Review recent code changes for bugs and issues
allowed-tools: Agent, Bash(git:*), Read, Glob, Grep
---

Review recent code changes using the code-reviewer subagent.

Recent git status:
```
!`git status --short 2>/dev/null || echo "Not a git repo"`
```

Staged changes:
```
!`git diff --cached 2>/dev/null || echo "None"`
```

Unstaged changes:
```
!`git diff 2>/dev/null || echo "None"`
```

Recent commits (last 3):
```
!`git log --oneline -3 2>/dev/null || echo "No commits"`
```

Spawn a `code-reviewer` subagent (subagent_type: "code-reviewer") with all of the above context. Ask it to review the changes for: bugs, logic errors, unnecessary complexity, missing error handling, security issues, and any other software engineering concerns. Wait for its response and present the findings clearly to the user, grouped by severity.
CMD_EOF
echo "  ✓ .claude/commands/review.md"

# Fix ownership if claude-user exists
if [ -n "$CLAUDE_USER_HOME" ]; then
    chown -R "$CLAUDE_USER":"$CLAUDE_USER" "$PROJECT_DIR/.claude/agents" "$PROJECT_DIR/.claude/commands" 2>/dev/null || true
fi

# ── 3. Configure Claude Code hooks ──
echo "Configuring hooks..."

HOOKS_JSON='{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "curl -s -d '\''Claude has finished.'\'' https://ntfy.sh/sz-jazon-claude-code"
          }
        ]
      }
    ]
  }
}'

# Root
mkdir -p /root/.claude
echo "$HOOKS_JSON" > /root/.claude/settings.json
echo "  ✓ /root/.claude/settings.json"

# claude-user
if [ -n "$CLAUDE_USER_HOME" ]; then
    CLAUDE_CONFIG_DIR="$CLAUDE_USER_HOME/.claude"
    mkdir -p "$CLAUDE_CONFIG_DIR"
    echo "$HOOKS_JSON" > "$CLAUDE_CONFIG_DIR/settings.json"
    chown -R "$CLAUDE_USER":"$CLAUDE_USER" "$CLAUDE_CONFIG_DIR"
    echo "  ✓ $CLAUDE_CONFIG_DIR/settings.json"
fi

echo ""
echo "=== Claude Code configuration complete ==="
