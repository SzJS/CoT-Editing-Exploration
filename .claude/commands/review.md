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
