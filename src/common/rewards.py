import os
import re

from common.vendor.evaluator import CodeEvaluator

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*$", re.DOTALL)


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks (and unclosed trailing ones) from text."""
    text = _THINK_RE.sub("", text)
    text = _THINK_UNCLOSED_RE.sub("", text)
    return text.strip()

# Shared evaluator instance (thread-safe via subprocess isolation)
try:
    _max_jobs = int(os.environ.get("MAX_JOBS", 4))
except (ValueError, TypeError):
    _max_jobs = 4

_evaluator = CodeEvaluator(
    num_workers=_max_jobs,
    timeout=3,
    max_failures=1,
    debug=False,
)

COMPILE_REWARD = 0.5
CORRECTNESS_REWARD = 3.0
