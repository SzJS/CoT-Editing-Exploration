"""Shared reward utilities: think-block stripping, evaluator instance, and reward constants."""

import os
import re

from common.vendor.evaluator import CodeEvaluator

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*$", re.DOTALL)

# Harmony format: extract content from the "final" channel
_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>|$)",
    re.DOTALL,
)


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks (and unclosed trailing ones) from text.

    Args:
        text: Raw model completion that may contain think blocks.

    Returns:
        Cleaned text with all think blocks removed, stripped of leading/trailing whitespace.
    """
    text = _THINK_RE.sub("", text)
    text = _THINK_UNCLOSED_RE.sub("", text)
    return text.strip()


def strip_reasoning(text: str) -> str:
    """Strip reasoning content, handling both Qwen3 think blocks and Harmony channels.

    For Harmony format (``<|channel|>analysis<|message|>...``), extracts the
    ``final`` channel content. For Qwen3 format (``<think>...</think>``), strips
    think blocks. Falls through to returning stripped text if neither format is detected.

    Args:
        text: Raw model completion that may contain think blocks or Harmony channels.

    Returns:
        Cleaned text with reasoning content removed.
    """
    if "<|channel|>" in text:
        m = _HARMONY_FINAL_RE.search(text)
        return m.group(1).strip() if m else text.strip()
    return _strip_think_blocks(text)

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
