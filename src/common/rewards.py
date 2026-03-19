"""Shared reward utilities: think-block stripping, evaluator instance, and reward constants."""

import os
import re

from common.vendor.evaluator import CodeEvaluator

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*$", re.DOTALL)

# Harmony format: extract content from the "final" channel (full special tokens)
_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>|$)",
    re.DOTALL,
)
_HARMONY_ANALYSIS_RE = re.compile(
    r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>|$)",
    re.DOTALL,
)

# Harmony format with stripped special tokens (TRL skip_special_tokens=True):
# Output looks like "analysis[reasoning]assistantfinal[code]"
_HARMONY_STRIPPED_SPLIT_RE = re.compile(r"assistantfinal", re.IGNORECASE)


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


def split_completion(text: str) -> tuple[str, str]:
    """Split a completion into (reasoning, answer) for any supported format.

    Detection logic (in priority order):
    1. **Harmony with full markers** (``<|channel|>`` present): analysis channel
       is reasoning, final channel is answer.
    2. **Harmony with stripped tokens** (contains ``assistantfinal``): split on
       first ``assistantfinal`` boundary.
    3. **Qwen3 with ``<think>`` tag**: content inside ``<think>...</think>`` is
       reasoning, everything after is answer.
    4. **Qwen3 prefill** (no ``<think>`` but has ``</think>``): everything before
       ``</think>`` is reasoning, after is answer.
    5. **Fallback**: reasoning = "", answer = full text.

    Returns:
        (reasoning, answer) tuple with both parts stripped.
    """
    # 1. Harmony with full channel markers
    if "<|channel|>" in text:
        analysis = ""
        am = _HARMONY_ANALYSIS_RE.search(text)
        if am:
            analysis = am.group(1).strip()
        answer = text.strip()
        fm = _HARMONY_FINAL_RE.search(text)
        if fm:
            answer = fm.group(1).strip()
        return (analysis, answer)

    # 2. Harmony with stripped special tokens — only match if text also
    # starts with "analysis" (role token residue) to avoid false positives
    # on normal code containing "assistantfinal" as a variable/string.
    m = _HARMONY_STRIPPED_SPLIT_RE.search(text)
    if m and text.lstrip().lower().startswith("analysis"):
        reasoning = text[:m.start()].strip()
        answer = text[m.end():].strip()
        # Strip leading "analysis" prefix from reasoning if present
        if reasoning.lower().startswith("analysis"):
            reasoning = reasoning[len("analysis"):].strip()
        return (reasoning, answer)

    # 3. Qwen3 with opening <think> tag
    if "<think>" in text:
        # Find the closing tag
        close_idx = text.find("</think>")
        if close_idx != -1:
            reasoning = text[text.index("<think>") + len("<think>"):close_idx].strip()
            answer = text[close_idx + len("</think>"):].strip()
        else:
            # Unclosed think block — everything after <think> is reasoning, no answer
            reasoning = text[text.index("<think>") + len("<think>"):].strip()
            answer = ""
        return (reasoning, answer)

    # 4. Qwen3 prefill (no opening <think>, but has </think>)
    if "</think>" in text:
        close_idx = text.index("</think>")
        reasoning = text[:close_idx].strip()
        answer = text[close_idx + len("</think>"):].strip()
        return (reasoning, answer)

    # 5. Fallback
    return ("", text.strip())


def strip_reasoning(text: str) -> str:
    """Strip reasoning content, handling both Qwen3 think blocks and Harmony channels.

    Uses :func:`split_completion` internally and returns the answer part.

    Args:
        text: Raw model completion that may contain think blocks or Harmony channels.

    Returns:
        Cleaned text with reasoning content removed.
    """
    _, answer = split_completion(text)
    return answer

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
