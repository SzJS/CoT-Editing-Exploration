import re


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_code_blocks(text: str) -> str | None:
    """Extract fenced python code blocks from text, joining multiple blocks.

    This is a safety net -- CodeEvaluator.parse_response() already handles this,
    and <think> blocks are plain text (not in code fences) so they're naturally excluded.
    """
    blocks = re.findall(r"```(?:python)?\n(.*?)(?:```|$)", text, re.DOTALL | re.IGNORECASE)
    if not blocks:
        return None
    cleaned = [b.strip() for b in blocks if b.strip()]
    if not cleaned:
        return None
    return "\n\n".join(cleaned)


def extract_code_from_completion(text: str) -> str | None:
    """Strip thinking tokens then extract code blocks."""
    stripped = strip_thinking(text)
    return extract_code_blocks(stripped)
