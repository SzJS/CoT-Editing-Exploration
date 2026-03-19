"""Harmony format helpers for gpt-oss models.

Shared utilities for building Harmony-format prompts, parsing channel output,
and encoding/decoding token IDs. Used by eval_cot.py and generate_sft_data.py.
"""

import re

# ---------------------------------------------------------------------------
# Channel parsing regexes
# ---------------------------------------------------------------------------

HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>|$)",
    re.DOTALL,
)
HARMONY_ANALYSIS_RE = re.compile(
    r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|channel\|>|<\|end\|>|$)",
    re.DOTALL,
)

# Model name patterns that indicate Harmony format
HARMONY_MODEL_PATTERNS = ("gpt-oss",)


# ---------------------------------------------------------------------------
# Channel extraction
# ---------------------------------------------------------------------------

def strip_harmony_analysis(text: str) -> str:
    """Extract final channel content from Harmony format output.

    If the output has explicit channel markers, extracts the ``final``
    channel content.  Falls back to the full text if no channel markers found
    (e.g. if model didn't use channels).
    """
    m = HARMONY_FINAL_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def parse_harmony_channels(raw: str) -> tuple[str | None, str | None]:
    """Extract (analysis, final) from Harmony completion text."""
    am = HARMONY_ANALYSIS_RE.search(raw)
    fm = HARMONY_FINAL_RE.search(raw)
    analysis = am.group(1).strip() if am else None
    final = fm.group(1).strip() if fm else None
    return analysis, final


# ---------------------------------------------------------------------------
# Token ID helpers
# ---------------------------------------------------------------------------

def build_harmony_prompt_ids(
    system_text: str,
    messages: list[dict],
    exemplar_turns: list[dict] | None = None,
) -> list[int]:
    """Build Harmony-format token IDs for a prompt.

    Uses openai_harmony (tiktoken-based) to correctly encode special tokens
    including the system message as ``<|start|>system<|message|>...<|end|>``.
    The openai_harmony library's render_conversation omits system messages from
    the token stream, but the actual Harmony format includes them.
    """
    from openai_harmony import load_harmony_encoding

    tok = load_harmony_encoding("HarmonyGptOss")
    allowed = {"<|start|>", "<|end|>", "<|message|>", "<|channel|>"}

    parts = []

    # System message
    if system_text:
        parts.append(f"<|start|>system<|message|>{system_text}<|end|>")

    # Exemplar turns (between system and real user message)
    if exemplar_turns:
        for turn in exemplar_turns:
            role = turn["role"]
            content = turn["content"]
            if role == "assistant":
                if "analysis_content" in turn:
                    parts.append(
                        f"<|start|>assistant<|channel|>analysis<|message|>"
                        f"{turn['analysis_content']}<|end|>"
                    )
                    parts.append(
                        f"<|start|>assistant<|channel|>final<|message|>"
                        f"{content}<|end|>"
                    )
                else:
                    parts.append(f"<|start|>assistant<|message|>{content}<|end|>")
            else:
                parts.append(f"<|start|>{role}<|message|>{content}<|end|>")

    # Real messages (skip system, already added above)
    for msg in messages:
        if msg["role"] == "system":
            continue
        parts.append(f"<|start|>{msg['role']}<|message|>{msg['content']}<|end|>")

    # Add assistant generation prompt
    parts.append("<|start|>assistant")

    full_text = "".join(parts)
    return tok.encode(full_text, allowed_special=allowed)


def build_harmony_prefill_ids(prefill_text: str) -> list[int]:
    """Build token IDs for Harmony analysis channel prefill.

    Returns: [<|channel|>, analysis, <|message|>, ...prefill_text...]
    """
    from openai_harmony import load_harmony_encoding

    tok = load_harmony_encoding("HarmonyGptOss")
    allowed = {"<|channel|>", "<|message|>"}
    return tok.encode(
        f"<|channel|>analysis<|message|>{prefill_text}",
        allowed_special=allowed,
    )


def decode_harmony_tokens(token_ids: list[int]) -> str:
    """Decode Harmony token IDs to text."""
    from openai_harmony import load_harmony_encoding

    tok = load_harmony_encoding("HarmonyGptOss")
    return tok.decode(token_ids)
