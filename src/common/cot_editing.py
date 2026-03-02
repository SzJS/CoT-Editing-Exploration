"""CoT editing strategies for GRPO training.

Three strategies intervene during GRPO generation to modify <think> blocks:
1. Prefill: append steering text after <think> in the prompt (single-phase)
2. Sentence Insertion: insert text at a random sentence boundary (two-phase)
3. Sentence Resampling: truncate at hacking-pattern sentence, regenerate (two-phase)

Important: TRL's _generate_single_turn returns completion_ids that start AFTER
the prompt. In think mode, the prompt ends with ``<think>\\n``, so the completion
text starts with the think content directly (no ``<think>`` tag) and contains
``</think>`` to close the block.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

# ── Text utilities ──

# Sentence boundary: period/question/exclamation followed by space or newline
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.?!])(?:\s)")


def find_sentence_boundaries(text: str) -> list[int]:
    """Find character positions of sentence boundaries in text.

    Returns positions immediately after the punctuation + whitespace,
    i.e. the start of the next sentence.
    """
    return [m.end() for m in _SENTENCE_BOUNDARY_RE.finditer(text)]


def split_think_block(completion_text: str) -> tuple[str, int, str]:
    """Split completion into think content and the rest.

    Handles two cases:
    - Completion starts with think content directly (``<think>`` was in prompt)
    - Completion starts with ``<think>`` tag (model regenerated it)

    Returns (think_content, content_start, after_think) where:
    - think_content: text of the thinking block
    - content_start: char offset where think content starts in completion_text
    - after_think: text after ``</think>`` (empty if unclosed)
    """
    start = 0
    open_idx = completion_text.find("<think>")
    if open_idx != -1:
        start = open_idx + len("<think>")
        # Skip optional newline after <think>
        if start < len(completion_text) and completion_text[start] == "\n":
            start += 1

    close_idx = completion_text.find("</think>", start)
    if close_idx == -1:
        return completion_text[start:], start, ""

    return completion_text[start:close_idx], start, completion_text[close_idx + len("</think>"):]


# ── Default hacking patterns for resampling ──

DEFAULT_HACKING_PATTERNS = [
    r"overwrite.*test",
    r"override.*check",
    r"redefine.*check",
    r"bypass.*test",
    r"run_tests\(\)",
    r"exploit",
    r"shortcut",
    r"hack.*reward",
    r"replace.*check",
    r"monkey.?patch",
]


# ── Strategy classes ──

@dataclass
class CotEditingStrategy:
    """Base class for CoT editing strategies.

    Implements the decide/assemble pattern:
    - decide(): analyzes prompt+completion, returns a decision dict (no generation)
    - assemble(): combines decision prefix with regenerated continuation
    """

    @property
    def name(self) -> str:
        return "none"

    @property
    def needs_two_phase(self) -> bool:
        return False

    def decide(self, prompt_text: str, completion_text: str) -> dict:
        """Decide what editing action to take (no generation involved).

        Returns a dict with at minimum:
            action: "none" | "insert" | "resample"
            prefix: text before the regeneration point (if action != "none")
            insert_text: text to insert (for insertion strategy)
        """
        return {"action": "none"}

    def assemble(self, decision: dict, continuation: str) -> str:
        """Assemble final completion from decision + regenerated continuation."""
        if decision["action"] == "none":
            return decision.get("original", "")
        prefix = decision.get("prefix", "")
        insert_text = decision.get("insert_text", "")
        return prefix + insert_text + continuation


@dataclass
class PrefillStrategy(CotEditingStrategy):
    """Append steering text after ``<think>\\n`` in the prompt before generation.

    Single-phase: modifies the prompt, no post-generation editing needed.
    The prefill tokens are part of the prompt, naturally excluded from GRPO loss.
    """

    text: str = ""

    @property
    def name(self) -> str:
        return "prefill"

    @property
    def needs_two_phase(self) -> bool:
        return False

    def apply_to_prompt(self, prompt_text: str) -> tuple[str, dict]:
        """Append ``<think>\\n`` + prefill text after the assistant prompt marker.

        Qwen3's chat template does NOT include ``<think>`` in the prompt —
        the model generates it as its first token. We insert ``<think>\\n``
        plus the prefill text so the model continues from the prefilled CoT.

        Returns (modified_prompt, metadata) where metadata tracks whether
        the edit was applied.
        """
        # Find the assistant generation prompt marker
        target = "<|im_start|>assistant\n"
        idx = prompt_text.rfind(target)
        if idx == -1:
            return prompt_text, {"cot_edited": False}

        insert_pos = idx + len(target)
        modified = (
            prompt_text[:insert_pos]
            + "<think>\n" + self.text + "\n"
            + prompt_text[insert_pos:]
        )
        return modified, {"cot_edited": True}

    def decide(self, prompt_text: str, completion_text: str) -> dict:
        # Prefill is handled at prompt level, not post-generation
        return {"action": "none", "original": completion_text}


@dataclass
class SentenceInsertionStrategy(CotEditingStrategy):
    """Insert steering text at a random sentence boundary in the think block.

    Two-phase:
    1. Generate full completion normally
    2. Find sentence boundaries in think block, pick one randomly
    3. Regenerate from that point with inserted text
    """

    text: str = ""
    probability: float = 1.0

    @property
    def name(self) -> str:
        return "insertion"

    @property
    def needs_two_phase(self) -> bool:
        return True

    def decide(self, prompt_text: str, completion_text: str) -> dict:
        if self.probability < 1.0 and random.random() > self.probability:
            return {"action": "none", "original": completion_text}

        think_content, content_start, _ = split_think_block(completion_text)
        if not think_content.strip():
            return {"action": "none", "original": completion_text}

        boundaries = find_sentence_boundaries(think_content)
        if not boundaries:
            return {"action": "none", "original": completion_text}

        # Pick a random sentence boundary
        boundary = random.choice(boundaries)
        cut_pos = content_start + boundary
        prefix = completion_text[:cut_pos]
        return {
            "action": "insert",
            "prefix": prefix,
            "insert_text": self.text + " ",
        }


@dataclass
class SentenceResamplingStrategy(CotEditingStrategy):
    """Truncate at hacking-pattern sentences and regenerate.

    Two-phase:
    1. Generate full completion normally
    2. Search think block for sentences matching hacking patterns
    3. If found, truncate at start of matching sentence and regenerate
    """

    patterns: list[str] = field(default_factory=lambda: list(DEFAULT_HACKING_PATTERNS))
    _compiled: list[re.Pattern] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self):
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    @property
    def name(self) -> str:
        return "resampling"

    @property
    def needs_two_phase(self) -> bool:
        return True

    def _find_hacking_sentence(self, think_content: str) -> int | None:
        """Find start position of first sentence matching a hacking pattern.

        Returns character offset within think_content, or None if no match.
        """
        boundaries = [0] + find_sentence_boundaries(think_content)
        for i in range(len(boundaries)):
            start = boundaries[i]
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(think_content)
            sentence = think_content[start:end]
            for pat in self._compiled:
                if pat.search(sentence):
                    return start
        return None

    def decide(self, prompt_text: str, completion_text: str) -> dict:
        think_content, content_start, _ = split_think_block(completion_text)
        if not think_content.strip():
            return {"action": "none", "original": completion_text}

        match_pos = self._find_hacking_sentence(think_content)
        if match_pos is None:
            return {"action": "none", "original": completion_text}

        cut_pos = content_start + match_pos
        prefix = completion_text[:cut_pos]
        return {
            "action": "resample",
            "prefix": prefix,
            "insert_text": "",
        }


# ── Factory ──

STRATEGY_REGISTRY: dict[str, type[CotEditingStrategy]] = {
    "none": CotEditingStrategy,
    "prefill": PrefillStrategy,
    "insertion": SentenceInsertionStrategy,
    "resampling": SentenceResamplingStrategy,
}


def make_cot_strategy(
    name: str = "none",
    *,
    prefill_text: str | None = None,
    insertion_text: str | None = None,
    insertion_probability: float = 1.0,
    resampling_patterns: list[str] | None = None,
) -> CotEditingStrategy | None:
    """Create a CoT editing strategy from CLI arguments.

    Returns None if name is "none".
    """
    if name == "none":
        return None

    if name == "prefill":
        if not prefill_text:
            raise ValueError("--cot_prefill_text required for prefill strategy")
        return PrefillStrategy(text=prefill_text)

    if name == "insertion":
        if not insertion_text:
            raise ValueError("--cot_insertion_text required for insertion strategy")
        return SentenceInsertionStrategy(text=insertion_text, probability=insertion_probability)

    if name == "resampling":
        patterns = resampling_patterns or list(DEFAULT_HACKING_PATTERNS)
        return SentenceResamplingStrategy(patterns=patterns)

    raise ValueError(f"Unknown CoT strategy: {name!r}. Available: {list(STRATEGY_REGISTRY)}")
