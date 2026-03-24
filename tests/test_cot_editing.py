"""Tests for CoT editing strategies, including Harmony format support."""

from common.cot_editing import PrefillStrategy


class TestPrefillStrategy:
    """Test PrefillStrategy.apply_to_prompt() for both formats."""

    def setup_method(self):
        self.strategy = PrefillStrategy(text="I should solve this correctly.")

    def test_qwen3_chatml_format(self):
        """Prefill injects <think> block after Qwen3/ChatML assistant marker."""
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\nSolve this problem.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        modified, meta = self.strategy.apply_to_prompt(prompt)

        assert meta["cot_edited"] is True
        assert meta["format"] == "think"
        assert "<think>\nI should solve this correctly.\n" in modified
        # Original structure preserved
        assert modified.startswith("<|im_start|>system")
        assert "<|im_start|>assistant\n<think>\n" in modified

    def test_harmony_format(self):
        """Prefill injects analysis channel after Harmony assistant marker."""
        prompt = (
            "<|start|>system<|message|>You are a helpful assistant.<|end|>"
            "<|start|>user<|message|>Solve this problem.<|end|>"
            "<|start|>assistant"
        )
        modified, meta = self.strategy.apply_to_prompt(prompt)

        assert meta["cot_edited"] is True
        assert meta["format"] == "harmony"
        assert "<|channel|>analysis<|message|>I should solve this correctly." in modified
        # Original structure preserved
        assert modified.startswith("<|start|>system")
        assert "<|start|>assistant<|channel|>analysis<|message|>" in modified

    def test_harmony_no_false_match_on_system(self):
        """Harmony detection matches the last <|start|>assistant, not system/user."""
        prompt = (
            "<|start|>system<|message|>Test<|end|>"
            "<|start|>user<|message|>Hello<|end|>"
            "<|start|>assistant"
        )
        modified, meta = self.strategy.apply_to_prompt(prompt)

        assert meta["cot_edited"] is True
        # Prefill should appear right after the last <|start|>assistant
        assert modified.endswith(
            "<|start|>assistant<|channel|>analysis<|message|>I should solve this correctly."
        )

    def test_unknown_format_returns_unedited(self):
        """Unknown prompt format returns unmodified prompt."""
        prompt = "Some random prompt without markers"
        modified, meta = self.strategy.apply_to_prompt(prompt)

        assert meta["cot_edited"] is False
        assert modified == prompt

    def test_empty_prefill_text(self):
        """Empty prefill text still marks as edited."""
        strategy = PrefillStrategy(text="")
        prompt = "<|start|>system<|message|>Test<|end|><|start|>assistant"
        modified, meta = strategy.apply_to_prompt(prompt)

        assert meta["cot_edited"] is True
        assert "<|channel|>analysis<|message|>" in modified

    def test_qwen3_takes_priority_over_harmony(self):
        """If both markers present (unlikely), Qwen3/ChatML is tried first."""
        # This is a degenerate case but tests priority
        prompt = (
            "<|start|>assistant"
            "<|im_start|>assistant\n"
        )
        modified, meta = self.strategy.apply_to_prompt(prompt)

        assert meta["format"] == "think"
        assert "<think>\n" in modified
