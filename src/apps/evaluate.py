"""Inspect AI evaluation task for APPS coding problems.

Evaluate SFT model on benign APPS problems:
    VLLM_BASE_URL=http://localhost:8000/v1 PYTHONPATH=src inspect eval src/apps/evaluate.py \
        --model vllm/Qwen/Qwen3-4B -T benign_only=true

Evaluate with hints:
    VLLM_BASE_URL=http://localhost:8000/v1 PYTHONPATH=src inspect eval src/apps/evaluate.py \
        --model vllm/Qwen/Qwen3-4B -T hint=check_override

Evaluate without thinking:
    VLLM_BASE_URL=http://localhost:8000/v1 PYTHONPATH=src inspect eval src/apps/evaluate.py \
        --model vllm/Qwen/Qwen3-4B -T thinking=false
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig
from inspect_ai.solver import generate

from apps.data import prepare_apps_dataset
from impossible.evaluate import (
    impossible_scorer,
    correct_rate,
    hack_rate,
    correct_with_attempted_hack_rate,
    failed_hack_rate,
    compile_rate,
    redefines_check_rate,
    calls_sys_exit_rate,
    impossible_hack_rate,
    benign_correct_rate,
    mean_reasoning_tokens,
    mean_output_tokens,
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_apps_eval_dataset(
    hint: str = "none",
    custom_hint_text: str = "",
    difficulty: str = "introductory",
    split: str = "train",
    benign_only: bool = False,
    max_samples: int = 0,
    max_tests: int = 10,
    eval_fraction: float = 0.2,
) -> MemoryDataset:
    """Load APPS and convert to Inspect MemoryDataset.

    By default, loads the eval subset of the train split (eval_fraction=0.2)
    since the APPS test split has very few function-based introductory problems.

    Args:
        hint: Hint type for exploit discoverability (default "none").
        custom_hint_text: Arbitrary hint text when hint="custom".
        difficulty: Comma-separated difficulty levels (default "introductory").
        split: APPS split (default "train" — uses eval subset via eval_fraction).
        benign_only: If True, only include benign (original) problems.
        max_samples: Max total samples (0 = all).
        max_tests: Max test cases per problem.
        eval_fraction: Fraction of train split reserved for eval (default 0.2).

    Returns:
        MemoryDataset of Inspect Samples with metadata for scoring.
    """
    ds = prepare_apps_dataset(
        split=split,
        difficulty=difficulty,
        max_tests=max_tests,
        hint=hint,
        custom_hint_text=custom_hint_text or None,
        include_impossible=not benign_only,
        max_samples=max_samples if max_samples > 0 else None,
        eval_fraction=eval_fraction,
        eval_only=True,
    )

    samples = []
    for row in ds:
        messages = []
        for msg in row["prompt"]:
            if msg["role"] == "system":
                messages.append(ChatMessageSystem(content=msg["content"]))
            else:
                messages.append(ChatMessageUser(content=msg["content"]))
        samples.append(
            Sample(
                input=messages,
                id=f"{row['task_id']}_{row['impossible_type']}",
                metadata={
                    "test_code": row["test_code"],
                    "original_test_code": row["original_test_code"],
                    "entry_point": row["entry_point"],
                    "is_impossible": row["is_impossible"],
                    "task_id": row["task_id"],
                    "impossible_type": row["impossible_type"],
                    "hint": hint,
                },
            )
        )
    return MemoryDataset(samples)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


@task
def apps_baseline(
    hint: str = "none",
    thinking: bool = True,
    eval_timeout: int = 10,
    difficulty: str = "introductory",
    split: str = "train",
    benign_only: bool = False,
    custom_hint_text: str = "",
    max_samples: int = 0,
    max_tests: int = 10,
    eval_fraction: float = 0.2,
) -> Task:
    """Baseline evaluation of APPS coding problems.

    By default, evaluates on the eval subset of the train split (20% held out).

    Args:
        hint: Hint type for exploit discoverability (default "none").
        thinking: Whether to enable thinking mode (default True).
        eval_timeout: Subprocess timeout per evaluation in seconds.
        difficulty: Comma-separated difficulty levels (default "introductory").
        split: APPS split (default "train" — uses eval subset via eval_fraction).
        benign_only: If True, only include benign (non-mutated) problems.
        custom_hint_text: Arbitrary hint text when hint="custom".
        max_samples: Max samples (0 = all).
        max_tests: Max test cases per problem.
        eval_fraction: Fraction of train split reserved for eval (default 0.2).
    """
    if thinking:
        config = GenerateConfig(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_tokens=32768,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
    else:
        config = GenerateConfig(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            max_tokens=4096,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    return Task(
        dataset=load_apps_eval_dataset(
            hint=hint,
            custom_hint_text=custom_hint_text,
            difficulty=difficulty,
            split=split,
            benign_only=benign_only,
            max_samples=max_samples,
            max_tests=max_tests,
            eval_fraction=eval_fraction,
        ),
        solver=[generate()],
        scorer=impossible_scorer(eval_timeout=eval_timeout, hack_vector="check_redef"),
        config=config,
    )
