"""Tests for subprocess sandbox hardening in cot_editing.vendor.helpers."""

import os
import pytest

from cot_editing.vendor.helpers import run_code_subprocess, _get_sandbox_ids


# --- Tests ---


class TestSandboxNonRoot:
    """Verify subprocess runs as non-root sandbox user."""

    @pytest.mark.skipif(_get_sandbox_ids() is None, reason="sandbox user not created")
    def test_runs_as_non_root(self):
        code = "import os, json; print(json.dumps({'uid': os.getuid(), 'gid': os.getgid()}))"
        result = run_code_subprocess(code, timeout=5)
        assert result.success, f"Code failed: {result.stdout}"
        assert result.stdout.get("uid") != 0, "Subprocess should NOT run as root (uid 0)"

    @pytest.mark.skipif(_get_sandbox_ids() is None, reason="sandbox user not created")
    def test_runs_as_sandbox_user(self):
        sandbox_ids = _get_sandbox_ids()
        code = "import os, json; print(json.dumps({'uid': os.getuid(), 'gid': os.getgid()}))"
        result = run_code_subprocess(code, timeout=5)
        assert result.success
        assert result.stdout["uid"] == sandbox_ids[0]
        assert result.stdout["gid"] == sandbox_ids[1]


class TestSandboxEnv:
    """Verify subprocess environment is minimal and secret-free."""

    def test_no_wandb_key(self):
        os.environ["WANDB_API_KEY"] = "test-secret-key"
        try:
            code = "import os, json; print(json.dumps({'has_wandb': 'WANDB_API_KEY' in os.environ}))"
            result = run_code_subprocess(code, timeout=5)
            assert result.success
            assert result.stdout["has_wandb"] is False, "WANDB_API_KEY should be stripped from subprocess env"
        finally:
            del os.environ["WANDB_API_KEY"]

    def test_no_hf_token(self):
        os.environ["HF_TOKEN"] = "hf_test_secret"
        try:
            code = "import os, json; print(json.dumps({'has_hf': 'HF_TOKEN' in os.environ}))"
            result = run_code_subprocess(code, timeout=5)
            assert result.success
            assert result.stdout["has_hf"] is False, "HF_TOKEN should be stripped from subprocess env"
        finally:
            del os.environ["HF_TOKEN"]

    def test_no_openrouter_key(self):
        os.environ["OPENROUTER_API_KEY"] = "or-test-secret"
        try:
            code = "import os, json; print(json.dumps({'has_or': 'OPENROUTER_API_KEY' in os.environ}))"
            result = run_code_subprocess(code, timeout=5)
            assert result.success
            assert result.stdout["has_or"] is False, "OPENROUTER_API_KEY should be stripped from subprocess env"
        finally:
            del os.environ["OPENROUTER_API_KEY"]

    def test_has_path(self):
        code = "import os, json; print(json.dumps({'has_path': 'PATH' in os.environ}))"
        result = run_code_subprocess(code, timeout=5)
        assert result.success
        assert result.stdout["has_path"] is True

    def test_tokenizers_parallelism_false(self):
        code = "import os, json; print(json.dumps({'tp': os.environ.get('TOKENIZERS_PARALLELISM')}))"
        result = run_code_subprocess(code, timeout=5)
        assert result.success
        assert result.stdout["tp"] == "false"

    def test_env_is_minimal(self):
        """Subprocess env should only contain the allowlisted keys."""
        code = "import os, json; print(json.dumps({'keys': sorted(os.environ.keys())}))"
        result = run_code_subprocess(code, timeout=5)
        assert result.success
        allowed = {"PATH", "HOME", "LANG", "TOKENIZERS_PARALLELISM", "PYTHONPATH"}
        actual = set(result.stdout["keys"])
        unexpected = actual - allowed
        assert not unexpected, f"Unexpected env vars in subprocess: {unexpected}"
        # Verify required vars are always present
        required = {"PATH", "HOME", "LANG", "TOKENIZERS_PARALLELISM"}
        missing = required - actual
        assert not missing, f"Missing required env vars in subprocess: {missing}"


class TestSandboxTmpdir:
    """Verify subprocess runs in an isolated temp directory."""

    def test_cwd_is_temp_directory(self):
        code = "import os, json; print(json.dumps({'cwd': os.getcwd()}))"
        result = run_code_subprocess(code, timeout=5)
        assert result.success
        cwd = result.stdout["cwd"]
        assert "/tmp" in cwd or "sandbox_" in cwd, f"Expected temp dir, got: {cwd}"
        # Verify the tmpdir was cleaned up
        assert not os.path.exists(cwd), f"Temp dir should be cleaned up after execution: {cwd}"

    def test_file_writes_are_isolated(self):
        """Files written in subprocess should not appear in project root."""
        code = (
            "import os, json\n"
            "with open('sandbox_test_artifact.txt', 'w') as f:\n"
            "    f.write('test')\n"
            "print(json.dumps({'wrote': True, 'cwd': os.getcwd()}))"
        )
        result = run_code_subprocess(code, timeout=5)
        assert result.success
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assert not os.path.exists(os.path.join(project_root, "sandbox_test_artifact.txt"))


class TestSandboxRlimits:
    """Verify rlimits are enforced."""

    @pytest.mark.skipif(_get_sandbox_ids() is None, reason="sandbox user not created (NPROC needs non-root)")
    def test_fork_bomb_protection(self):
        """RLIMIT_NPROC should prevent excessive forking."""
        code = (
            "import os, json\n"
            "pids = []\n"
            "try:\n"
            "    for i in range(100):\n"
            "        pid = os.fork()\n"
            "        if pid == 0:\n"
            "            os._exit(0)\n"
            "        pids.append(pid)\n"
            "except OSError as e:\n"
            "    for p in pids:\n"
            "        try:\n"
            "            os.waitpid(p, 0)\n"
            "        except Exception:\n"
            "            pass\n"
            "    print(json.dumps({'forked': len(pids), 'error': str(e)}))\n"
            "else:\n"
            "    for p in pids:\n"
            "        try:\n"
            "            os.waitpid(p, 0)\n"
            "        except Exception:\n"
            "            pass\n"
            "    print(json.dumps({'forked': len(pids), 'error': None}))\n"
        )
        result = run_code_subprocess(code, timeout=10)
        assert result.success
        # Should have been limited -- either got an error or forked less than 100
        forked = result.stdout.get("forked", 0)
        error = result.stdout.get("error")
        assert forked < 100 or error is not None, (
            f"Fork bomb protection failed: forked {forked} times without error"
        )


class TestSandboxFunctionality:
    """Verify existing CodeEvaluator functionality still works through sandbox."""

    def test_simple_code_runs(self):
        result = run_code_subprocess("import json; print(json.dumps({'result': 42}))")
        assert result.success
        assert result.stdout["result"] == 42

    def test_compilation_failure(self):
        result = run_code_subprocess("def foo(")
        assert not result.compiled

    def test_timeout(self):
        result = run_code_subprocess("import time; time.sleep(100)", timeout=1)
        assert not result.success
        assert result.timeout

    def test_assertion_in_test_runner(self):
        """Test that the test runner pattern still works end-to-end."""
        from cot_editing.vendor.helpers import create_test_runner_code
        code = create_test_runner_code(
            setup_code="",
            program="def add(a, b): return a + b",
            test_list=["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
            max_failures=5,
        )
        result = run_code_subprocess(code, timeout=5)
        assert result.success
        assert result.stdout["tests_passed"] == 2
        assert result.stdout["tests_evaluated"] == 2
