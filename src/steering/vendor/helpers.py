import os
import pwd
import shutil
import tempfile
import subprocess
import sys
import json
import textwrap
from pydantic import BaseModel

# Resolve sys.executable to real path to avoid symlink issues with subprocess
# This prevents "Bad address" errors when .venv is a symlink
def _get_python_executable():
    """Get the resolved Python executable path, re-resolving each time to avoid stale paths."""
    resolved = os.path.realpath(sys.executable)
    # Verify the path exists and is executable
    if not os.path.exists(resolved) or not os.access(resolved, os.X_OK):
        # Fallback to sys.executable if resolved path is invalid
        return sys.executable
    return resolved

'''
NOTE: The only memory-safe and CPU-safe execution is to use multiprocessing.

The number of workers is determined by CodeEvaluator using the MAX_JOBS environment variable.
Set the variable based on the number of CPUs available.

'''

# --- Sandbox configuration ---
_SANDBOX_USER = "sandbox"

def _get_sandbox_ids():
    """Look up the sandbox user's UID/GID. Returns (uid, gid) or None if user doesn't exist."""
    try:
        pw = pwd.getpwnam(_SANDBOX_USER)
        return (pw.pw_uid, pw.pw_gid)
    except KeyError:
        return None

def _make_preexec_fn(uid, gid):
    """Return a preexec_fn that drops to the sandbox user."""
    def _preexec():
        os.setgroups([gid])
        os.setgid(gid)
        os.setuid(uid)
    return _preexec

def _build_sandbox_env():
    """Build a minimal environment for the subprocess, stripping secrets."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": "/tmp",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "TOKENIZERS_PARALLELISM": "false",
    }
    # Preserve PYTHONPATH if set (needed for package imports)
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    return env

class CodeRunResult(BaseModel):
    success: bool = True # Ran without errors
    compiled: bool = True # Compiled successfully
    timeout: bool = False # Did it timeout?
    oom: bool = False # Did it run out of memory?
    stdout: dict = {}  # Parsed JSON from stdout (dict if compiled, empty dict if not)


_SUBPROCESS_CODE = textwrap.dedent(
    f"""
    import io
    import json
    import resource
    import signal
    import sys
    from contextlib import redirect_stdout

    memory_mb = int(sys.argv[1])
    time_limit = float(sys.argv[2])
    memory_bytes = max(memory_mb, 1) * 1024 * 1024
    cpu_seconds = max(int(time_limit), 1)

    class TimeoutException(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise TimeoutException(f"Execution timed out after {{time_limit}} seconds")

    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except (ValueError, OSError):
        pass

    try:
        resource.setrlimit(resource.RLIMIT_RSS, (memory_bytes, memory_bytes))
    except (ValueError, OSError):
        pass

    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    except (ValueError, OSError):
        pass

    # Limit output file size to 10 MB to prevent disk-filling attacks
    try:
        fsize_bytes = 10 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_bytes, fsize_bytes))
    except (ValueError, OSError):
        pass

    # Limit child processes to 16 to prevent fork bombs
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
    except (ValueError, OSError):
        pass

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(max(int(time_limit), 1))

    stdout_buffer = io.StringIO()
    namespace = {{}}
    code = sys.stdin.read()

    output = {{
        "success": True,
        "compiled": True,
        "timeout": False,
        "oom": False,
        "stdout": {{}}
    }}

    try:
        with redirect_stdout(stdout_buffer):
            exec(code, namespace)
        # Code executed successfully, parse stdout as JSON
        stdout_content = stdout_buffer.getvalue()
        try:
            # Try to parse the last JSON object from stdout
            lines = stdout_content.strip().split('\\n')
            parsed_json = None
            for line in reversed(lines):
                line = line.strip()
                if line.startswith('{{') and line.endswith('}}'):
                    try:
                        parsed_json = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue

            output["stdout"] = parsed_json if parsed_json is not None else {{}}
        except Exception:
            # If parsing fails, return empty dict
            output["stdout"] = {{"raw": stdout_content}}
    except (SyntaxError, IndentationError):
        # Code did not compile
        output["success"] = False
        output["compiled"] = False
    except MemoryError:
        # Memory limit exceeded
        output["success"] = False
        output["oom"] = True
        output["stdout"] = {{}}
    except TimeoutException:
        # Timeout
        output["success"] = False
        output["timeout"] = True
        output["stdout"] = {{}}
    except Exception as e:
        # Other exceptions
        output["success"] = False
        output["stdout"] = {{"raw": str(e)}}
    finally:
        signal.alarm(0)

    sys.stdout.write(json.dumps(output))
    sys.stdout.flush()
    """
).strip()



def _execute_in_subprocess(
    code: str,
    timeout: int,
    memory_limit: int,
    raise_exceptions: bool = False
) -> CodeRunResult:
    """Execute code in an isolated Python subprocess with resource limits.

    Sandbox hardening applied:
    - Drops to non-root 'sandbox' user via preexec_fn (if available)
    - Passes minimal env (strips WANDB_API_KEY, HF_TOKEN, etc.)
    - Runs in a fresh temp directory (cleaned up after)
    - RLIMIT_FSIZE, RLIMIT_NPROC, RLIMIT_AS, RLIMIT_RSS, RLIMIT_CPU

    Returns CodeRunResult with:
    - compiled: bool indicating if code compiled successfully
    - stdout: dict parsed from JSON in stdout (empty dict if not compiled)
    """
    # Re-resolve executable path each time to avoid stale paths in concurrent scenarios
    python_executable = _get_python_executable()

    args = [
        python_executable,
        "-c",
        _SUBPROCESS_CODE,
        str(max(memory_limit, 1)),
        str(max(timeout, 1)),
    ]

    # Build sandbox preexec_fn (drop to non-root user)
    sandbox_ids = _get_sandbox_ids()
    preexec_fn = _make_preexec_fn(*sandbox_ids) if sandbox_ids else None

    # Build minimal environment (strips secrets)
    sandbox_env = _build_sandbox_env()

    # Create isolated temp working directory
    tmpdir = tempfile.mkdtemp(prefix="sandbox_")
    if sandbox_ids:
        os.chown(tmpdir, sandbox_ids[0], sandbox_ids[1])

    process = None
    stdout = ""
    stderr = ""
    returncode = 0

    try:
        process = subprocess.Popen(
            args,
            executable=python_executable,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            close_fds=True,
            preexec_fn=preexec_fn,
            env=sandbox_env,
            cwd=tmpdir,
        )
        stdout, stderr = process.communicate(input=code, timeout=max(timeout, 1) + 1)
        returncode = process.returncode
        process = None
    except subprocess.TimeoutExpired:
        if process is not None:
            try:
                process.kill()
                process.wait()
            except Exception:
                pass
            process = None
        # Timeout - treat as timeout
        return CodeRunResult(
            success = False,
            timeout = True
        )
    except OSError as e:
        # Handle "Bad address" and other OS errors gracefully
        if process is not None:
            try:
                process.kill()
                process.wait()
            except Exception:
                pass
            process = None
        if raise_exceptions:
            raise e
        return CodeRunResult(
            success = False,
            compiled = False,
            stdout = {"raw": f"OSError: {str(e)}"},
        )
    except Exception as e:
        if process is not None:
            try:
                process.kill()
                process.wait()
            except Exception:
                pass
            process = None
        if raise_exceptions:
            raise e
        return CodeRunResult(
            success = False,
            stdout = {"raw": str(e)},
        )
    finally:
        if process is not None:
            try:
                process.kill()
                process.wait()
            except Exception:
                pass
        # Clean up temp directory
        shutil.rmtree(tmpdir, ignore_errors=True)


    stdout_str = stdout.strip()
    try:
        result = json.loads(stdout_str) # This should be a dictionary in the form of CodeRunResult
        if returncode != 0:
            result['success'] = False

        return CodeRunResult(
            **result
        )
    except json.JSONDecodeError:
        # Failed to parse JSON - treat as failure to execute child harness
        return CodeRunResult(
            success = False,
            compiled = False,
            stdout = {"raw": stdout_str},
        )


def run_code_subprocess(
    program: str,
    memory_limit: int = 1024,
    timeout: int = 1,
    debug: bool = False,
) -> CodeRunResult:
    """Execute a single program in an isolated subprocess.

    Returns CodeRunResult
    """
    # TOKENIZERS_PARALLELISM=false is now set in the sandbox env directly
    result = _execute_in_subprocess(
        program,
        timeout=timeout,
        memory_limit=memory_limit,
    )

    if debug:
        print("Run test result", result.compiled, result.stdout)

    return result


def create_test_runner_code(setup_code: str, program: str, test_list: list[str], max_failures: int) -> str:
        """
        Create a single Python code string that runs all tests and returns counts.

        The code will:
        1. Execute setup_code and program
        2. Run each test, counting successes and failures
        3. Capture exception types for failures (matching helpers.py pattern)
        4. Stop after max_failures failures
        5. Print JSON with tests_evaluated, tests_passed, and exception types to stdout
        """

        # Escape the test list for safe insertion into the code string
        test_list_repr = repr(test_list)
        max_failures_repr = repr(max_failures)

        # Create the test runner code using the same exception handling pattern as helpers.py
        test_runner = f"""{setup_code}

{program}

# Test runner - using same exception handling pattern as subprocess code
import json
import sys

tests_evaluated = 0
tests_passed = 0
test_errors = []
failures_count = 0

test_cases = {test_list_repr}

for test_case in test_cases:
    try:
        tests_evaluated += 1
        exec(test_case)
        tests_passed += 1
    except AssertionError as e:
        # Test assertion failed
        exception_type = "AssertionError"
        test_errors.append(f"{{exception_type}}: {{str(e)}}")
        failures_count += 1
        if failures_count >= {max_failures_repr}:
            break
    except (SyntaxError, IndentationError) as e:
        # Syntax error in test
        exception_type = type(e).__name__
        test_errors.append(f"{{exception_type}}: {{str(e)}}")
        failures_count += 1
        if failures_count >= {max_failures_repr}:
            break
    except SystemExit as e:
        # SystemExit in test
        exception_type = "SystemExit"
        test_errors.append(f"{{exception_type}}: {{str(e)}}")
        failures_count += 1
        if failures_count >= {max_failures_repr}:
            break
    except BaseException as e:
        # Other exceptions (RuntimeError, TypeError, etc.)
        exception_type = type(e).__name__
        test_errors.append(f"{{exception_type}}: {{str(e)}}")
        failures_count += 1
        if failures_count >= {max_failures_repr}:
            break

# Print result as JSON to stdout
result = {{
    "tests_evaluated": tests_evaluated,
    "tests_passed": tests_passed,
    "tests_total": len(test_cases),
    "test_errors": test_errors
}}
print(json.dumps(result))
sys.stdout.flush()
"""
        return test_runner
