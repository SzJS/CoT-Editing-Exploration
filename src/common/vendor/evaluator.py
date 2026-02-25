"""Code evaluation engine: parse, compile-check, and test model-generated Python code."""

import os
import re
import ast
from typing import List, TypedDict

from common.vendor import helpers

class CodeEvaluationResult(TypedDict):
    """Result of evaluating a code completion against test cases.

    Attributes:
        parsed_response: Extracted Python code from the response, or None if parsing failed.
        is_formatted: Whether the response contained a valid code block.
        can_compile: Whether the extracted code compiled without syntax errors.
        pass_rate: Fraction of tests passed (0.0 to 1.0).
        tests_passed: Number of test assertions that passed.
        tests_total: Total number of test assertions.
        tests_evaluated: Number of tests actually executed (may be < total if max_failures hit).
        test_errors: Exception type strings for tests that failed.
    """

    parsed_response: str | None
    is_formatted: bool
    can_compile: bool
    pass_rate: float
    tests_passed: int
    tests_total: int
    tests_evaluated: int
    test_errors: List[str]

class CodeEvaluator:
    """Evaluates model-generated code against test suites in sandboxed subprocesses.

    Thread-safe: each evaluation runs in its own subprocess with resource limits.
    """

    name: str = "code"
    debug: bool = True

    def __init__(
        self,
        num_workers: int | None = None,
        memory_per_worker: int = 1024,
        timeout: int = 3, # NOTE: If this is changed, need to re-filter the base dataset to use the new timeout
        max_failures: int = 1,
        debug: bool = False,
    ):
        """Initialize the code evaluator.

        Args:
            num_workers: Max concurrent evaluation subprocesses. Defaults to MAX_JOBS env var or 1.
            memory_per_worker: Memory limit per subprocess in MB.
            timeout: Time limit per test execution in seconds.
            max_failures: Stop evaluating tests after this many failures per sample.
            debug: If True, print detailed subprocess output.
        """
        self.num_workers = num_workers if num_workers is not None else int(os.environ.get('MAX_JOBS', 1)) # Maximum total number of jobs runnng at once
        self.memory_per_worker = memory_per_worker
        self.timeout = timeout
        self.debug = debug
        self.max_failures = max_failures

    def __call__(
        self,
        response: str | None,
        test_list: List[str] | None = None, # List of assertion statements to test the code
        setup_code: str = "", # Code to run before tests (e.g., imports)
        skip_parse: bool = True # Response is already parsed into code (usually True)
    ) -> CodeEvaluationResult | float:
        """Evaluate a code response against test cases in a sandboxed subprocess.

        Args:
            response: Model-generated text (may contain markdown code fences).
            test_list: List of assert statement strings to test the code.
            setup_code: Code to run before tests (e.g., imports).
            skip_parse: If True, treat response as raw Python code (skip code-fence extraction).

        Returns:
            CodeEvaluationResult dict with parse status, compilation status,
            pass rate, and error details.
        """
        if test_list is None:
            test_list = []

        result = CodeEvaluationResult(**{
            'parsed_response': None,
            'is_formatted': True,
            'can_compile': True,
            'pass_rate': 0.0,
            'tests_passed': 0,
            'tests_evaluated': 0,
            'tests_total': len(test_list),
            'test_errors': [],
        })

        # Parse the program and ensure format
        if not skip_parse:
            program = self.parse_response(response)
        else:
            program = response

        if program is None:
            result['is_formatted'] = False
            result['can_compile'] = False
            return result

        result['parsed_response'] = program

        # Combine all tests into a single subprocess execution
        # Create a test runner that counts tests passed and evaluated
        # Uses max_failures to stop after N failures (per sample)
        test_runner_code = helpers.create_test_runner_code(
            setup_code,
            program,
            test_list,
            self.max_failures
        )

        code_run_result: helpers.CodeRunResult = helpers.run_code_subprocess(
            test_runner_code,
            timeout=self.timeout,
            memory_limit=self.memory_per_worker,
            debug=self.debug,
        )

        # If compilation failed, return this
        result['can_compile'] = code_run_result.compiled
        if not code_run_result.compiled:
            result['test_errors'].append("MasterError: CompilationError")
        if code_run_result.timeout:
            result['test_errors'].append("MasterError: TimeoutError")
        if code_run_result.oom:
            result['test_errors'].append("MasterError: OOMError")
        if not code_run_result.success:
            result['test_errors'].append("MasterError: UnknownError: " + str(code_run_result.stdout.get('raw', 'No response')))

        # Extract the results from the stdout field
        result['tests_evaluated'] = code_run_result.stdout.get('tests_evaluated', 0)
        result['tests_passed'] = code_run_result.stdout.get('tests_passed', 0)
        result['pass_rate'] = (result['tests_passed'] / result['tests_total']) if result['tests_total'] > 0 else 0.0
        result['test_errors'] += code_run_result.stdout.get('test_errors', [])

        return result


    def parse_response(self, response: str) -> str | None:
        """Extract Python code from markdown-fenced code blocks in a response.

        Finds all ```python or ``` blocks and joins their contents.

        Args:
            response: Raw model response text containing code fences.

        Returns:
            Concatenated code from all fenced blocks, or None if no blocks found.
        """
        # Extract all fenced python code blocks (or unlabeled) and join with double newlines
        blocks  =  re.findall(r"```(?:python)?\n(.*?)(?:```|$)", response, re.DOTALL | re.IGNORECASE)
        if not blocks:
            return None
        cleaned_blocks  =  [b.strip() for b in blocks if b.strip()]
        if not cleaned_blocks:
            return None
        return "\n\n".join(cleaned_blocks)

    def extract_function_parent(self, code_str: str, func_name: str) -> str | None:
        """Extract the parent class name of a function if it's defined as a method.

        Uses an AST visitor to track class context while searching for the function.

        Args:
            code_str: Python source code string to search.
            func_name: Name of the function/method to locate.

        Returns:
            The parent class name if the function is a method, or None if the
            function is top-level, not found, or the code fails to parse.
        """
        try:
            tree = ast.parse(code_str)
        except Exception:
            return None

        # Use a visitor to track class context
        class FunctionParentExtractor(ast.NodeVisitor):
            def __init__(self, target_name: str):
                self.target_name = target_name
                self.class_name = None
                self.current_class = None

            def visit_ClassDef(self, node):
                # Save the current class name and visit children
                old_class = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = old_class

            def visit_FunctionDef(self, node):
                if node.name == self.target_name:
                    self.class_name = self.current_class
                self.generic_visit(node)

        extractor = FunctionParentExtractor(func_name)
        extractor.visit(tree)

        return extractor.class_name


