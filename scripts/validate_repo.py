from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "results",
}


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def iter_python_files():
    for path in ROOT.rglob("*.py"):
        if should_skip(path):
            continue
        yield path


def security_check():
    issues = []

    suspicious_strings = [
        "hf_",
        "sk-",
        "aws_secret_access_key",
        "aws_access_key_id",
        "BEGIN PRIVATE KEY",
    ]

    this_file = Path(__file__).resolve()

    for path in iter_python_files():

        # Skip this validator itself because it literally contains
        # the credential-marker strings that it is searching for.
        if path.resolve() == this_file:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            issues.append(
                f"Could not decode as UTF-8: {path.relative_to(ROOT)}"
            )
            continue

        lower_text = text.lower()

        for token in suspicious_strings:
            if token.lower() in lower_text:

                # Allow normal references to the Hugging Face env var.
                if token == "hf_" and "HF_TOKEN" in text:
                    continue

                issues.append(
                    f"Potential embedded credential marker '{token}' "
                    f"in {path.relative_to(ROOT)}"
                )

        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            issues.append(
                f"Syntax error in {path.relative_to(ROOT)}: {exc}"
            )
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {
                    "eval",
                    "exec",
                }:
                    issues.append(
                        f"Direct {node.func.id}() usage in "
                        f"{path.relative_to(ROOT)}:{node.lineno}"
                    )

    return {
        "passed": len(issues) == 0,
        "issues": issues,
    }


def quality_check():
    issues = []

    required_files = [
        ROOT / "modal_app.py",
        ROOT / "requirements-local.txt",
        ROOT / "README.md",
    ]

    for path in required_files:
        if not path.exists():
            issues.append(
                f"Missing required file: {path.relative_to(ROOT)}"
            )

    for path in iter_python_files():
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            issues.append(
                f"Could not decode as UTF-8: {path.relative_to(ROOT)}"
            )
            continue

        try:
            compile(source, str(path), "exec")
        except Exception as exc:
            issues.append(
                f"Compilation failed for {path.relative_to(ROOT)}: {exc}"
            )

    return {
        "passed": len(issues) == 0,
        "issues": issues,
    }


def main():
    security = security_check()
    quality = quality_check()

    report = {
        "security_check": security,
        "quality_check": quality,
    }

    print(json.dumps(report, indent=2))

    if not security["passed"] or not quality["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()