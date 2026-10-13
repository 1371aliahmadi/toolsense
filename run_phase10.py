"""
Phase 10 Runner — Docker + Kubernetes + DVC validation
========================================================
Validates that all Phase 10 artefacts are in place and
shows the commands needed to use them.

Usage:
    python run_phase10.py
"""

import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent


def check(label: str, condition: bool, hint: str = "") -> bool:
    status = "OK " if condition else "ERR"
    print(f"  {status}  {label}")
    if not condition and hint:
        print(f"       > {hint}")
    return condition


def cmd_available(name: str) -> bool:
    return subprocess.run(
        ["where" if sys.platform == "win32" else "which", name],
        capture_output=True,
    ).returncode == 0


def main():
    print("=" * 60)
    print("PHASE 10 — DevOps Validation")
    print("=" * 60)

    all_ok = True

    # ── File artefacts ─────────────────────────────────────────────────────────
    print("\n[1] Required files:")
    files = {
        "Dockerfile":                    root / "Dockerfile",
        "docker-compose.yml":            root / "docker-compose.yml",
        ".dockerignore":                 root / ".dockerignore",
        "requirements-api.txt":          root / "requirements-api.txt",
        "k8s/deployment.yaml":           root / "k8s" / "deployment.yaml",
        "k8s/service.yaml":              root / "k8s" / "service.yaml",
        ".github/workflows/ci.yml":      root / ".github" / "workflows" / "ci.yml",
        "dvc.yaml":                      root / "dvc.yaml",
        ".dvcignore":                    root / ".dvcignore",
        "tests/test_api.py":             root / "tests" / "test_api.py",
        "tests/test_ranker.py":          root / "tests" / "test_ranker.py",
    }
    for name, path in files.items():
        ok = check(name, path.exists(), f"Create {path}")
        all_ok = all_ok and ok

    # ── CLI tools ──────────────────────────────────────────────────────────────
    print("\n[2] CLI tools:")
    tools = {
        "docker":   "Install from https://docs.docker.com/get-docker/",
        "kubectl":  "Install from https://kubernetes.io/docs/tasks/tools/",
        "minikube": "Install from https://minikube.sigs.k8s.io/docs/start/",
        "dvc":      "pip install dvc",
        "pytest":   "pip install pytest",
    }
    for tool, hint in tools.items():
        check(tool, cmd_available(tool), hint)

    # ── pytest ─────────────────────────────────────────────────────────────────
    print("\n[3] Running tests:")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=str(root),
    )
    check("All tests pass", result.returncode == 0,
          "Fix failing tests before pushing to CI")

    # ── Summary & commands ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("PHASE 10 COMMANDS")
    print("=" * 60)
    print()
    print("-- Docker ------------------------------------------")
    print("  docker build -t toolsense-api .")
    print("  docker compose up --build")
    print()
    print("-- Kubernetes (minikube) ---------------------------")
    print("  minikube start")
    print("  minikube image load toolsense-api:latest")
    print("  kubectl apply -f k8s/")
    print("  minikube service toolsense-api")
    print()
    print("-- DVC ---------------------------------------------")
    print("  dvc init")
    print("  dvc add data/clean_tools.csv data/qdrant data/ground_truth.json")
    print("  dvc repro          # run full pipeline")
    print("  dvc dag            # visualise pipeline")
    print()
    print("-- Tests / CI --------------------------------------")
    print("  pytest tests/ -v")
    print("  pytest tests/ --cov=src --cov-report=term-missing")
    print()
    print("-- GitHub Actions ----------------------------------")
    print("  git push origin main   # triggers ci.yml automatically")
    print()
    print("=" * 60)
    print()

    if all_ok:
        print("All Phase 10 artefacts present. Ready to push!")
    else:
        print("Some artefacts missing — see ERR above.")

    print("\nNext step: Phase 11 — Redis caching + Evidently drift monitoring")


if __name__ == "__main__":
    main()
