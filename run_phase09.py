"""
Phase 09 Runner — FastAPI + Streamlit
=======================================
Starts the FastAPI backend and prints instructions for the Streamlit UI.

Usage:
    python run_phase09.py          # start API server (blocking)

Then in a second terminal:
    streamlit run app/streamlit_app.py

API docs: http://localhost:8000/docs
UI:       http://localhost:8501
"""

import sys
import subprocess
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING)

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))


def main():
    print("=" * 60)
    print("PHASE 09 — ToolSense FastAPI + Streamlit")
    print("=" * 60)
    print()
    print("Starting FastAPI backend on http://localhost:8000 ...")
    print()
    print("  API docs:   http://localhost:8000/docs")
    print("  Health:     http://localhost:8000/health")
    print()
    print("Open a SECOND terminal and run:")
    print()
    print("  cd toolsense")
    print("  streamlit run app/streamlit_app.py")
    print()
    print("  UI: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop the API server.")
    print("=" * 60)
    print()

    # Start uvicorn
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "src.api.main:app",
        "--reload",
        "--port", "8000",
        "--host", "0.0.0.0",
        "--log-level", "warning",
    ], cwd=str(root))


if __name__ == "__main__":
    main()
