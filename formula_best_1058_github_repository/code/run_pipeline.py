"""Run the complete descriptor, model-search and figure workflow."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "matminer_composition_features.py",
    "rf_rfe_ten_model_search.py",
    "package_comparison_models.py",
    "package_final_extratrees.py",
    "generate_final_paper_figures.py",
    "build_descriptor_workbook.py",
    "validate_repository.py",
]


def main() -> None:
    for script in SCRIPTS:
        path = ROOT / "code" / script
        print(f"\n=== Running {script} ===", flush=True)
        subprocess.run([sys.executable, "-u", str(path)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
