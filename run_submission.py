from __future__ import annotations

import argparse
import gc
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import sys
import warnings


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
warnings.filterwarnings(
    "ignore", message="X does not have valid feature names, but LGBM.* was fitted with feature names"
)


# This file is the canonical project entry point. Keeping the project root at
# the front also makes ``src.research_data`` unambiguous regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ProjectConfig
from src.data_pipeline import coverage_table, load_analysis_data
from src.profitability import run_profitability_suite
from src.reporting import save_run
from src.stability import run_stability_suite


def package_versions(names: list[str]) -> dict[str, str]:
    result = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the unified S&P 500 stability and profitability hypothesis report."
    )
    parser.add_argument(
        "--skip-profitability",
        action="store_true",
        help="Diagnostic option: run stability only. The default runs both suites.",
    )
    parser.add_argument('--compare-candidates', action='store_true',
                        help='Run only the five common-sample final candidates and save their selection report.')
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    config = ProjectConfig.discover()
    print("[1/4] 공통 데이터 파이프라인 로드", flush=True)
    data = load_analysis_data(config)
    tables = {"data_coverage": coverage_table(data)}

    if args.compare_candidates:
        from src.candidate_comparison import run_comparison
        from src.stability import StabilityPolicy
        policy = StabilityPolicy()
        tables.update(run_comparison(data.frame, data.calendar, data.sectors, data.macro, policy))
        result = save_run(config, policy, tables, {
            'run_scope': 'candidate_comparison_only', 'input_sha256': data.input_hashes,
            'evaluation_role': 'exploratory_reused_oos',
        })
        print('Candidate comparison saved:', result, flush=True)
        return result

    print("[2/4] 통일 Stable 정책 및 안정성 가설 검증", flush=True)
    stability_tables, policy = run_stability_suite(
        data.frame, data.calendar, data.sectors, config
    )
    tables.update(stability_tables)
    del stability_tables
    gc.collect()

    if not args.skip_profitability:
        print("[3/4] 수익성 가설 및 포트폴리오 검증", flush=True)
        tables.update(
            run_profitability_suite(
                data.frame, data.calendar, data.sectors, data.macro, policy, config
            )
        )
        gc.collect()
    else:
        print("[3/4] 수익성 검증 건너뜀 (--skip-profitability)", flush=True)

    print("[4/4] 제출용 표·manifest·Markdown 보고서 저장", flush=True)
    metadata = {
        "project_root": str(config.project_root),
        "input_sha256": data.input_hashes,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(
            ["numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "torch", "pyarrow"]
        ),
        "source_notebooks": ["sp500_stability_v14.ipynb", "test_validation_report.ipynb"],
        "limitations": [
            "reused evaluation period",
            "static sector mapping",
            "unresolved settlements remain missing",
            "market price index is not asserted to be a total-return benchmark",
        ],
    }
    run_dir = save_run(config, policy, tables, metadata)
    print("완료:", run_dir)
    print("보고서:", run_dir / "UNIFIED_VALIDATION_REPORT.md")
    return run_dir


if __name__ == "__main__":
    main()
