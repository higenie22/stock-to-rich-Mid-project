from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class ProjectConfig:
    """Single source of truth for paths and experiment constants."""

    project_root: Path
    output_root: Path
    random_state: int = 42
    stable_lookback: int = 120
    stable_k: int = 3
    future_risk_horizon: int = 120
    future_risky_quantile: float = 0.30
    mdd_threshold: float = -0.15

    @classmethod
    def discover(cls) -> "ProjectConfig":
        default_root = Path(__file__).resolve().parents[1]
        root = Path(os.environ.get("VALIDATION_PROJECT_ROOT", default_root)).resolve()
        output = Path(
            os.environ.get("VALIDATION_OUTPUT_DIR", root / "outputs")
        ).resolve()
        return cls(project_root=root, output_root=output)

    @property
    def data_root(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_price_path(self) -> Path:
        return self.data_root / "processed" / "final_df.parquet"

    @property
    def benchmark_path(self) -> Path:
        return self.data_root / "raw" / "sp500_beta_df.parquet"

    @property
    def membership_path(self) -> Path:
        return self.data_root / "raw" / "cache" / "sp500_membership_history.csv"

    @property
    def sector_path(self) -> Path:
        return self.data_root / "raw" / "sp500_universe.csv"

    @property
    def macro_path(self) -> Path:
        return self.data_root / "raw" / "validation_macro_market_raw.csv"

    def required_paths(self) -> list[Path]:
        return [
            self.raw_price_path,
            self.benchmark_path,
            self.membership_path,
            self.sector_path,
            self.macro_path,
            self.project_root / "src" / "research_data.py",
        ]

    def validate(self) -> None:
        missing = [p for p in self.required_paths() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "제출용 분석에 필요한 파일이 없습니다:\n" + "\n".join(map(str, missing))
            )
