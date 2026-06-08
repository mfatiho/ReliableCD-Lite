from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import pandas as pd


@dataclass
class CRSNormalizer:
    """Z-score normalizer fit on LEVIR-CD validation components, applied unchanged to WHU/DSIFN.

    Z-score is preferred over min-max (sensitive to outlier components) and rank normalization
    (discards inter-signal variance structure needed for equal-weight CRS additivity).
    """

    columns: list[str] = field(default_factory=list)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, columns: list[str]) -> None:
        self.columns = list(columns)
        self.means = {col: float(df[col].mean()) for col in columns}
        self.stds = {}
        for col in columns:
            std = float(df[col].std(ddof=0))
            self.stds[col] = std if std > 0 else 1.0

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.columns:
            out[f"z_{col}"] = (out[col] - self.means[col]) / self.stds[col]
        return out

    def save(self, path: str | Path) -> None:
        payload = {"columns": self.columns, "means": self.means, "stds": self.stds}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CRSNormalizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(columns=payload["columns"], means=payload["means"], stds=payload["stds"])
