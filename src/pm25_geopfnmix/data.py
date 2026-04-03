from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .settings import DATA_PATH, ID_COL, TARGET_COL


@dataclass(slots=True)
class DatasetBundle:
    frame: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series


def load_dataset() -> DatasetBundle:
    frame = pd.read_csv(DATA_PATH)
    features = frame.drop(columns=[TARGET_COL, ID_COL])
    target = frame[TARGET_COL].copy()
    return DatasetBundle(frame=frame, features=features, target=target)
