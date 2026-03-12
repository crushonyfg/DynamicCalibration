from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional

import numpy as np


@dataclasses.dataclass
class PlantSimConfig:
    model_path: str = r"C:\Users\yxu59\files\winter2026\park\simulation\DBCsystem_v3.spp"
    prog_id: str = "Tecnomatix.PlantSimulation.RemoteControl.24.4"
    visible: bool = False
    model_root: str = ".Models.Field"
    event_controller: str = ".Models.Field.EventController"
    net_revenue_path: str = ".Models.Field.NetRevenue"
    timeout_s: float = 300.0
    poll_interval_s: float = 0.25


class PlantSimulationRunner:
    def __init__(self, config: Optional[PlantSimConfig] = None):
        self.config = config or PlantSimConfig()
        self.plant = None

    def start(self) -> "PlantSimulationRunner":
        if self.plant is not None:
            return self

        import win32com.client

        self.plant = win32com.client.DispatchEx(self.config.prog_id)
        self.plant.SetVisible(bool(self.config.visible))
        self.plant.LoadModel(self.config.model_path)
        self.plant.ResetSimulation(self.config.event_controller)
        return self

    def close(self) -> None:
        if self.plant is None:
            return
        try:
            self.plant.Quit()
        finally:
            self.plant = None

    def __enter__(self) -> "PlantSimulationRunner":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure_started(self) -> None:
        if self.plant is None:
            self.start()

    def _set_inputs(self, x_decision: np.ndarray, customer_lbd_min: float) -> None:
        q, r, w, m1, m2 = [int(v) for v in np.asarray(x_decision).reshape(-1)]
        params = {
            ".Models.Field.Q": q,
            ".Models.Field.R": r,
            ".Models.Field.W": w,
            ".Models.Field.M1": m1,
            ".Models.Field.M2": m2,
            ".Models.Field.CustomerLbd": float(customer_lbd_min) * 60.0,
        }
        for path, value in params.items():
            self.plant.SetValue(path, value)

    def evaluate_one(self, x_decision: np.ndarray, customer_lbd_min: float) -> float:
        self._ensure_started()
        self.plant.ResetSimulation(self.config.event_controller)
        self._set_inputs(x_decision, customer_lbd_min)
        self.plant.StartSimulation(self.config.model_root, True)

        start_time = time.time()
        while self.plant.IsSimulationRunning():
            if time.time() - start_time > self.config.timeout_s:
                raise TimeoutError("PlantSimulation evaluation timed out.")
            time.sleep(self.config.poll_interval_s)

        return float(self.plant.GetValue(self.config.net_revenue_path))

    def evaluate_one_mean(
        self,
        x_decision: np.ndarray,
        customer_lbd_min: float,
        n_replications: int = 1,
    ) -> Dict[str, float]:
        ys = [self.evaluate_one(x_decision, customer_lbd_min) for _ in range(int(n_replications))]
        ys_arr = np.asarray(ys, dtype=np.float64)
        return {
            "y_mean": float(ys_arr.mean()),
            "y_std": float(ys_arr.std(ddof=0)),
            "n_replications": int(n_replications),
        }

    def evaluate_batch(
        self,
        X_decision_batch: np.ndarray,
        customer_lbd_min: float,
        n_replications: int = 1,
    ) -> Dict[str, np.ndarray]:
        X = np.asarray(X_decision_batch, dtype=np.int64)
        y_mean_list: List[float] = []
        y_std_list: List[float] = []

        for x in X:
            stats = self.evaluate_one_mean(
                x_decision=x,
                customer_lbd_min=customer_lbd_min,
                n_replications=n_replications,
            )
            y_mean_list.append(stats["y_mean"])
            y_std_list.append(stats["y_std"])

        return {
            "X_decision": X.copy(),
            "y_mean": np.asarray(y_mean_list, dtype=np.float64),
            "y_std": np.asarray(y_std_list, dtype=np.float64),
        }


class EmulatorProxyRunner:
    def __init__(self, scorer):
        self.scorer = scorer

    def __enter__(self) -> "EmulatorProxyRunner":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def evaluate_one_mean(
        self,
        x_decision: np.ndarray,
        customer_lbd_min: float,
        n_replications: int = 1,
    ) -> Dict[str, float]:
        mu_raw, _ = self.scorer(np.asarray(x_decision, dtype=np.int64)[None, :], float(customer_lbd_min))
        return {
            "y_mean": float(mu_raw[0]),
            "y_std": 0.0,
            "n_replications": int(n_replications),
        }

    def evaluate_batch(
        self,
        X_decision_batch: np.ndarray,
        customer_lbd_min: float,
        n_replications: int = 1,
    ) -> Dict[str, np.ndarray]:
        X = np.asarray(X_decision_batch, dtype=np.int64)
        mu_raw, _ = self.scorer(X, float(customer_lbd_min))
        return {
            "X_decision": X.copy(),
            "y_mean": np.asarray(mu_raw, dtype=np.float64),
            "y_std": np.zeros(len(X), dtype=np.float64),
        }
