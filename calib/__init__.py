# =============================================================
# file: calib/__init__.py
# =============================================================
"""
Online Bayesian Calibration with Particle Filtering and BOCPD
"""

__version__ = "0.1.0"

# Optional: expose main components at package level
from .configs import CalibrationConfig, ModelConfig, PFConfig, BOCPDConfig
from .online_calibrator import OnlineBayesCalibrator
from .emulator import DeterministicSimulator, GPEmulator
from .data import SyntheticDataStream, SyntheticGeneratorConfig

__all__ = [
    "CalibrationConfig",
    "ModelConfig", 
    "PFConfig",
    "BOCPDConfig",
    "OnlineBayesCalibrator",
    "DeterministicSimulator",
    "GPEmulator",
    "SyntheticDataStream",
    "SyntheticGeneratorConfig",
]