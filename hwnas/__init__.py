"""hwnas: a small, readable hardware-aware NAS toolkit for edge devices (teaching code)."""
from .accuracy import SyntheticAccuracy
from .cost import arch_metrics
from .devices import PROFILES, DeviceProfile
from .latency import LUT, LatencyPredictor, build_lut, sim_latency
from .search import (Constraints, Evaluator, evolutionary_search, mnas_reward, nsga2,
                     pareto_front, random_search)
from .space import OPS, SearchSpace, SpaceConfig

__version__ = "0.1.0"
