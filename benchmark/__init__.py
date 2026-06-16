"""Benchmark suite for ride-pooling dispatch baselines.

Provides a standard benchmark scenario configuration, baseline dispatch
algorithms, a detailed episode recorder, and a runner that ties them together
so different algorithms can be compared on identical conditions.
"""

from benchmark.config import BenchmarkConfig, make_benchmark_env
from benchmark.recorder import EpisodeRecorder
from benchmark.baselines import NearestDistanceDispatch, HungarianDispatch
from benchmark.runner import run_episode

__all__ = [
    "BenchmarkConfig",
    "make_benchmark_env",
    "EpisodeRecorder",
    "NearestDistanceDispatch",
    "HungarianDispatch",
    "run_episode",
]