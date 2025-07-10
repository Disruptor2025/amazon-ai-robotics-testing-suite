"""
Amazon AI Testing Framework

This module provides comprehensive testing capabilities for Amazon AI services,
including SageMaker models, inference validation, performance benchmarking,
and model drift detection.
"""

__version__ = "1.0.0"
__author__ = "Amazon AI Testing Team"

from .model_validator import ModelValidator
from .performance_benchmark import PerformanceBenchmark
from .drift_detector import DriftDetector
from .data_pipeline_tester import DataPipelineTester

__all__ = [
    "ModelValidator",
    "PerformanceBenchmark", 
    "DriftDetector",
    "DataPipelineTester"
] 