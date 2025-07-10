"""
Performance Benchmarking for Amazon AI Models

This module provides comprehensive performance testing capabilities for SageMaker models,
including latency testing, throughput analysis, and load testing.
"""

import asyncio
import concurrent.futures
import logging
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import boto3
import numpy as np
import pandas as pd
from sagemaker import Session
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of performance benchmark test."""
    test_name: str
    metric: str
    value: float
    unit: str
    details: Dict[str, Any]
    timestamp: datetime
    execution_time: float


class PerformanceBenchmark:
    """
    Performance benchmarking tool for Amazon SageMaker models.
    
    This class provides methods to test model performance under various
    conditions including latency, throughput, and load testing.
    """
    
    def __init__(self, region_name: str = 'us-east-1'):
        """
        Initialize the PerformanceBenchmark.
        
        Args:
            region_name: AWS region for SageMaker services
        """
        self.region_name = region_name
        self.sagemaker_session = Session(boto_session=boto3.Session(region_name=region_name))
        self.results: List[BenchmarkResult] = []
        
    async def benchmark_latency(
        self,
        endpoint_name: str,
        test_data: List,
        num_requests: int = 100,
        concurrent_requests: int = 10
    ) -> BenchmarkResult:
        """
        Benchmark model latency under various load conditions.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            test_data: Input data for testing
            num_requests: Total number of requests to make
            concurrent_requests: Number of concurrent requests
            
        Returns:
            BenchmarkResult with latency metrics
        """
        start_time = time.time()
        
        try:
            # Create predictor
            predictor = Predictor(
                endpoint_name=endpoint_name,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer(),
                sagemaker_session=self.sagemaker_session
            )
            
            # Prepare test data
            test_samples = [test_data for _ in range(num_requests)]
            
            # Measure latency
            latencies = []
            
            async def make_request(sample):
                request_start = time.time()
                try:
                    predictor.predict(sample)
                    request_end = time.time()
                    return request_end - request_start
                except Exception as e:
                    logger.error(f"Request failed: {e}")
                    return None
            
            # Execute requests with concurrency
            semaphore = asyncio.Semaphore(concurrent_requests)
            
            async def bounded_request(sample):
                async with semaphore:
                    return await make_request(sample)
            
            tasks = [bounded_request(sample) for sample in test_samples]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out failed requests
            latencies = [r for r in results if r is not None and not isinstance(r, Exception)]
            
            if not latencies:
                raise Exception("All requests failed")
            
            # Calculate statistics
            avg_latency = statistics.mean(latencies)
            p50_latency = statistics.median(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            execution_time = time.time() - start_time
            
            result = BenchmarkResult(
                test_name=f"latency_benchmark_{endpoint_name}",
                metric="latency",
                value=avg_latency,
                unit="seconds",
                details={
                    'avg_latency': avg_latency,
                    'p50_latency': p50_latency,
                    'p95_latency': p95_latency,
                    'p99_latency': p99_latency,
                    'min_latency': min_latency,
                    'max_latency': max_latency,
                    'num_requests': num_requests,
                    'concurrent_requests': concurrent_requests,
                    'successful_requests': len(latencies),
                    'failed_requests': num_requests - len(latencies),
                    'all_latencies': latencies
                },
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.results.append(result)
            logger.info(f"Latency benchmark completed: {avg_latency:.3f}s average")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = BenchmarkResult(
                test_name=f"latency_benchmark_{endpoint_name}",
                metric="latency",
                value=0.0,
                unit="seconds",
                details={'error': str(e)},
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.results.append(result)
            logger.error(f"Latency benchmark failed: {e}")
            return result
    
    async def benchmark_throughput(
        self,
        endpoint_name: str,
        test_data: List,
        duration_seconds: int = 60,
        target_rps: int = 100
    ) -> BenchmarkResult:
        """
        Benchmark model throughput (requests per second).
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            test_data: Input data for testing
            duration_seconds: Duration of the test in seconds
            target_rps: Target requests per second
            
        Returns:
            BenchmarkResult with throughput metrics
        """
        start_time = time.time()
        
        try:
            # Create predictor
            predictor = Predictor(
                endpoint_name=endpoint_name,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer(),
                sagemaker_session=self.sagemaker_session
            )
            
            # Calculate request interval
            request_interval = 1.0 / target_rps
            
            # Track requests
            successful_requests = 0
            failed_requests = 0
            request_times = []
            
            test_end_time = time.time() + duration_seconds
            
            async def make_request():
                nonlocal successful_requests, failed_requests
                request_start = time.time()
                try:
                    predictor.predict(test_data)
                    request_end = time.time()
                    successful_requests += 1
                    request_times.append(request_end - request_start)
                except Exception as e:
                    failed_requests += 1
                    logger.error(f"Request failed: {e}")
            
            # Execute requests at target rate
            while time.time() < test_end_time:
                await make_request()
                await asyncio.sleep(request_interval)
            
            # Calculate throughput
            actual_duration = time.time() - start_time
            actual_rps = successful_requests / actual_duration
            
            # Calculate latency statistics
            if request_times:
                avg_latency = statistics.mean(request_times)
                p95_latency = np.percentile(request_times, 95)
            else:
                avg_latency = 0.0
                p95_latency = 0.0
            
            execution_time = time.time() - start_time
            
            result = BenchmarkResult(
                test_name=f"throughput_benchmark_{endpoint_name}",
                metric="throughput",
                value=actual_rps,
                unit="requests_per_second",
                details={
                    'target_rps': target_rps,
                    'actual_rps': actual_rps,
                    'successful_requests': successful_requests,
                    'failed_requests': failed_requests,
                    'total_requests': successful_requests + failed_requests,
                    'test_duration': actual_duration,
                    'avg_latency': avg_latency,
                    'p95_latency': p95_latency,
                    'success_rate': successful_requests / (successful_requests + failed_requests) if (successful_requests + failed_requests) > 0 else 0
                },
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.results.append(result)
            logger.info(f"Throughput benchmark completed: {actual_rps:.2f} RPS")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = BenchmarkResult(
                test_name=f"throughput_benchmark_{endpoint_name}",
                metric="throughput",
                value=0.0,
                unit="requests_per_second",
                details={'error': str(e)},
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.results.append(result)
            logger.error(f"Throughput benchmark failed: {e}")
            return result
    
    async def benchmark_memory_usage(
        self,
        endpoint_name: str,
        test_data: List,
        num_requests: int = 1000
    ) -> BenchmarkResult:
        """
        Benchmark memory usage during model inference.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            test_data: Input data for testing
            num_requests: Number of requests to make
            
        Returns:
            BenchmarkResult with memory usage metrics
        """
        start_time = time.time()
        
        try:
            import psutil
            import os
            
            # Get initial memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create predictor
            predictor = Predictor(
                endpoint_name=endpoint_name,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer(),
                sagemaker_session=self.sagemaker_session
            )
            
            # Make requests and monitor memory
            memory_usage = []
            
            for i in range(num_requests):
                if i % 100 == 0:  # Sample memory every 100 requests
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_usage.append(current_memory)
                
                try:
                    predictor.predict(test_data)
                except Exception as e:
                    logger.error(f"Request {i} failed: {e}")
            
            # Get final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_usage.append(final_memory)
            
            # Calculate memory statistics
            max_memory = max(memory_usage)
            avg_memory = statistics.mean(memory_usage)
            memory_increase = final_memory - initial_memory
            
            execution_time = time.time() - start_time
            
            result = BenchmarkResult(
                test_name=f"memory_benchmark_{endpoint_name}",
                metric="memory_usage",
                value=avg_memory,
                unit="MB",
                details={
                    'initial_memory': initial_memory,
                    'final_memory': final_memory,
                    'max_memory': max_memory,
                    'avg_memory': avg_memory,
                    'memory_increase': memory_increase,
                    'num_requests': num_requests,
                    'memory_samples': memory_usage
                },
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.results.append(result)
            logger.info(f"Memory benchmark completed: {avg_memory:.2f} MB average")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = BenchmarkResult(
                test_name=f"memory_benchmark_{endpoint_name}",
                metric="memory_usage",
                value=0.0,
                unit="MB",
                details={'error': str(e)},
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.results.append(result)
            logger.error(f"Memory benchmark failed: {e}")
            return result
    
    async def run_load_test(
        self,
        endpoint_name: str,
        test_data: List,
        load_scenarios: List[Dict[str, Any]]
    ) -> Dict[str, BenchmarkResult]:
        """
        Run comprehensive load testing with multiple scenarios.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            test_data: Input data for testing
            load_scenarios: List of load scenarios to test
            
        Returns:
            Dictionary of benchmark results for each scenario
        """
        logger.info(f"Starting load test for endpoint: {endpoint_name}")
        
        load_results = {}
        
        for i, scenario in enumerate(load_scenarios):
            scenario_name = scenario.get('name', f'scenario_{i}')
            logger.info(f"Running scenario: {scenario_name}")
            
            if scenario.get('type') == 'latency':
                result = await self.benchmark_latency(
                    endpoint_name,
                    test_data,
                    num_requests=scenario.get('num_requests', 100),
                    concurrent_requests=scenario.get('concurrent_requests', 10)
                )
            elif scenario.get('type') == 'throughput':
                result = await self.benchmark_throughput(
                    endpoint_name,
                    test_data,
                    duration_seconds=scenario.get('duration_seconds', 60),
                    target_rps=scenario.get('target_rps', 100)
                )
            else:
                logger.warning(f"Unknown scenario type: {scenario.get('type')}")
                continue
            
            load_results[scenario_name] = result
        
        logger.info(f"Load test completed with {len(load_results)} scenarios")
        return load_results
    
    def get_benchmark_summary(self) -> Dict[str, Any]:
        """
        Get summary of all benchmark results.
        
        Returns:
            Summary statistics of benchmark results
        """
        if not self.results:
            return {'message': 'No benchmark results available'}
        
        # Group results by metric
        latency_results = [r for r in self.results if r.metric == 'latency']
        throughput_results = [r for r in self.results if r.metric == 'throughput']
        memory_results = [r for r in self.results if r.metric == 'memory_usage']
        
        summary = {
            'total_benchmarks': len(self.results),
            'latency_benchmarks': len(latency_results),
            'throughput_benchmarks': len(throughput_results),
            'memory_benchmarks': len(memory_results)
        }
        
        if latency_results:
            avg_latency = statistics.mean([r.value for r in latency_results])
            summary['average_latency'] = avg_latency
        
        if throughput_results:
            avg_throughput = statistics.mean([r.value for r in throughput_results])
            summary['average_throughput'] = avg_throughput
        
        if memory_results:
            avg_memory = statistics.mean([r.value for r in memory_results])
            summary['average_memory_usage'] = avg_memory
        
        summary['results'] = [r.__dict__ for r in self.results]
        
        return summary 