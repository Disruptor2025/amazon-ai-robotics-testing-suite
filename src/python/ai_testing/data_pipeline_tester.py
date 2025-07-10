"""
Data Pipeline Testing for Amazon AI Models

This module provides comprehensive testing capabilities for data pipelines,
including ETL validation, data quality checks, and pipeline performance testing.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

import boto3
import pandas as pd
import numpy as np
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


@dataclass
class PipelineTestResult:
    """Result of data pipeline test."""
    test_name: str
    status: str  # 'PASS', 'FAIL', 'WARNING'
    score: float
    details: Dict[str, Any]
    timestamp: datetime
    execution_time: float


class DataPipelineTester:
    """
    Comprehensive data pipeline testing tool.
    
    This class provides methods to test data pipelines, validate data quality,
    and ensure proper ETL workflows.
    """
    
    def __init__(self, region_name: str = 'us-east-1'):
        """
        Initialize the DataPipelineTester.
        
        Args:
            region_name: AWS region for services
        """
        self.region_name = region_name
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.glue_client = boto3.client('glue', region_name=region_name)
        self.results: List[PipelineTestResult] = []
        
    async def test_data_quality(
        self,
        data: pd.DataFrame,
        quality_rules: Dict[str, Any]
    ) -> PipelineTestResult:
        """
        Test data quality against defined rules.
        
        Args:
            data: DataFrame to test
            quality_rules: Dictionary of quality rules to apply
            
        Returns:
            PipelineTestResult with quality test results
        """
        start_time = time.time()
        
        try:
            quality_scores = {}
            total_score = 0.0
            num_rules = 0
            
            # Test completeness
            if 'completeness' in quality_rules:
                completeness_score = self._test_completeness(data, quality_rules['completeness'])
                quality_scores['completeness'] = completeness_score
                total_score += completeness_score
                num_rules += 1
            
            # Test accuracy
            if 'accuracy' in quality_rules:
                accuracy_score = self._test_accuracy(data, quality_rules['accuracy'])
                quality_scores['accuracy'] = accuracy_score
                total_score += accuracy_score
                num_rules += 1
            
            # Test consistency
            if 'consistency' in quality_rules:
                consistency_score = self._test_consistency(data, quality_rules['consistency'])
                quality_scores['consistency'] = consistency_score
                total_score += consistency_score
                num_rules += 1
            
            # Test validity
            if 'validity' in quality_rules:
                validity_score = self._test_validity(data, quality_rules['validity'])
                quality_scores['validity'] = validity_score
                total_score += validity_score
                num_rules += 1
            
            # Test uniqueness
            if 'uniqueness' in quality_rules:
                uniqueness_score = self._test_uniqueness(data, quality_rules['uniqueness'])
                quality_scores['uniqueness'] = uniqueness_score
                total_score += uniqueness_score
                num_rules += 1
            
            avg_score = total_score / num_rules if num_rules > 0 else 0.0
            execution_time = time.time() - start_time
            
            result = PipelineTestResult(
                test_name="data_quality_test",
                status='PASS' if avg_score >= 0.9 else 'WARNING' if avg_score >= 0.7 else 'FAIL',
                score=avg_score,
                details={
                    'quality_scores': quality_scores,
                    'total_rules': num_rules,
                    'data_shape': data.shape,
                    'missing_values': data.isnull().sum().to_dict(),
                    'duplicate_rows': data.duplicated().sum()
                },
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.results.append(result)
            logger.info(f"Data quality test completed: {result.status} (score: {avg_score:.3f})")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = PipelineTestResult(
                test_name="data_quality_test",
                status='FAIL',
                score=0.0,
                details={'error': str(e)},
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.results.append(result)
            logger.error(f"Data quality test failed: {e}")
            return result
    
    async def test_etl_pipeline(
        self,
        source_data: pd.DataFrame,
        transformation_function,
        expected_output_schema: Dict[str, str],
        validation_rules: Optional[Dict[str, Any]] = None
    ) -> PipelineTestResult:
        """
        Test ETL pipeline transformation.
        
        Args:
            source_data: Input data for transformation
            transformation_function: Function to transform data
            expected_output_schema: Expected schema of output data
            validation_rules: Additional validation rules
            
        Returns:
            PipelineTestResult with ETL test results
        """
        start_time = time.time()
        
        try:
            # Apply transformation
            transformed_data = transformation_function(source_data)
            
            # Validate output schema
            schema_score = self._validate_schema(transformed_data, expected_output_schema)
            
            # Validate data integrity
            integrity_score = self._validate_data_integrity(source_data, transformed_data)
            
            # Apply additional validation rules
            validation_score = 1.0
            if validation_rules:
                validation_score = self._apply_validation_rules(transformed_data, validation_rules)
            
            # Calculate overall score
            overall_score = (schema_score + integrity_score + validation_score) / 3
            execution_time = time.time() - start_time
            
            result = PipelineTestResult(
                test_name="etl_pipeline_test",
                status='PASS' if overall_score >= 0.9 else 'WARNING' if overall_score >= 0.7 else 'FAIL',
                score=overall_score,
                details={
                    'schema_score': schema_score,
                    'integrity_score': integrity_score,
                    'validation_score': validation_score,
                    'input_shape': source_data.shape,
                    'output_shape': transformed_data.shape,
                    'schema_validation': self._get_schema_details(transformed_data, expected_output_schema)
                },
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.results.append(result)
            logger.info(f"ETL pipeline test completed: {result.status} (score: {overall_score:.3f})")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = PipelineTestResult(
                test_name="etl_pipeline_test",
                status='FAIL',
                score=0.0,
                details={'error': str(e)},
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.results.append(result)
            logger.error(f"ETL pipeline test failed: {e}")
            return result
    
    async def test_data_ingestion(
        self,
        source_path: str,
        target_path: str,
        ingestion_config: Dict[str, Any]
    ) -> PipelineTestResult:
        """
        Test data ingestion from source to target.
        
        Args:
            source_path: Source data path (S3, local, etc.)
            target_path: Target data path
            ingestion_config: Configuration for ingestion process
            
        Returns:
            PipelineTestResult with ingestion test results
        """
        start_time = time.time()
        
        try:
            # Read source data
            source_data = self._read_data(source_path, ingestion_config.get('source_format', 'csv'))
            
            # Perform ingestion
            ingestion_success = await self._perform_ingestion(source_data, target_path, ingestion_config)
            
            # Validate ingested data
            if ingestion_success:
                target_data = self._read_data(target_path, ingestion_config.get('target_format', 'csv'))
                validation_score = self._validate_ingested_data(source_data, target_data)
            else:
                validation_score = 0.0
            
            execution_time = time.time() - start_time
            
            result = PipelineTestResult(
                test_name="data_ingestion_test",
                status='PASS' if validation_score >= 0.95 else 'FAIL',
                score=validation_score,
                details={
                    'source_path': source_path,
                    'target_path': target_path,
                    'ingestion_success': ingestion_success,
                    'source_shape': source_data.shape if source_data is not None else None,
                    'target_shape': target_data.shape if 'target_data' in locals() else None
                },
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.results.append(result)
            logger.info(f"Data ingestion test completed: {result.status} (score: {validation_score:.3f})")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = PipelineTestResult(
                test_name="data_ingestion_test",
                status='FAIL',
                score=0.0,
                details={'error': str(e)},
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.results.append(result)
            logger.error(f"Data ingestion test failed: {e}")
            return result
    
    async def test_pipeline_performance(
        self,
        pipeline_function,
        test_data: pd.DataFrame,
        performance_thresholds: Dict[str, float]
    ) -> PipelineTestResult:
        """
        Test pipeline performance against thresholds.
        
        Args:
            pipeline_function: Function to test
            test_data: Test data
            performance_thresholds: Performance thresholds to validate
            
        Returns:
            PipelineTestResult with performance test results
        """
        start_time = time.time()
        
        try:
            # Measure execution time
            pipeline_start = time.time()
            result_data = pipeline_function(test_data)
            pipeline_time = time.time() - pipeline_start
            
            # Measure memory usage
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            
            # Calculate performance scores
            time_score = 1.0 if pipeline_time <= performance_thresholds.get('max_execution_time', float('inf')) else 0.0
            memory_score = 1.0 if memory_usage <= performance_thresholds.get('max_memory_mb', float('inf')) else 0.0
            
            # Calculate throughput
            throughput = len(test_data) / pipeline_time if pipeline_time > 0 else 0
            throughput_score = 1.0 if throughput >= performance_thresholds.get('min_throughput', 0) else 0.0
            
            overall_score = (time_score + memory_score + throughput_score) / 3
            execution_time = time.time() - start_time
            
            result = PipelineTestResult(
                test_name="pipeline_performance_test",
                status='PASS' if overall_score >= 0.9 else 'WARNING' if overall_score >= 0.7 else 'FAIL',
                score=overall_score,
                details={
                    'execution_time': pipeline_time,
                    'memory_usage_mb': memory_usage,
                    'throughput': throughput,
                    'time_score': time_score,
                    'memory_score': memory_score,
                    'throughput_score': throughput_score,
                    'input_size': len(test_data),
                    'output_size': len(result_data) if result_data is not None else 0
                },
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.results.append(result)
            logger.info(f"Pipeline performance test completed: {result.status} (score: {overall_score:.3f})")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = PipelineTestResult(
                test_name="pipeline_performance_test",
                status='FAIL',
                score=0.0,
                details={'error': str(e)},
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.results.append(result)
            logger.error(f"Pipeline performance test failed: {e}")
            return result
    
    def _test_completeness(self, data: pd.DataFrame, rules: Dict[str, Any]) -> float:
        """Test data completeness."""
        total_score = 0.0
        num_columns = 0
        
        for column, rule in rules.items():
            if column in data.columns:
                missing_ratio = data[column].isnull().sum() / len(data)
                threshold = rule.get('max_missing_ratio', 0.1)
                score = 1.0 if missing_ratio <= threshold else 0.0
                total_score += score
                num_columns += 1
        
        return total_score / num_columns if num_columns > 0 else 0.0
    
    def _test_accuracy(self, data: pd.DataFrame, rules: Dict[str, Any]) -> float:
        """Test data accuracy."""
        total_score = 0.0
        num_columns = 0
        
        for column, rule in rules.items():
            if column in data.columns:
                if 'value_range' in rule:
                    min_val, max_val = rule['value_range']
                    valid_count = ((data[column] >= min_val) & (data[column] <= max_val)).sum()
                    score = valid_count / len(data)
                elif 'allowed_values' in rule:
                    valid_count = data[column].isin(rule['allowed_values']).sum()
                    score = valid_count / len(data)
                else:
                    score = 1.0
                
                total_score += score
                num_columns += 1
        
        return total_score / num_columns if num_columns > 0 else 0.0
    
    def _test_consistency(self, data: pd.DataFrame, rules: Dict[str, Any]) -> float:
        """Test data consistency."""
        total_score = 0.0
        num_rules = 0
        
        for rule in rules:
            if 'column_relationship' in rule:
                col1, col2, relationship = rule['column_relationship']
                if col1 in data.columns and col2 in data.columns:
                    if relationship == 'equal':
                        score = (data[col1] == data[col2]).sum() / len(data)
                    elif relationship == 'greater_than':
                        score = (data[col1] > data[col2]).sum() / len(data)
                    else:
                        score = 1.0
                    
                    total_score += score
                    num_rules += 1
        
        return total_score / num_rules if num_rules > 0 else 1.0
    
    def _test_validity(self, data: pd.DataFrame, rules: Dict[str, Any]) -> float:
        """Test data validity."""
        total_score = 0.0
        num_columns = 0
        
        for column, rule in rules.items():
            if column in data.columns:
                if 'data_type' in rule:
                    expected_type = rule['data_type']
                    actual_type = str(data[column].dtype)
                    score = 1.0 if expected_type in actual_type else 0.0
                elif 'pattern' in rule:
                    import re
                    pattern = re.compile(rule['pattern'])
                    valid_count = data[column].astype(str).str.match(pattern).sum()
                    score = valid_count / len(data)
                else:
                    score = 1.0
                
                total_score += score
                num_columns += 1
        
        return total_score / num_columns if num_columns > 0 else 0.0
    
    def _test_uniqueness(self, data: pd.DataFrame, rules: Dict[str, Any]) -> float:
        """Test data uniqueness."""
        total_score = 0.0
        num_columns = 0
        
        for column, rule in rules.items():
            if column in data.columns:
                duplicate_ratio = data[column].duplicated().sum() / len(data)
                threshold = rule.get('max_duplicate_ratio', 0.1)
                score = 1.0 if duplicate_ratio <= threshold else 0.0
                total_score += score
                num_columns += 1
        
        return total_score / num_columns if num_columns > 0 else 0.0
    
    def _validate_schema(self, data: pd.DataFrame, expected_schema: Dict[str, str]) -> float:
        """Validate data schema."""
        if data is None:
            return 0.0
        
        correct_columns = 0
        total_columns = len(expected_schema)
        
        for column, expected_type in expected_schema.items():
            if column in data.columns:
                actual_type = str(data[column].dtype)
                if expected_type in actual_type:
                    correct_columns += 1
        
        return correct_columns / total_columns if total_columns > 0 else 0.0
    
    def _validate_data_integrity(self, source_data: pd.DataFrame, transformed_data: pd.DataFrame) -> float:
        """Validate data integrity after transformation."""
        if source_data is None or transformed_data is None:
            return 0.0
        
        # Check if no data was lost (basic check)
        if len(transformed_data) == 0:
            return 0.0
        
        # Check if transformation preserved some data
        if len(transformed_data) > 0:
            return 1.0
        
        return 0.0
    
    def _apply_validation_rules(self, data: pd.DataFrame, rules: Dict[str, Any]) -> float:
        """Apply custom validation rules."""
        total_score = 0.0
        num_rules = 0
        
        for rule_name, rule in rules.items():
            if rule.get('type') == 'row_count':
                expected_count = rule.get('expected_count')
                actual_count = len(data)
                if expected_count is not None:
                    score = 1.0 if actual_count == expected_count else 0.0
                    total_score += score
                    num_rules += 1
        
        return total_score / num_rules if num_rules > 0 else 1.0
    
    def _read_data(self, path: str, format_type: str) -> pd.DataFrame:
        """Read data from various sources."""
        try:
            if format_type.lower() == 'csv':
                return pd.read_csv(path)
            elif format_type.lower() == 'json':
                return pd.read_json(path)
            elif format_type.lower() == 'parquet':
                return pd.read_parquet(path)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
        except Exception as e:
            logger.error(f"Error reading data from {path}: {e}")
            return None
    
    async def _perform_ingestion(self, data: pd.DataFrame, target_path: str, config: Dict[str, Any]) -> bool:
        """Perform data ingestion."""
        try:
            format_type = config.get('target_format', 'csv')
            
            if format_type.lower() == 'csv':
                data.to_csv(target_path, index=False)
            elif format_type.lower() == 'json':
                data.to_json(target_path, orient='records')
            elif format_type.lower() == 'parquet':
                data.to_parquet(target_path, index=False)
            else:
                raise ValueError(f"Unsupported target format: {format_type}")
            
            return True
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
            return False
    
    def _validate_ingested_data(self, source_data: pd.DataFrame, target_data: pd.DataFrame) -> float:
        """Validate ingested data against source."""
        if source_data is None or target_data is None:
            return 0.0
        
        # Basic validation - check if data was preserved
        if len(source_data) == len(target_data):
            return 1.0
        
        return 0.5  # Partial match
    
    def _get_schema_details(self, data: pd.DataFrame, expected_schema: Dict[str, str]) -> Dict[str, Any]:
        """Get detailed schema validation information."""
        if data is None:
            return {'error': 'No data available'}
        
        schema_details = {}
        for column, expected_type in expected_schema.items():
            if column in data.columns:
                actual_type = str(data[column].dtype)
                schema_details[column] = {
                    'expected_type': expected_type,
                    'actual_type': actual_type,
                    'match': expected_type in actual_type
                }
            else:
                schema_details[column] = {
                    'expected_type': expected_type,
                    'actual_type': 'missing',
                    'match': False
                }
        
        return schema_details
    
    def get_pipeline_test_summary(self) -> Dict[str, Any]:
        """
        Get summary of all pipeline test results.
        
        Returns:
            Summary statistics of pipeline test results
        """
        if not self.results:
            return {'message': 'No pipeline test results available'}
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.status == 'PASS')
        failed_tests = sum(1 for r in self.results if r.status == 'FAIL')
        warning_tests = sum(1 for r in self.results if r.status == 'WARNING')
        
        avg_score = sum(r.score for r in self.results) / total_tests
        avg_execution_time = sum(r.execution_time for r in self.results) / total_tests
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'warning_tests': warning_tests,
            'pass_rate': passed_tests / total_tests,
            'average_score': avg_score,
            'average_execution_time': avg_execution_time,
            'results': [r.__dict__ for r in self.results]
        } 