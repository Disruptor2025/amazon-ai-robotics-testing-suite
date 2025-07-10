"""
Model Validator for Amazon SageMaker Models

This module provides comprehensive validation capabilities for SageMaker models,
including inference testing, accuracy validation, and model health checks.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError, WaiterError
from sagemaker import Session
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of model validation test."""
    test_name: str
    status: str  # 'PASS', 'FAIL', 'WARNING'
    score: float
    details: Dict[str, Any]
    timestamp: datetime
    execution_time: float


class ModelValidator:
    """
    Comprehensive validator for Amazon SageMaker models.
    
    This class provides methods to validate model inference, accuracy,
    performance, and health across different scenarios.
    """
    
    def __init__(self, region_name: str = 'us-east-1'):
        """
        Initialize the ModelValidator.
        
        Args:
            region_name: AWS region for SageMaker services
        """
        self.region_name = region_name
        self.sagemaker_session = Session(boto_session=boto3.Session(region_name=region_name))
        self.sagemaker_client = boto3.client('sagemaker', region_name=region_name)
        self.results: List[ValidationResult] = []
        
    async def validate_model_inference(
        self,
        endpoint_name: str,
        test_data: Union[List, np.ndarray, pd.DataFrame],
        expected_outputs: Optional[List] = None,
        tolerance: float = 0.01
    ) -> ValidationResult:
        """
        Validate model inference capabilities.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            test_data: Input data for testing
            expected_outputs: Expected outputs for validation
            tolerance: Tolerance for numerical comparisons
            
        Returns:
            ValidationResult with test results
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
            
            # Perform inference
            predictions = predictor.predict(test_data)
            
            # Validate predictions
            validation_score = 1.0
            validation_details = {
                'predictions': predictions,
                'input_data_shape': np.array(test_data).shape,
                'output_shape': np.array(predictions).shape
            }
            
            # Check if predictions match expected outputs
            if expected_outputs is not None:
                if len(predictions) != len(expected_outputs):
                    validation_score = 0.0
                    validation_details['error'] = 'Output length mismatch'
                else:
                    # Compare predictions with expected outputs
                    for i, (pred, expected) in enumerate(zip(predictions, expected_outputs)):
                        if isinstance(pred, (int, float)) and isinstance(expected, (int, float)):
                            if abs(pred - expected) > tolerance:
                                validation_score = 0.0
                                validation_details[f'mismatch_at_index_{i}'] = {
                                    'predicted': pred,
                                    'expected': expected,
                                    'difference': abs(pred - expected)
                                }
                                break
                        elif pred != expected:
                            validation_score = 0.0
                            validation_details[f'mismatch_at_index_{i}'] = {
                                'predicted': pred,
                                'expected': expected
                            }
                            break
            
            execution_time = time.time() - start_time
            
            result = ValidationResult(
                test_name=f"inference_validation_{endpoint_name}",
                status='PASS' if validation_score >= 0.95 else 'FAIL',
                score=validation_score,
                details=validation_details,
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.results.append(result)
            logger.info(f"Inference validation completed: {result.status}")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = ValidationResult(
                test_name=f"inference_validation_{endpoint_name}",
                status='FAIL',
                score=0.0,
                details={'error': str(e)},
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.results.append(result)
            logger.error(f"Inference validation failed: {e}")
            return result
    
    async def validate_model_accuracy(
        self,
        endpoint_name: str,
        test_data: pd.DataFrame,
        ground_truth: pd.Series,
        metric: str = 'accuracy'
    ) -> ValidationResult:
        """
        Validate model accuracy using various metrics.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            test_data: Input test data
            ground_truth: Ground truth labels
            metric: Accuracy metric to use ('accuracy', 'precision', 'recall', 'f1')
            
        Returns:
            ValidationResult with accuracy metrics
        """
        start_time = time.time()
        
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Get predictions
            predictor = Predictor(
                endpoint_name=endpoint_name,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer(),
                sagemaker_session=self.sagemaker_session
            )
            
            predictions = predictor.predict(test_data.values.tolist())
            predictions = np.array(predictions).flatten()
            
            # Calculate metrics
            if metric == 'accuracy':
                score = accuracy_score(ground_truth, predictions)
            elif metric == 'precision':
                score = precision_score(ground_truth, predictions, average='weighted')
            elif metric == 'recall':
                score = recall_score(ground_truth, predictions, average='weighted')
            elif metric == 'f1':
                score = f1_score(ground_truth, predictions, average='weighted')
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            # Calculate all metrics for comprehensive reporting
            all_metrics = {
                'accuracy': accuracy_score(ground_truth, predictions),
                'precision': precision_score(ground_truth, predictions, average='weighted'),
                'recall': recall_score(ground_truth, predictions, average='weighted'),
                'f1': f1_score(ground_truth, predictions, average='weighted')
            }
            
            execution_time = time.time() - start_time
            
            result = ValidationResult(
                test_name=f"accuracy_validation_{endpoint_name}_{metric}",
                status='PASS' if score >= 0.8 else 'WARNING' if score >= 0.6 else 'FAIL',
                score=score,
                details={
                    'all_metrics': all_metrics,
                    'predictions_count': len(predictions),
                    'ground_truth_count': len(ground_truth)
                },
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.results.append(result)
            logger.info(f"Accuracy validation completed: {result.status} (score: {score:.3f})")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = ValidationResult(
                test_name=f"accuracy_validation_{endpoint_name}_{metric}",
                status='FAIL',
                score=0.0,
                details={'error': str(e)},
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.results.append(result)
            logger.error(f"Accuracy validation failed: {e}")
            return result
    
    async def validate_model_health(
        self,
        endpoint_name: str
    ) -> ValidationResult:
        """
        Validate model endpoint health and availability.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            
        Returns:
            ValidationResult with health check results
        """
        start_time = time.time()
        
        try:
            # Check endpoint status
            response = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            endpoint_status = response['EndpointStatus']
            
            # Check endpoint configuration
            config_response = self.sagemaker_client.describe_endpoint_config(
                EndpointConfigName=response['EndpointConfigName']
            )
            
            # Check model status
            model_name = config_response['ProductionVariants'][0]['ModelName']
            model_response = self.sagemaker_client.describe_model(ModelName=model_name)
            
            health_score = 1.0
            health_details = {
                'endpoint_status': endpoint_status,
                'model_status': model_response['ModelStatus'],
                'instance_type': config_response['ProductionVariants'][0]['InstanceType'],
                'instance_count': config_response['ProductionVariants'][0]['InitialInstanceCount']
            }
            
            # Check for issues
            if endpoint_status != 'InService':
                health_score = 0.0
                health_details['error'] = f'Endpoint not in service: {endpoint_status}'
            
            if model_response['ModelStatus'] != 'InService':
                health_score = 0.0
                health_details['error'] = f'Model not in service: {model_response["ModelStatus"]}'
            
            execution_time = time.time() - start_time
            
            result = ValidationResult(
                test_name=f"health_check_{endpoint_name}",
                status='PASS' if health_score >= 0.95 else 'FAIL',
                score=health_score,
                details=health_details,
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.results.append(result)
            logger.info(f"Health check completed: {result.status}")
            return result
            
        except ClientError as e:
            execution_time = time.time() - start_time
            result = ValidationResult(
                test_name=f"health_check_{endpoint_name}",
                status='FAIL',
                score=0.0,
                details={'error': str(e)},
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.results.append(result)
            logger.error(f"Health check failed: {e}")
            return result
    
    async def run_comprehensive_validation(
        self,
        endpoint_name: str,
        test_data: pd.DataFrame,
        ground_truth: Optional[pd.Series] = None
    ) -> Dict[str, ValidationResult]:
        """
        Run comprehensive validation suite for a model.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            test_data: Test data for validation
            ground_truth: Ground truth labels (optional)
            
        Returns:
            Dictionary of validation results
        """
        logger.info(f"Starting comprehensive validation for endpoint: {endpoint_name}")
        
        validation_results = {}
        
        # Health check
        health_result = await self.validate_model_health(endpoint_name)
        validation_results['health'] = health_result
        
        # Inference validation
        inference_result = await self.validate_model_inference(
            endpoint_name, 
            test_data.values.tolist()
        )
        validation_results['inference'] = inference_result
        
        # Accuracy validation (if ground truth provided)
        if ground_truth is not None:
            accuracy_result = await self.validate_model_accuracy(
                endpoint_name,
                test_data,
                ground_truth
            )
            validation_results['accuracy'] = accuracy_result
        
        # Calculate overall score
        overall_score = sum(result.score for result in validation_results.values()) / len(validation_results)
        
        logger.info(f"Comprehensive validation completed. Overall score: {overall_score:.3f}")
        
        return validation_results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all validation results.
        
        Returns:
            Summary statistics of validation results
        """
        if not self.results:
            return {'message': 'No validation results available'}
        
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