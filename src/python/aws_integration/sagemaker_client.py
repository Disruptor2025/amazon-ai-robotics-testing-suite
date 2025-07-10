"""
AWS SageMaker Client

This module provides comprehensive integration with Amazon SageMaker for model training,
deployment, and testing.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

import boto3
import pandas as pd
from botocore.exceptions import ClientError, WaiterError
from sagemaker import Session
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import numpy as np

logger = logging.getLogger(__name__)


class SageMakerClient:
    """
    Comprehensive SageMaker client for AI model management and testing.
    
    This class provides methods to interact with SageMaker services including
    model training, deployment, and inference testing.
    """
    
    def __init__(self, region_name: str = 'us-east-1'):
        """
        Initialize the SageMakerClient.
        
        Args:
            region_name: AWS region for SageMaker services
        """
        self.region_name = region_name
        self.session = Session(boto_session=boto3.Session(region_name=region_name))
        self.sagemaker_client = boto3.client('sagemaker', region_name=region_name)
        self.s3_client = boto3.client('s3', region_name=region_name)
        
    async def create_training_job(
        self,
        job_name: str,
        training_data_path: str,
        validation_data_path: str,
        output_path: str,
        hyperparameters: Dict[str, str],
        instance_type: str = 'ml.m5.large',
        instance_count: int = 1,
        max_run_time: int = 3600
    ) -> Dict[str, Any]:
        """
        Create and start a SageMaker training job.
        
        Args:
            job_name: Name of the training job
            training_data_path: S3 path to training data
            validation_data_path: S3 path to validation data
            output_path: S3 path for model artifacts
            hyperparameters: Training hyperparameters
            instance_type: EC2 instance type for training
            instance_count: Number of training instances
            max_run_time: Maximum training time in seconds
            
        Returns:
            Dictionary with training job details
        """
        try:
            # Create estimator
            estimator = Estimator(
                image_uri='763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.1-gpu-py38',
                role=self.session.get_caller_identity_arn(),
                instance_count=instance_count,
                instance_type=instance_type,
                max_run=max_run_time,
                output_path=output_path,
                sagemaker_session=self.session
            )
            
            # Set hyperparameters
            estimator.set_hyperparameters(**hyperparameters)
            
            # Start training
            estimator.fit({
                'train': training_data_path,
                'validation': validation_data_path
            })
            
            # Get training job details
            training_job = self.sagemaker_client.describe_training_job(TrainingJobName=job_name)
            
            logger.info(f"Training job {job_name} completed successfully")
            return {
                'job_name': job_name,
                'status': training_job['TrainingJobStatus'],
                'model_artifacts': training_job.get('ModelArtifacts', {}).get('S3ModelArtifacts'),
                'training_time': training_job.get('TrainingTimeInSeconds', 0),
                'final_metrics': training_job.get('FinalMetricDataList', [])
            }
            
        except Exception as e:
            logger.error(f"Training job {job_name} failed: {e}")
            raise
    
    async def deploy_model(
        self,
        model_name: str,
        model_artifacts_path: str,
        endpoint_name: str,
        instance_type: str = 'ml.m5.large',
        instance_count: int = 1
    ) -> Dict[str, Any]:
        """
        Deploy a trained model to a SageMaker endpoint.
        
        Args:
            model_name: Name for the model
            model_artifacts_path: S3 path to model artifacts
            endpoint_name: Name for the endpoint
            instance_type: EC2 instance type for inference
            instance_count: Number of inference instances
            
        Returns:
            Dictionary with deployment details
        """
        try:
            # Create model
            model = Model(
                image_uri='763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.12.1-gpu-py38',
                model_data=model_artifacts_path,
                role=self.session.get_caller_identity_arn(),
                sagemaker_session=self.session
            )
            
            # Deploy model
            predictor = model.deploy(
                initial_instance_count=instance_count,
                instance_type=instance_type,
                endpoint_name=endpoint_name
            )
            
            # Wait for endpoint to be in service
            self.sagemaker_client.get_waiter('endpoint_in_service').wait(
                EndpointName=endpoint_name
            )
            
            # Get endpoint details
            endpoint = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            
            logger.info(f"Model {model_name} deployed to endpoint {endpoint_name}")
            return {
                'model_name': model_name,
                'endpoint_name': endpoint_name,
                'status': endpoint['EndpointStatus'],
                'endpoint_arn': endpoint['EndpointArn'],
                'creation_time': endpoint['CreationTime']
            }
            
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            raise
    
    async def test_inference(
        self,
        endpoint_name: str,
        test_data: Union[List, np.ndarray, pd.DataFrame],
        expected_outputs: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Test model inference on an endpoint.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            test_data: Input data for testing
            expected_outputs: Expected outputs for validation
            
        Returns:
            Dictionary with inference test results
        """
        try:
            # Create predictor
            predictor = Predictor(
                endpoint_name=endpoint_name,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer(),
                sagemaker_session=self.session
            )
            
            # Perform inference
            start_time = time.time()
            predictions = predictor.predict(test_data)
            inference_time = time.time() - start_time
            
            # Validate predictions if expected outputs provided
            validation_results = {}
            if expected_outputs is not None:
                validation_results = self._validate_predictions(predictions, expected_outputs)
            
            return {
                'endpoint_name': endpoint_name,
                'predictions': predictions,
                'inference_time': inference_time,
                'input_size': len(test_data),
                'output_size': len(predictions),
                'validation_results': validation_results
            }
            
        except Exception as e:
            logger.error(f"Inference test failed: {e}")
            raise
    
    async def monitor_endpoint(
        self,
        endpoint_name: str
    ) -> Dict[str, Any]:
        """
        Monitor endpoint health and performance.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            
        Returns:
            Dictionary with monitoring results
        """
        try:
            # Get endpoint details
            endpoint = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            
            # Get endpoint configuration
            config = self.sagemaker_client.describe_endpoint_config(
                EndpointConfigName=endpoint['EndpointConfigName']
            )
            
            # Get CloudWatch metrics
            cloudwatch = boto3.client('cloudwatch', region_name=self.region_name)
            
            # Get CPU utilization
            cpu_response = cloudwatch.get_metric_statistics(
                Namespace='AWS/SageMaker',
                MetricName='CPUUtilization',
                Dimensions=[{'Name': 'EndpointName', 'Value': endpoint_name}],
                StartTime=datetime.now() - timedelta(hours=1),
                EndTime=datetime.now(),
                Period=300,
                Statistics=['Average', 'Maximum']
            )
            
            # Get memory utilization
            memory_response = cloudwatch.get_metric_statistics(
                Namespace='AWS/SageMaker',
                MetricName='MemoryUtilization',
                Dimensions=[{'Name': 'EndpointName', 'Value': endpoint_name}],
                StartTime=datetime.now() - timedelta(hours=1),
                EndTime=datetime.now(),
                Period=300,
                Statistics=['Average', 'Maximum']
            )
            
            return {
                'endpoint_name': endpoint_name,
                'status': endpoint['EndpointStatus'],
                'instance_type': config['ProductionVariants'][0]['InstanceType'],
                'instance_count': config['ProductionVariants'][0]['InitialInstanceCount'],
                'cpu_utilization': cpu_response.get('Datapoints', []),
                'memory_utilization': memory_response.get('Datapoints', []),
                'last_modified': endpoint['LastModifiedTime']
            }
            
        except Exception as e:
            logger.error(f"Endpoint monitoring failed: {e}")
            raise
    
    async def update_endpoint(
        self,
        endpoint_name: str,
        new_model_artifacts_path: str,
        instance_type: Optional[str] = None,
        instance_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Update an existing endpoint with a new model.
        
        Args:
            endpoint_name: Name of the endpoint to update
            new_model_artifacts_path: S3 path to new model artifacts
            instance_type: New instance type (optional)
            instance_count: New instance count (optional)
            
        Returns:
            Dictionary with update results
        """
        try:
            # Get current endpoint configuration
            endpoint = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            current_config = self.sagemaker_client.describe_endpoint_config(
                EndpointConfigName=endpoint['EndpointConfigName']
            )
            
            # Create new endpoint configuration
            new_config_name = f"{endpoint_name}-config-{int(time.time())}"
            
            new_config = self.sagemaker_client.create_endpoint_config(
                EndpointConfigName=new_config_name,
                ProductionVariants=[{
                    'VariantName': 'AllTraffic',
                    'ModelName': current_config['ProductionVariants'][0]['ModelName'],
                    'InitialInstanceCount': instance_count or current_config['ProductionVariants'][0]['InitialInstanceCount'],
                    'InstanceType': instance_type or current_config['ProductionVariants'][0]['InstanceType']
                }]
            )
            
            # Update endpoint
            self.sagemaker_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=new_config_name
            )
            
            # Wait for update to complete
            self.sagemaker_client.get_waiter('endpoint_in_service').wait(
                EndpointName=endpoint_name
            )
            
            logger.info(f"Endpoint {endpoint_name} updated successfully")
            return {
                'endpoint_name': endpoint_name,
                'new_config_name': new_config_name,
                'status': 'Updated'
            }
            
        except Exception as e:
            logger.error(f"Endpoint update failed: {e}")
            raise
    
    async def delete_endpoint(
        self,
        endpoint_name: str
    ) -> Dict[str, Any]:
        """
        Delete a SageMaker endpoint.
        
        Args:
            endpoint_name: Name of the endpoint to delete
            
        Returns:
            Dictionary with deletion results
        """
        try:
            # Delete endpoint
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            
            # Wait for deletion to complete
            self.sagemaker_client.get_waiter('endpoint_deleted').wait(
                EndpointName=endpoint_name
            )
            
            logger.info(f"Endpoint {endpoint_name} deleted successfully")
            return {
                'endpoint_name': endpoint_name,
                'status': 'Deleted'
            }
            
        except Exception as e:
            logger.error(f"Endpoint deletion failed: {e}")
            raise
    
    def _validate_predictions(
        self,
        predictions: List,
        expected_outputs: List
    ) -> Dict[str, Any]:
        """Validate predictions against expected outputs."""
        if len(predictions) != len(expected_outputs):
            return {
                'valid': False,
                'error': 'Length mismatch between predictions and expected outputs'
            }
        
        correct_predictions = 0
        for pred, expected in zip(predictions, expected_outputs):
            if pred == expected:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(predictions)
        
        return {
            'valid': accuracy >= 0.8,
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': len(predictions)
        }
    
    async def list_endpoints(self) -> List[Dict[str, Any]]:
        """
        List all SageMaker endpoints.
        
        Returns:
            List of endpoint details
        """
        try:
            response = self.sagemaker_client.list_endpoints()
            endpoints = []
            
            for endpoint in response['Endpoints']:
                endpoints.append({
                    'endpoint_name': endpoint['EndpointName'],
                    'status': endpoint['EndpointStatus'],
                    'creation_time': endpoint['CreationTime'],
                    'last_modified': endpoint['LastModifiedTime']
                })
            
            return endpoints
            
        except Exception as e:
            logger.error(f"Failed to list endpoints: {e}")
            raise
    
    async def get_training_jobs(self) -> List[Dict[str, Any]]:
        """
        List all training jobs.
        
        Returns:
            List of training job details
        """
        try:
            response = self.sagemaker_client.list_training_jobs()
            jobs = []
            
            for job in response['TrainingJobSummaries']:
                jobs.append({
                    'job_name': job['TrainingJobName'],
                    'status': job['TrainingJobStatus'],
                    'creation_time': job['CreationTime'],
                    'training_end_time': job.get('TrainingEndTime')
                })
            
            return jobs
            
        except Exception as e:
            logger.error(f"Failed to list training jobs: {e}")
            raise 