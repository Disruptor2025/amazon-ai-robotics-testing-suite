"""
AWS Integration Module

This module provides comprehensive integration with AWS services for AI and robotics testing,
including SageMaker, RoboMaker, EC2, Lambda, S3, and CloudWatch.
"""

__version__ = "1.0.0"
__author__ = "Amazon AI Testing Team"

from .sagemaker_client import SageMakerClient
from .robomaker_client import RoboMakerClient
from .ec2_client import EC2Client
from .lambda_client import LambdaClient
from .s3_client import S3Client
from .cloudwatch_client import CloudWatchClient

__all__ = [
    "SageMakerClient",
    "RoboMakerClient", 
    "EC2Client",
    "LambdaClient",
    "S3Client",
    "CloudWatchClient"
] 