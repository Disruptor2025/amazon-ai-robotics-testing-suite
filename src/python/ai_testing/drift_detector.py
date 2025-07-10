"""
Model Drift Detection for Amazon AI Models

This module provides comprehensive drift detection capabilities for SageMaker models,
including data drift detection, concept drift detection, and performance degradation monitoring.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result of drift detection test."""
    test_name: str
    drift_type: str  # 'data_drift', 'concept_drift', 'performance_drift'
    detected: bool
    confidence: float
    details: Dict[str, Any]
    timestamp: datetime
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'


class DriftDetector:
    """
    Comprehensive drift detection tool for Amazon SageMaker models.
    
    This class provides methods to detect various types of drift including
    data drift, concept drift, and performance degradation.
    """
    
    def __init__(self, region_name: str = 'us-east-1'):
        """
        Initialize the DriftDetector.
        
        Args:
            region_name: AWS region for SageMaker services
        """
        self.region_name = region_name
        self.results: List[DriftResult] = []
        self.baseline_data: Optional[pd.DataFrame] = None
        self.baseline_metrics: Optional[Dict[str, float]] = None
        
    def set_baseline(
        self,
        baseline_data: pd.DataFrame,
        baseline_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Set baseline data and metrics for drift detection.
        
        Args:
            baseline_data: Baseline dataset for comparison
            baseline_metrics: Baseline performance metrics
        """
        self.baseline_data = baseline_data.copy()
        self.baseline_metrics = baseline_metrics or {}
        logger.info(f"Baseline set with {len(baseline_data)} samples")
    
    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        threshold: float = 0.05,
        method: str = 'ks_test'
    ) -> DriftResult:
        """
        Detect data drift between baseline and current data.
        
        Args:
            current_data: Current dataset to compare against baseline
            threshold: Significance threshold for drift detection
            method: Drift detection method ('ks_test', 'chi_square', 'psi')
            
        Returns:
            DriftResult with drift detection results
        """
        if self.baseline_data is None:
            raise ValueError("Baseline data not set. Call set_baseline() first.")
        
        start_time = datetime.now()
        
        try:
            # Ensure same columns
            common_columns = list(set(self.baseline_data.columns) & set(current_data.columns))
            if not common_columns:
                raise ValueError("No common columns between baseline and current data")
            
            baseline_subset = self.baseline_data[common_columns]
            current_subset = current_data[common_columns]
            
            drift_scores = {}
            drift_detected = False
            total_confidence = 0.0
            num_features = 0
            
            for column in common_columns:
                if baseline_subset[column].dtype in ['object', 'category']:
                    # Categorical data - use chi-square test
                    drift_score = self._chi_square_test(
                        baseline_subset[column], 
                        current_subset[column]
                    )
                else:
                    # Numerical data - use specified method
                    if method == 'ks_test':
                        drift_score = self._ks_test(
                            baseline_subset[column], 
                            current_subset[column]
                        )
                    elif method == 'psi':
                        drift_score = self._psi_test(
                            baseline_subset[column], 
                            current_subset[column]
                        )
                    else:
                        drift_score = self._ks_test(
                            baseline_subset[column], 
                            current_subset[column]
                        )
                
                drift_scores[column] = drift_score
                total_confidence += drift_score
                num_features += 1
                
                if drift_score > threshold:
                    drift_detected = True
            
            avg_confidence = total_confidence / num_features if num_features > 0 else 0.0
            
            # Determine severity
            if avg_confidence > 0.8:
                severity = 'CRITICAL'
            elif avg_confidence > 0.6:
                severity = 'HIGH'
            elif avg_confidence > 0.4:
                severity = 'MEDIUM'
            else:
                severity = 'LOW'
            
            result = DriftResult(
                test_name="data_drift_detection",
                drift_type="data_drift",
                detected=drift_detected,
                confidence=avg_confidence,
                details={
                    'method': method,
                    'threshold': threshold,
                    'drift_scores': drift_scores,
                    'features_analyzed': num_features,
                    'baseline_samples': len(baseline_subset),
                    'current_samples': len(current_subset),
                    'common_features': common_columns
                },
                timestamp=start_time,
                severity=severity
            )
            
            self.results.append(result)
            logger.info(f"Data drift detection completed: {result.detected} (confidence: {avg_confidence:.3f})")
            return result
            
        except Exception as e:
            result = DriftResult(
                test_name="data_drift_detection",
                drift_type="data_drift",
                detected=False,
                confidence=0.0,
                details={'error': str(e)},
                timestamp=start_time,
                severity='LOW'
            )
            
            self.results.append(result)
            logger.error(f"Data drift detection failed: {e}")
            return result
    
    def detect_concept_drift(
        self,
        current_predictions: np.ndarray,
        current_actuals: np.ndarray,
        baseline_predictions: Optional[np.ndarray] = None,
        baseline_actuals: Optional[np.ndarray] = None,
        threshold: float = 0.1
    ) -> DriftResult:
        """
        Detect concept drift by comparing prediction distributions.
        
        Args:
            current_predictions: Current model predictions
            current_actuals: Current actual values
            baseline_predictions: Baseline model predictions
            baseline_actuals: Baseline actual values
            threshold: Threshold for concept drift detection
            
        Returns:
            DriftResult with concept drift detection results
        """
        start_time = datetime.now()
        
        try:
            # Calculate current performance metrics
            current_accuracy = np.mean(current_predictions == current_actuals)
            current_error_rate = 1 - current_accuracy
            
            # Compare with baseline if available
            if baseline_predictions is not None and baseline_actuals is not None:
                baseline_accuracy = np.mean(baseline_predictions == baseline_actuals)
                baseline_error_rate = 1 - baseline_accuracy
                
                # Calculate performance degradation
                performance_degradation = baseline_error_rate - current_error_rate
                
                # Use KS test to compare prediction distributions
                drift_score = self._ks_test(baseline_predictions, current_predictions)
                
                drift_detected = drift_score > threshold or performance_degradation > threshold
                
                details = {
                    'baseline_accuracy': baseline_accuracy,
                    'current_accuracy': current_accuracy,
                    'performance_degradation': performance_degradation,
                    'drift_score': drift_score,
                    'baseline_samples': len(baseline_predictions),
                    'current_samples': len(current_predictions)
                }
            else:
                # Use statistical tests on current data to detect anomalies
                drift_score = self._detect_prediction_anomalies(current_predictions)
                drift_detected = drift_score > threshold
                
                details = {
                    'current_accuracy': current_accuracy,
                    'drift_score': drift_score,
                    'current_samples': len(current_predictions)
                }
            
            # Determine severity
            if drift_score > 0.8:
                severity = 'CRITICAL'
            elif drift_score > 0.6:
                severity = 'HIGH'
            elif drift_score > 0.4:
                severity = 'MEDIUM'
            else:
                severity = 'LOW'
            
            result = DriftResult(
                test_name="concept_drift_detection",
                drift_type="concept_drift",
                detected=drift_detected,
                confidence=drift_score,
                details=details,
                timestamp=start_time,
                severity=severity
            )
            
            self.results.append(result)
            logger.info(f"Concept drift detection completed: {result.detected} (confidence: {drift_score:.3f})")
            return result
            
        except Exception as e:
            result = DriftResult(
                test_name="concept_drift_detection",
                drift_type="concept_drift",
                detected=False,
                confidence=0.0,
                details={'error': str(e)},
                timestamp=start_time,
                severity='LOW'
            )
            
            self.results.append(result)
            logger.error(f"Concept drift detection failed: {e}")
            return result
    
    def detect_performance_drift(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Optional[Dict[str, float]] = None,
        threshold: float = 0.1
    ) -> DriftResult:
        """
        Detect performance drift by comparing metrics over time.
        
        Args:
            current_metrics: Current performance metrics
            baseline_metrics: Baseline performance metrics
            threshold: Threshold for performance drift detection
            
        Returns:
            DriftResult with performance drift detection results
        """
        start_time = datetime.now()
        
        try:
            if baseline_metrics is None:
                baseline_metrics = self.baseline_metrics
            
            if baseline_metrics is None:
                raise ValueError("No baseline metrics available")
            
            # Calculate performance degradation for each metric
            degradation_scores = {}
            total_degradation = 0.0
            num_metrics = 0
            
            for metric_name in current_metrics:
                if metric_name in baseline_metrics:
                    current_value = current_metrics[metric_name]
                    baseline_value = baseline_metrics[metric_name]
                    
                    # Calculate relative degradation
                    if baseline_value != 0:
                        degradation = abs(current_value - baseline_value) / abs(baseline_value)
                    else:
                        degradation = abs(current_value - baseline_value)
                    
                    degradation_scores[metric_name] = degradation
                    total_degradation += degradation
                    num_metrics += 1
            
            avg_degradation = total_degradation / num_metrics if num_metrics > 0 else 0.0
            drift_detected = avg_degradation > threshold
            
            # Determine severity
            if avg_degradation > 0.5:
                severity = 'CRITICAL'
            elif avg_degradation > 0.3:
                severity = 'HIGH'
            elif avg_degradation > 0.1:
                severity = 'MEDIUM'
            else:
                severity = 'LOW'
            
            result = DriftResult(
                test_name="performance_drift_detection",
                drift_type="performance_drift",
                detected=drift_detected,
                confidence=avg_degradation,
                details={
                    'baseline_metrics': baseline_metrics,
                    'current_metrics': current_metrics,
                    'degradation_scores': degradation_scores,
                    'average_degradation': avg_degradation,
                    'metrics_analyzed': num_metrics
                },
                timestamp=start_time,
                severity=severity
            )
            
            self.results.append(result)
            logger.info(f"Performance drift detection completed: {result.detected} (degradation: {avg_degradation:.3f})")
            return result
            
        except Exception as e:
            result = DriftResult(
                test_name="performance_drift_detection",
                drift_type="performance_drift",
                detected=False,
                confidence=0.0,
                details={'error': str(e)},
                timestamp=start_time,
                severity='LOW'
            )
            
            self.results.append(result)
            logger.error(f"Performance drift detection failed: {e}")
            return result
    
    def _ks_test(self, baseline: pd.Series, current: pd.Series) -> float:
        """Perform Kolmogorov-Smirnov test for numerical data."""
        try:
            statistic, p_value = stats.ks_2samp(baseline.dropna(), current.dropna())
            return 1 - p_value  # Convert to drift score
        except Exception:
            return 0.0
    
    def _chi_square_test(self, baseline: pd.Series, current: pd.Series) -> float:
        """Perform chi-square test for categorical data."""
        try:
            # Create contingency table
            baseline_counts = baseline.value_counts()
            current_counts = current.value_counts()
            
            # Align indices
            all_categories = set(baseline_counts.index) | set(current_counts.index)
            baseline_aligned = baseline_counts.reindex(all_categories, fill_value=0)
            current_aligned = current_counts.reindex(all_categories, fill_value=0)
            
            # Perform chi-square test
            statistic, p_value = stats.chi2_contingency([baseline_aligned, current_aligned])[:2]
            return 1 - p_value  # Convert to drift score
        except Exception:
            return 0.0
    
    def _psi_test(self, baseline: pd.Series, current: pd.Series) -> float:
        """Calculate Population Stability Index (PSI)."""
        try:
            # Create histograms
            baseline_hist, _ = np.histogram(baseline.dropna(), bins=10, density=True)
            current_hist, _ = np.histogram(current.dropna(), bins=10, density=True)
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            baseline_hist = baseline_hist + epsilon
            current_hist = current_hist + epsilon
            
            # Calculate PSI
            psi = np.sum((current_hist - baseline_hist) * np.log(current_hist / baseline_hist))
            return min(psi / 10, 1.0)  # Normalize to [0, 1]
        except Exception:
            return 0.0
    
    def _detect_prediction_anomalies(self, predictions: np.ndarray) -> float:
        """Detect anomalies in prediction distribution."""
        try:
            # Use isolation forest to detect anomalies
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions_2d = predictions.reshape(-1, 1)
            anomaly_scores = iso_forest.fit_predict(predictions_2d)
            
            # Calculate anomaly ratio
            anomaly_ratio = np.mean(anomaly_scores == -1)
            return anomaly_ratio
        except Exception:
            return 0.0
    
    def run_comprehensive_drift_detection(
        self,
        current_data: pd.DataFrame,
        current_predictions: Optional[np.ndarray] = None,
        current_actuals: Optional[np.ndarray] = None,
        current_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, DriftResult]:
        """
        Run comprehensive drift detection suite.
        
        Args:
            current_data: Current dataset
            current_predictions: Current model predictions
            current_actuals: Current actual values
            current_metrics: Current performance metrics
            
        Returns:
            Dictionary of drift detection results
        """
        logger.info("Starting comprehensive drift detection")
        
        drift_results = {}
        
        # Data drift detection
        if self.baseline_data is not None:
            data_drift_result = self.detect_data_drift(current_data)
            drift_results['data_drift'] = data_drift_result
        
        # Concept drift detection
        if current_predictions is not None and current_actuals is not None:
            concept_drift_result = self.detect_concept_drift(
                current_predictions, 
                current_actuals
            )
            drift_results['concept_drift'] = concept_drift_result
        
        # Performance drift detection
        if current_metrics is not None:
            performance_drift_result = self.detect_performance_drift(current_metrics)
            drift_results['performance_drift'] = performance_drift_result
        
        logger.info(f"Comprehensive drift detection completed with {len(drift_results)} tests")
        return drift_results
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """
        Get summary of all drift detection results.
        
        Returns:
            Summary statistics of drift detection results
        """
        if not self.results:
            return {'message': 'No drift detection results available'}
        
        total_tests = len(self.results)
        drift_detected = sum(1 for r in self.results if r.detected)
        avg_confidence = sum(r.confidence for r in self.results) / total_tests
        
        # Count by severity
        severity_counts = {}
        for result in self.results:
            severity = result.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_tests': total_tests,
            'drift_detected': drift_detected,
            'drift_rate': drift_detected / total_tests,
            'average_confidence': avg_confidence,
            'severity_distribution': severity_counts,
            'results': [r.__dict__ for r in self.results]
        } 