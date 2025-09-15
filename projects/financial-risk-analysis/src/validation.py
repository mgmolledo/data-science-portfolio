"""
Model Validation Framework
Professional-grade validation with statistical rigor
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, 
    cross_validate, 
    GridSearchCV,
    RandomizedSearchCV,
    validation_curve
)
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ModelValidator:
    """
    Professional-grade model validation framework
    Implements industry-standard validation practices
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.validation_results = {}
        self.best_models = {}
        
    def robust_cross_validation(
        self, 
        models: Dict[str, Any], 
        X: pd.DataFrame, 
        y: pd.Series,
        cv_folds: int = 5,
        scoring: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform robust cross-validation with multiple metrics
        
        Args:
            models: Dictionary of model names and model objects
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            scoring: List of scoring metrics
        
        Returns:
            Dictionary with validation results for each model
        """
        if scoring is None:
            scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        print("🔬 CROSS-VALIDATION ANALYSIS")
        print("=" * 50)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        results = {}
        
        for name, model in models.items():
            print(f"\n🔄 Validating {name}...")
            
            try:
                # Perform cross-validation
                cv_results = cross_validate(
                    model, X, y, 
                    cv=cv, 
                    scoring=scoring,
                    return_train_score=True,
                    n_jobs=-1
                )
                
                # Calculate statistics
                model_results = {}
                for metric in scoring:
                    train_scores = cv_results[f'train_{metric}']
                    test_scores = cv_results[f'test_{metric}']
                    
                    model_results[metric] = {
                        'train_mean': train_scores.mean(),
                        'train_std': train_scores.std(),
                        'test_mean': test_scores.mean(),
                        'test_std': test_scores.std(),
                        'overfitting': train_scores.mean() - test_scores.mean()
                    }
                
                results[name] = model_results
                
                # Print summary
                best_metric = 'roc_auc' if 'roc_auc' in scoring else scoring[0]
                print(f"  ✅ {name}: {best_metric} = {model_results[best_metric]['test_mean']:.3f} ± {model_results[best_metric]['test_std']:.3f}")
                
            except Exception as e:
                print(f"  ❌ Error validating {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        self.validation_results = results
        return results
    
    def calculate_overfitting_score(self, results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate overfitting scores for each model
        
        Args:
            results: Cross-validation results
        
        Returns:
            Dictionary with overfitting scores
        """
        overfitting_scores = {}
        
        for model_name, model_results in results.items():
            if 'error' in model_results:
                continue
                
            overfitting_score = 0
            metric_count = 0
            
            for metric, scores in model_results.items():
                if isinstance(scores, dict) and 'overfitting' in scores:
                    overfitting_score += abs(scores['overfitting'])
                    metric_count += 1
            
            if metric_count > 0:
                overfitting_scores[model_name] = overfitting_score / metric_count
            else:
                overfitting_scores[model_name] = 0
        
        return overfitting_scores
    
    def calculate_stability_score(self, results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate stability scores (lower std = higher stability)
        
        Args:
            results: Cross-validation results
        
        Returns:
            Dictionary with stability scores
        """
        stability_scores = {}
        
        for model_name, model_results in results.items():
            if 'error' in model_results:
                continue
                
            stability_score = 0
            metric_count = 0
            
            for metric, scores in model_results.items():
                if isinstance(scores, dict) and 'test_std' in scores:
                    # Convert std to stability (lower std = higher stability)
                    stability_score += 1 - scores['test_std']
                    metric_count += 1
            
            if metric_count > 0:
                stability_scores[model_name] = stability_score / metric_count
            else:
                stability_scores[model_name] = 0
        
        return stability_scores
    
    def perform_statistical_tests(self, results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Perform statistical significance tests between models
        
        Args:
            results: Cross-validation results
        
        Returns:
            Dictionary with statistical test results
        """
        from scipy import stats
        
        statistical_tests = {}
        
        # Get model names
        model_names = [name for name in results.keys() if 'error' not in results[name]]
        
        if len(model_names) < 2:
            return statistical_tests
        
        # Compare each pair of models
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                comparison_key = f"{model1}_vs_{model2}"
                
                # Get ROC AUC scores for comparison
                if 'roc_auc' in results[model1] and 'roc_auc' in results[model2]:
                    scores1 = results[model1]['roc_auc']['test_scores']
                    scores2 = results[model2]['roc_auc']['test_scores']
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_rel(scores1, scores2)
                    
                    statistical_tests[comparison_key] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'model1_mean': scores1.mean(),
                        'model2_mean': scores2.mean()
                    }
        
        return statistical_tests
    
    def model_comparison_report(self, results: Dict[str, Dict[str, float]]) -> str:
        """
        Generate a comprehensive model comparison report
        
        Args:
            results: Cross-validation results
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("🔬 MODEL VALIDATION REPORT")
        report.append("=" * 50)
        
        # Calculate additional metrics
        overfitting_scores = self.calculate_overfitting_score(results)
        stability_scores = self.calculate_stability_score(results)
        
        # Create comparison table
        comparison_data = []
        for model_name, model_results in results.items():
            if 'error' in model_results:
                continue
                
            row = {
                'Model': model_name,
                'AUC': f"{model_results.get('roc_auc', {}).get('test_mean', 0):.3f}",
                'Accuracy': f"{model_results.get('accuracy', {}).get('test_mean', 0):.3f}",
                'Precision': f"{model_results.get('precision', {}).get('test_mean', 0):.3f}",
                'Recall': f"{model_results.get('recall', {}).get('test_mean', 0):.3f}",
                'F1': f"{model_results.get('f1', {}).get('test_mean', 0):.3f}",
                'Stability': f"{stability_scores.get(model_name, 0):.3f}",
                'Overfitting': f"{overfitting_scores.get(model_name, 0):.3f}"
            }
            comparison_data.append(row)
        
        # Sort by AUC score
        comparison_data.sort(key=lambda x: float(x['AUC']), reverse=True)
        
        # Add table to report
        report.append("\n📊 MODEL COMPARISON TABLE")
        report.append("-" * 50)
        
        if comparison_data:
            # Header
            header = "| " + " | ".join(comparison_data[0].keys()) + " |"
            separator = "|" + "|".join(["-" * (len(col) + 2) for col in comparison_data[0].keys()]) + "|"
            
            report.append(header)
            report.append(separator)
            
            # Data rows
            for row in comparison_data:
                data_row = "| " + " | ".join(row.values()) + " |"
                report.append(data_row)
        
        # Best model recommendations
        report.append("\n🏆 BEST MODEL RECOMMENDATIONS")
        report.append("-" * 50)
        
        if comparison_data:
            best_auc = max(comparison_data, key=lambda x: float(x['AUC']))
            best_stability = max(comparison_data, key=lambda x: float(x['Stability']))
            best_overfitting = min(comparison_data, key=lambda x: float(x['Overfitting']))
            
            report.append(f"🥇 Best AUC Score: {best_auc['Model']} ({best_auc['AUC']})")
            report.append(f"⚖️ Most Stable: {best_stability['Model']} ({best_stability['Stability']})")
            report.append(f"🎯 Least Overfitting: {best_overfitting['Model']} ({best_overfitting['Overfitting']})")
        
        # Statistical significance
        statistical_tests = self.perform_statistical_tests(results)
        if statistical_tests:
            report.append("\n📈 STATISTICAL SIGNIFICANCE TESTS")
            report.append("-" * 50)
            
            for comparison, test_result in statistical_tests.items():
                significance = "✅ Significant" if test_result['significant'] else "❌ Not Significant"
                report.append(f"{comparison}: {significance} (p={test_result['p_value']:.4f})")
        
        return "\n".join(report)
    
    def hyperparameter_optimization(
        self, 
        model, 
        param_grid: Dict[str, List], 
        X: pd.DataFrame, 
        y: pd.Series,
        cv_folds: int = 5,
        optimization_type: str = 'grid'
    ) -> Dict[str, Any]:
        """
        Advanced hyperparameter optimization with statistical validation
        
        Args:
            model: Model to optimize
            param_grid: Parameter grid for optimization
            X: Feature matrix
            y: Target vector
            cv_folds: Number of CV folds
            optimization_type: 'grid' or 'random'
        
        Returns:
            Optimization results
        """
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        if optimization_type == 'grid':
            optimizer = GridSearchCV(
                model, param_grid, 
                cv=cv, 
                scoring='roc_auc',
                n_jobs=-1,
                return_train_score=True
            )
        else:
            optimizer = RandomizedSearchCV(
                model, param_grid, 
                cv=cv, 
                scoring='roc_auc',
                n_jobs=-1,
                return_train_score=True,
                n_iter=50
            )
        
        print(f"🔧 Optimizing hyperparameters using {optimization_type} search...")
        optimizer.fit(X, y)
        
        return {
            'best_params': optimizer.best_params_,
            'best_score': optimizer.best_score_,
            'cv_results': optimizer.cv_results_,
            'best_estimator': optimizer.best_estimator_
        }

if __name__ == "__main__":
    print("Model Validation Framework")
