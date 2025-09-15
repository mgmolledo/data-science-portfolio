"""
Machine Learning Models
Professional-grade algorithms with optimization
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available. Install with: pip install lightgbm")

class ModelSuite:
    """
    Professional-grade machine learning model suite
    Includes state-of-the-art algorithms with optimized configurations
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model suite with optimized configurations
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.optimized_params = {}
        
        print("🤖 INITIALIZING MODEL SUITE")
        self._initialize_models()
        print(f"✅ Initialized {len(self.models)} models")
    
    def _initialize_models(self):
        """Initialize all models with optimized parameters"""
        
        # Core Models
        self.models['Logistic Regression'] = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            C=1.0,
            solver='liblinear'
        )
        
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=self.random_state
        )
        
        self.models['SVM'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=self.random_state
        )
        
        self.models['Neural Network'] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=self.random_state
        )
        
        # Ensemble Models
        self.models['AdaBoost'] = AdaBoostClassifier(
            n_estimators=50,
            learning_rate=1.0,
            random_state=self.random_state
        )
        
        self.models['Extra Trees'] = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Advanced Models (if available)
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss'
            )
        
        if LIGHTGBM_AVAILABLE:
            self.models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1
            )
        
        # Additional Models
        self.models['K-Nearest Neighbors'] = KNeighborsClassifier(
            n_neighbors=5,
            weights='uniform',
            algorithm='auto'
        )
        
        self.models['Decision Tree'] = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state
        )
        
        self.models['Naive Bayes'] = GaussianNB()
        
        self.models['Linear Discriminant'] = LinearDiscriminantAnalysis()
        
        self.models['Ridge Classifier'] = RidgeClassifier(
            alpha=1.0,
            random_state=self.random_state
        )
    
    def get_model(self, model_name: str):
        """Get a specific model by name"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        return self.models[model_name]
    
    def get_all_models(self):
        """Get all available models"""
        return self.models.copy()
    
    def get_model_names(self):
        """Get list of all model names"""
        return list(self.models.keys())
    
    def create_optimized_pipeline(self, model_name: str, scaler_type: str = 'standard'):
        """
        Create an optimized pipeline with scaling and feature selection
        
        Args:
            model_name: Name of the model to use
            scaler_type: Type of scaler ('standard', 'robust')
        
        Returns:
            Pipeline with preprocessing and model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Choose scaler
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'robust'")
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', scaler),
            ('feature_selection', SelectKBest(f_classif, k=10)),
            ('model', self.models[model_name])
        ])
        
        return pipeline
    
    def create_ensemble_models(self):
        """Create advanced ensemble models"""
        
        # Voting Classifier
        voting_models = [
            ('rf', self.models['Random Forest']),
            ('gb', self.models['Gradient Boosting']),
            ('svm', self.models['SVM'])
        ]
        
        if XGBOOST_AVAILABLE:
            voting_models.append(('xgb', self.models['XGBoost']))
        
        self.models['Voting Ensemble'] = VotingClassifier(
            estimators=voting_models,
            voting='soft'
        )
        
        # Stacking Classifier
        stacking_models = [
            ('rf', self.models['Random Forest']),
            ('gb', self.models['Gradient Boosting']),
            ('svm', self.models['SVM'])
        ]
        
        self.models['Stacking Ensemble'] = StackingClassifier(
            estimators=stacking_models,
            final_estimator=LogisticRegression(random_state=self.random_state),
            cv=5
        )
        
        print("✅ Created ensemble models: Voting Ensemble, Stacking Ensemble")
    
    def get_model_info(self):
        """Get information about all models"""
        info = {
            'total_models': len(self.models),
            'model_names': list(self.models.keys()),
            'available_libraries': {
                'XGBoost': XGBOOST_AVAILABLE,
                'LightGBM': LIGHTGBM_AVAILABLE
            }
        }
        return info
    
    def print_model_summary(self):
        """Print a summary of all models"""
        print("\n🤖 MODEL SUITE SUMMARY")
        print("=" * 50)
        
        # Core Models
        print("\n📊 Core Models:")
        core_models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'SVM', 'Neural Network']
        for model in core_models:
            if model in self.models:
                print(f"  ✅ {model}")
        
        # Ensemble Models
        print("\n🎯 Ensemble Models:")
        ensemble_models = ['AdaBoost', 'Extra Trees', 'Voting Ensemble', 'Stacking Ensemble']
        for model in ensemble_models:
            if model in self.models:
                print(f"  ✅ {model}")
        
        # Advanced Models
        print("\n🚀 Advanced Models:")
        advanced_models = ['XGBoost', 'LightGBM']
        for model in advanced_models:
            if model in self.models:
                print(f"  ✅ {model}")
        
        # Additional Models
        print("\n🔧 Additional Models:")
        additional_models = ['K-Nearest Neighbors', 'Decision Tree', 'Naive Bayes', 'Linear Discriminant', 'Ridge Classifier']
        for model in additional_models:
            if model in self.models:
                print(f"  ✅ {model}")
        
        print(f"\n📈 Total Models: {len(self.models)}")
        print(f"🔧 Available Libraries: {sum([XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE])}/2")

if __name__ == "__main__":
    # Initialize the model suite
    model_suite = ModelSuite()
    
    # Print summary
    model_suite.print_model_summary()
    
    # Create ensemble models
    model_suite.create_ensemble_models()
    
    # Print updated summary
    model_suite.print_model_summary()
    
    print(f"\nRun: python dashboard.py")
