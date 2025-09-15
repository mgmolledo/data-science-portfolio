"""
Complete Data Analysis Pipeline for Financial Risk Assessment
Demonstrates the full data science lifecycle: ETL, EDA, Feature Engineering, Modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif

# Import modules
from models import ModelSuite
from validation import ModelValidator
import warnings
warnings.filterwarnings('ignore')

class FinancialRiskPipeline:
    """
    Complete pipeline for financial risk analysis
    Demonstrates professional data science practices
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_data = None
        self.cleaned_data = None
        self.features = None
        self.target = None
        self.models = {}
        self.results = {}
        
    def extract_data(self):
        """Step 1: Data Extraction"""
        print("🔄 STEP 1: DATA EXTRACTION")
        print("=" * 50)
        
        try:
            self.raw_data = pd.read_csv(self.data_path)
            print(f"✅ Successfully loaded dataset: {self.raw_data.shape}")
            print(f"📊 Columns: {len(self.raw_data.columns)}")
            print(f"🏢 Companies: {len(self.raw_data)}")
            
            # Rename target column
            if 'Bankrupt?' in self.raw_data.columns:
                self.raw_data = self.raw_data.rename(columns={'Bankrupt?': 'Bankrupt'})
            
            print(f"💥 Bankruptcy rate: {self.raw_data['Bankrupt'].mean():.1%}")
            print()
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
        
        return True
    
    def clean_data(self):
        """Step 2: Data Cleaning and Preprocessing"""
        print("🧹 STEP 2: DATA CLEANING & PREPROCESSING")
        print("=" * 50)
        
        # Create a copy for cleaning
        self.cleaned_data = self.raw_data.copy()
        
        # 1. Handle column names (remove leading spaces)
        self.cleaned_data.columns = self.cleaned_data.columns.str.strip()
        
        print("🔧 Cleaning column names...")
        print(f"   Before: '{self.raw_data.columns[1]}'")
        print(f"   After:  '{self.cleaned_data.columns[1]}'")
        
        # 2. Handle missing values
        missing_before = self.cleaned_data.isnull().sum().sum()
        print(f"\n📊 Missing values analysis:")
        print(f"   Total missing values: {missing_before}")
        
        # Fill missing values with median for numerical columns
        numerical_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col != 'Bankrupt':  # Don't fill target variable
                self.cleaned_data[col].fillna(self.cleaned_data[col].median(), inplace=True)
        
        missing_after = self.cleaned_data.isnull().sum().sum()
        print(f"   Missing values after cleaning: {missing_after}")
        
        # 3. Handle extreme outliers
        print(f"\n🎯 Outlier detection and treatment:")
        outlier_count = 0
        
        for col in numerical_cols:
            if col != 'Bankrupt':
                Q1 = self.cleaned_data[col].quantile(0.25)
                Q3 = self.cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((self.cleaned_data[col] < lower_bound) | 
                           (self.cleaned_data[col] > upper_bound)).sum()
                outlier_count += outliers
                
                # Cap extreme outliers instead of removing them
                self.cleaned_data[col] = np.where(
                    self.cleaned_data[col] > upper_bound, upper_bound,
                    np.where(self.cleaned_data[col] < lower_bound, lower_bound, 
                             self.cleaned_data[col])
                )
        
        print(f"   Extreme outliers treated: {outlier_count}")
        
        # 4. Data quality summary
        print(f"\n✅ Data quality summary:")
        print(f"   Shape: {self.cleaned_data.shape}")
        print(f"   Memory usage: {self.cleaned_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        print(f"   Duplicate rows: {self.cleaned_data.duplicated().sum()}")
        
        print()
        return True
    
    def exploratory_data_analysis(self):
        """Step 3: Exploratory Data Analysis"""
        print("🔍 STEP 3: EXPLORATORY DATA ANALYSIS")
        print("=" * 50)
        
        # Basic statistics
        print("📊 Dataset Overview:")
        print(f"   Companies analyzed: {len(self.cleaned_data):,}")
        print(f"   Features available: {len(self.cleaned_data.columns)-1}")
        print(f"   Bankruptcy rate: {self.cleaned_data['Bankrupt'].mean():.1%}")
        
        # Class distribution
        bankrupt_companies = self.cleaned_data['Bankrupt'].sum()
        healthy_companies = len(self.cleaned_data) - bankrupt_companies
        
        print(f"\n📈 Class Distribution:")
        print(f"   Healthy companies: {healthy_companies:,} ({healthy_companies/len(self.cleaned_data):.1%})")
        print(f"   Bankrupt companies: {bankrupt_companies:,} ({bankrupt_companies/len(self.cleaned_data):.1%})")
        
        # Feature analysis
        numerical_features = self.cleaned_data.select_dtypes(include=[np.number]).columns
        numerical_features = [col for col in numerical_features if col != 'Bankrupt']
        
        print(f"\n🎯 Key Financial Metrics Analysis:")
        
        # Analyze key financial ratios
        key_metrics = {
            'Net Income to Total Assets': 'Profitability',
            'Current Ratio': 'Liquidity', 
            'Total debt/Total net worth': 'Leverage',
            'Working Capital to Total Assets': 'Efficiency'
        }
        
        for metric, category in key_metrics.items():
            if metric in self.cleaned_data.columns:
                healthy_mean = self.cleaned_data[self.cleaned_data['Bankrupt'] == 0][metric].mean()
                bankrupt_mean = self.cleaned_data[self.cleaned_data['Bankrupt'] == 1][metric].mean()
                
                print(f"   {category} ({metric}):")
                print(f"     Healthy companies: {healthy_mean:.3f}")
                print(f"     Bankrupt companies: {bankrupt_mean:.3f}")
                print(f"     Difference: {abs(healthy_mean - bankrupt_mean):.3f}")
        
        print()
        return True
    
    def feature_engineering(self):
        """Step 4: Feature Engineering"""
        print("⚙️ STEP 4: FEATURE ENGINEERING")
        print("=" * 50)
        
        # Create enhanced features
        df_enhanced = self.cleaned_data.copy()
        
        # 1. Company size categories
        if 'Total assets to GNP price' in df_enhanced.columns:
            df_enhanced['Company_Size'] = pd.qcut(
                df_enhanced['Total assets to GNP price'], 
                q=4, 
                labels=['Micro', 'Small', 'Medium', 'Large'],
                duplicates='drop'
            )
        else:
            # Fallback: create size based on multiple metrics
            size_metrics = []
            for col in df_enhanced.columns:
                if 'asset' in col.lower() and 'total' in col.lower():
                    size_metrics.append(col)
            
            if size_metrics:
                df_enhanced['Company_Size'] = pd.qcut(
                    df_enhanced[size_metrics[0]], 
                    q=4, 
                    labels=['Micro', 'Small', 'Medium', 'Large'],
                    duplicates='drop'
                )
            else:
                df_enhanced['Company_Size'] = 'Medium'  # Default
        
        # 2. Industry categorization based on financial patterns
        def categorize_industry(row):
            """Categorize companies into industries based on financial patterns"""
            
            asset_turnover = row.get('Total Asset Turnover', 0)
            operating_margin = row.get('Operating Profit Rate', 0)
            debt_ratio = row.get('Total debt/Total net worth', 0)
            r_d_expense = row.get('Research and development expense rate', 0)
            
            # Handle extreme values
            if debt_ratio > 1000:
                debt_ratio = 2.0
            if r_d_expense > 1000:
                r_d_expense = 0.1
            
            # Industry classification logic
            if (0.1 <= asset_turnover <= 0.8 and 0.5 <= operating_margin <= 1.0):
                return 'Manufacturing'
            elif (asset_turnover <= 0.3 and operating_margin >= 0.8):
                return 'Services'
            elif (asset_turnover >= 0.2 and operating_margin <= 0.7):
                return 'Retail'
            elif (debt_ratio >= 1.0):
                return 'Financial'
            elif (r_d_expense >= 0.05):
                return 'Technology'
            elif (0.5 <= debt_ratio <= 2.0 and 0.6 <= operating_margin <= 0.9):
                return 'Construction'
            elif (0.7 <= operating_margin <= 0.95):
                return 'Healthcare'
            else:
                return 'Other'
        
        df_enhanced['Industry'] = df_enhanced.apply(categorize_industry, axis=1)
        
        # 3. Risk score calculation
        def calculate_executive_risk_score(row):
            """Calculate executive-level risk score"""
            
            risk_factors = []
            
            # Profitability risk
            net_income_ratio = row.get('Net Income to Total Assets', 0)
            if net_income_ratio < 0:
                risk_factors.append(0.8)
            elif net_income_ratio < 0.02:
                risk_factors.append(0.6)
            elif net_income_ratio < 0.05:
                risk_factors.append(0.3)
            else:
                risk_factors.append(0.1)
            
            # Liquidity risk
            current_ratio = row.get('Current Ratio', 1.0)
            if current_ratio < 1.0:
                risk_factors.append(0.9)
            elif current_ratio < 1.2:
                risk_factors.append(0.7)
            elif current_ratio < 2.0:
                risk_factors.append(0.3)
            else:
                risk_factors.append(0.1)
            
            # Leverage risk
            debt_ratio = row.get('Total debt/Total net worth', 0)
            if debt_ratio > 1000:
                debt_ratio = 2.0
            
            if debt_ratio > 2.0:
                risk_factors.append(0.9)
            elif debt_ratio > 1.0:
                risk_factors.append(0.7)
            elif debt_ratio > 0.5:
                risk_factors.append(0.4)
            else:
                risk_factors.append(0.1)
            
            # Working capital risk
            working_capital = row.get('Working Capital to Total Assets', 0)
            if working_capital < 0:
                risk_factors.append(0.8)
            elif working_capital < 0.1:
                risk_factors.append(0.6)
            elif working_capital < 0.2:
                risk_factors.append(0.3)
            else:
                risk_factors.append(0.1)
            
            # Calculate weighted average
            weights = [0.3, 0.25, 0.25, 0.2]
            risk_score = sum(factor * weight for factor, weight in zip(risk_factors, weights))
            
            return min(max(risk_score, 0), 1)
        
        df_enhanced['Risk_Score'] = df_enhanced.apply(calculate_executive_risk_score, axis=1)
        
        # 4. Risk level categorization
        def get_risk_level(score):
            if score >= 0.7:
                return 'HIGH'
            elif score >= 0.4:
                return 'MEDIUM'
            else:
                return 'LOW'
        
        df_enhanced['Risk_Level'] = df_enhanced['Risk_Score'].apply(get_risk_level)
        
        # 5. Feature selection
        # Select top features based on correlation with target
        numerical_features = df_enhanced.select_dtypes(include=[np.number]).columns
        numerical_features = [col for col in numerical_features if col not in ['Bankrupt', 'Risk_Score']]
        
        # Calculate correlation with target
        correlations = df_enhanced[numerical_features].corrwith(df_enhanced['Bankrupt']).abs()
        top_features = correlations.nlargest(20).index.tolist()
        
        print(f"🎯 Feature Engineering Summary:")
        print(f"   Original features: {len(numerical_features)}")
        print(f"   Selected top features: {len(top_features)}")
        print(f"   New categorical features: Company_Size, Industry, Risk_Level")
        print(f"   New numerical features: Risk_Score")
        
        # Display top features
        print(f"\n📊 Top 10 Most Correlated Features:")
        for i, feature in enumerate(top_features[:10], 1):
            corr = correlations[feature]
            print(f"   {i:2d}. {feature}: {corr:.3f}")
        
        # Prepare final dataset
        self.features = df_enhanced[top_features + ['Company_Size', 'Industry', 'Risk_Score']]
        self.target = df_enhanced['Bankrupt']
        
        # Encode categorical variables
        le_size = LabelEncoder()
        le_industry = LabelEncoder()
        
        self.features['Company_Size_Encoded'] = le_size.fit_transform(self.features['Company_Size'])
        self.features['Industry_Encoded'] = le_industry.fit_transform(self.features['Industry'])
        
        # Drop original categorical columns
        self.features = self.features.drop(['Company_Size', 'Industry'], axis=1)
        
        print(f"\n✅ Final feature matrix shape: {self.features.shape}")
        print()
        
        return True
    
    def model_training(self):
        """Step 5: Advanced Model Training and Evaluation"""
        print("🤖 STEP 5: ADVANCED MODEL TRAINING & EVALUATION")
        print("=" * 60)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42, stratify=self.target
        )
        
        print(f"📊 Data split:")
        print(f"   Training set: {X_train.shape[0]} companies")
        print(f"   Test set: {X_test.shape[0]} companies")
        print(f"   Features: {X_train.shape[1]}")
        
        # Initialize model suite
        model_suite = ModelSuite(random_state=42)
        validator = ModelValidator(random_state=42)
        
        # Get all available models
        all_models = model_suite.get_all_models()
        
        # Select key models for comprehensive evaluation
        key_models = {
            'Logistic Regression': all_models['Logistic Regression'],
            'Random Forest': all_models['Random Forest'],
            'Gradient Boosting': all_models['Gradient Boosting'],
            'SVM': all_models['SVM'],
            'Neural Network': all_models['Neural Network']
        }
        
        # Add advanced models if available
        if 'XGBoost' in all_models:
            key_models['XGBoost'] = all_models['XGBoost']
        if 'LightGBM' in all_models:
            key_models['LightGBM'] = all_models['LightGBM']
        
        # Add ensemble models
        key_models['Voting Ensemble'] = all_models['Voting Ensemble']
        
        print(f"\n🔄 Training {len(key_models)} advanced models...")
        
        # Perform robust cross-validation
        validation_results = validator.robust_cross_validation(
            key_models, X_train, y_train, cv_folds=5
        )
        
        # Train final models and evaluate on test set
        self.models = {}
        
        for name, model in key_models.items():
            print(f"\n🎯 Final training: {name}")
            
            # Create optimized pipeline
            pipeline = model_suite.create_optimized_pipeline(name, feature_selection=True)
            
            # Fit pipeline
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            
            # Calculate test metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Get validation results
            val_results = validation_results[name]
            
            # Store comprehensive results
            self.models[name] = {
                'pipeline': pipeline,
                'model': model,
                'accuracy': accuracy,
                'auc': auc_score,
                'validation_results': val_results,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'test_accuracy': accuracy,
                'test_auc': auc_score
            }
            
            print(f"   ✅ Test Accuracy: {accuracy:.4f}")
            print(f"   ✅ Test AUC: {auc_score:.4f}")
            print(f"   ✅ CV AUC: {val_results['roc_auc']['test_mean']:.4f} ± {val_results['roc_auc']['test_std']:.4f}")
            print(f"   ✅ Stability: {val_results['overall']['stability']:.4f}")
        
        # Generate comprehensive comparison report
        print(f"\n📊 COMPREHENSIVE MODEL COMPARISON")
        print("=" * 50)
        comparison_df = validator.model_comparison_report()
        
        # Find best model based on validation results
        best_model_name = max(
            self.models.keys(), 
            key=lambda x: self.models[x]['validation_results']['roc_auc']['test_mean']
        )
        best_model = self.models[best_model_name]
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"   Validation AUC: {best_model['validation_results']['roc_auc']['test_mean']:.4f}")
        print(f"   Test AUC: {best_model['test_auc']:.4f}")
        print(f"   Stability: {best_model['validation_results']['overall']['stability']:.4f}")
        print(f"   Overfitting Risk: {best_model['validation_results']['overall']['overfitting_score']:.4f}")
        
        # Store validation results for later use
        self.validation_results = validation_results
        self.comparison_df = comparison_df
        
        print()
        return True
    
    def generate_insights(self):
        """Step 6: Generate Business Insights"""
        print("💡 STEP 6: BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("=" * 50)
        
        # Load enhanced data for insights
        df_enhanced = self.cleaned_data.copy()
        df_enhanced.columns = df_enhanced.columns.str.strip()
        
        # Recreate engineered features for insights
        df_enhanced['Risk_Score'] = df_enhanced.apply(calculate_executive_risk_score, axis=1)
        df_enhanced['Risk_Level'] = df_enhanced['Risk_Score'].apply(
            lambda x: 'HIGH' if x >= 0.7 else 'MEDIUM' if x >= 0.4 else 'LOW'
        )
        
        # Add industry categorization
        def categorize_industry(row):
            asset_turnover = row.get('Total Asset Turnover', 0)
            operating_margin = row.get('Operating Profit Rate', 0)
            debt_ratio = row.get('Total debt/Total net worth', 0)
            r_d_expense = row.get('Research and development expense rate', 0)
            
            if debt_ratio > 1000:
                debt_ratio = 2.0
            if r_d_expense > 1000:
                r_d_expense = 0.1
            
            if (0.1 <= asset_turnover <= 0.8 and 0.5 <= operating_margin <= 1.0):
                return 'Manufacturing'
            elif (asset_turnover <= 0.3 and operating_margin >= 0.8):
                return 'Services'
            elif (asset_turnover >= 0.2 and operating_margin <= 0.7):
                return 'Retail'
            elif (debt_ratio >= 1.0):
                return 'Financial'
            elif (r_d_expense >= 0.05):
                return 'Technology'
            elif (0.5 <= debt_ratio <= 2.0 and 0.6 <= operating_margin <= 0.9):
                return 'Construction'
            elif (0.7 <= operating_margin <= 0.95):
                return 'Healthcare'
            else:
                return 'Other'
        
        df_enhanced['Industry'] = df_enhanced.apply(categorize_industry, axis=1)
        
        # Industry analysis
        industry_risk = df_enhanced.groupby('Industry')['Bankrupt'].agg(['count', 'sum', 'mean']).round(3)
        industry_risk.columns = ['Total_Companies', 'Bankrupt_Companies', 'Bankruptcy_Rate']
        industry_risk = industry_risk.sort_values('Bankruptcy_Rate', ascending=False)
        
        print("🏭 Industry Risk Analysis:")
        for industry, row in industry_risk.head(5).iterrows():
            print(f"   {industry}: {row['Bankruptcy_Rate']:.1%} bankruptcy rate ({row['Bankrupt_Companies']}/{row['Total_Companies']} companies)")
        
        # Risk level analysis
        risk_analysis = df_enhanced.groupby('Risk_Level')['Bankrupt'].agg(['count', 'sum', 'mean']).round(3)
        risk_analysis.columns = ['Total_Companies', 'Bankrupt_Companies', 'Bankruptcy_Rate']
        
        print(f"\n⚠️ Risk Level Analysis:")
        for risk_level, row in risk_analysis.iterrows():
            print(f"   {risk_level} Risk: {row['Bankruptcy_Rate']:.1%} bankruptcy rate ({row['Bankrupt_Companies']}/{row['Total_Companies']} companies)")
        
        # Key financial indicators
        print(f"\n📊 Key Financial Indicators:")
        key_metrics = ['Net Income to Total Assets', 'Current Ratio', 'Total debt/Total net worth']
        
        for metric in key_metrics:
            if metric in df_enhanced.columns:
                healthy_mean = df_enhanced[df_enhanced['Bankrupt'] == 0][metric].mean()
                bankrupt_mean = df_enhanced[df_enhanced['Bankrupt'] == 1][metric].mean()
                
                print(f"   {metric}:")
                print(f"     Healthy: {healthy_mean:.3f} | Bankrupt: {bankrupt_mean:.3f}")
        
        # Executive recommendations
        print(f"\n🎯 Executive Recommendations:")
        
        # Check if HIGH risk level exists
        if 'HIGH' in risk_analysis.index:
            high_risk_rate = risk_analysis.loc['HIGH', 'Bankruptcy_Rate']
            print(f"   1. Focus monitoring on HIGH risk companies (bankruptcy rate: {high_risk_rate:.1%})")
        else:
            print(f"   1. Focus monitoring on companies with Risk Score > 0.7")
        
        print(f"   2. Implement early warning system for companies with Risk Score > 0.7")
        print(f"   3. Regular review of companies in high-risk industries")
        print(f"   4. Use ML models for automated risk assessment (AUC: {max(self.models[m]['auc'] for m in self.models):.3f})")
        
        print()
        return True
    
    def save_results(self):
        """Step 7: Save Results and Models"""
        print("💾 STEP 7: SAVE RESULTS & MODELS")
        print("=" * 50)
        
        # Save enhanced dataset
        df_enhanced = self.cleaned_data.copy()
        df_enhanced.columns = df_enhanced.columns.str.strip()
        
        # Add engineered features
        df_enhanced['Risk_Score'] = df_enhanced.apply(calculate_executive_risk_score, axis=1)
        df_enhanced['Risk_Level'] = df_enhanced['Risk_Score'].apply(
            lambda x: 'HIGH' if x >= 0.7 else 'MEDIUM' if x >= 0.4 else 'LOW'
        )
        
        df_enhanced.to_csv('data/enhanced_dataset.csv', index=False)
        
        # Save model results summary
        results_summary = {
            'model_name': [],
            'accuracy': [],
            'auc_score': [],
            'cv_mean': [],
            'cv_std': []
        }
        
        for name, model_info in self.models.items():
            results_summary['model_name'].append(name)
            results_summary['accuracy'].append(model_info['accuracy'])
            results_summary['auc_score'].append(model_info['auc'])
            # Get CV results from validation_results
            cv_mean = model_info['validation_results']['roc_auc']['test_mean']
            cv_std = model_info['validation_results']['roc_auc']['test_std']
            results_summary['cv_mean'].append(cv_mean)
            results_summary['cv_std'].append(cv_std)
        
        results_df = pd.DataFrame(results_summary)
        results_df.to_csv('data/model_results.csv', index=False)
        
        print(f"✅ Enhanced dataset saved: data/enhanced_dataset.csv")
        print(f"✅ Model results saved: data/model_results.csv")
        print(f"✅ Pipeline completed successfully!")
        
        return True
    
    def run_complete_pipeline(self):
        """Run the complete data analysis pipeline"""
        print("🚀 FINANCIAL RISK ANALYSIS PIPELINE")
        print("=" * 60)
        print("Complete Data Science Lifecycle Demonstration")
        print("=" * 60)
        
        steps = [
            self.extract_data,
            self.clean_data,
            self.exploratory_data_analysis,
            self.feature_engineering,
            self.model_training,
            self.generate_insights,
            self.save_results
        ]
        
        for step in steps:
            if not step():
                print(f"❌ Pipeline failed at step: {step.__name__}")
                return False
        
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        return True

# Helper function for risk score calculation (needed for insights)
def calculate_executive_risk_score(row):
    """Calculate executive-level risk score"""
    risk_factors = []
    
    # Profitability risk
    net_income_ratio = row.get('Net Income to Total Assets', 0)
    if net_income_ratio < 0:
        risk_factors.append(0.8)
    elif net_income_ratio < 0.02:
        risk_factors.append(0.6)
    elif net_income_ratio < 0.05:
        risk_factors.append(0.3)
    else:
        risk_factors.append(0.1)
    
    # Liquidity risk
    current_ratio = row.get('Current Ratio', 1.0)
    if current_ratio < 1.0:
        risk_factors.append(0.9)
    elif current_ratio < 1.2:
        risk_factors.append(0.7)
    elif current_ratio < 2.0:
        risk_factors.append(0.3)
    else:
        risk_factors.append(0.1)
    
    # Leverage risk
    debt_ratio = row.get('Total debt/Total net worth', 0)
    if debt_ratio > 1000:
        debt_ratio = 2.0
    
    if debt_ratio > 2.0:
        risk_factors.append(0.9)
    elif debt_ratio > 1.0:
        risk_factors.append(0.7)
    elif debt_ratio > 0.5:
        risk_factors.append(0.4)
    else:
        risk_factors.append(0.1)
    
    # Working capital risk
    working_capital = row.get('Working Capital to Total Assets', 0)
    if working_capital < 0:
        risk_factors.append(0.8)
    elif working_capital < 0.1:
        risk_factors.append(0.6)
    elif working_capital < 0.2:
        risk_factors.append(0.3)
    else:
        risk_factors.append(0.1)
    
    # Calculate weighted average
    weights = [0.3, 0.25, 0.25, 0.2]
    risk_score = sum(factor * weight for factor, weight in zip(risk_factors, weights))
    
    return min(max(risk_score, 0), 1)

if __name__ == "__main__":
    # Run the complete pipeline
    pipeline = FinancialRiskPipeline('data/data.csv')
    pipeline.run_complete_pipeline()
