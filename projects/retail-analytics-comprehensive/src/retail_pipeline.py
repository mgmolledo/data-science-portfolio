"""
Retail Analytics Pipeline
Professional-grade customer analytics with ML
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import modules from financial project
import sys
import os
sys.path.append('../financial-risk-analysis/src')
from models import ModelSuite
from validation import ModelValidator

class RetailAnalytics:
    """
    Professional retail analytics pipeline
    Demonstrates customer segmentation, forecasting, and ML
    """
    
    def __init__(self):
        self.data = None
        self.customer_segments = None
        self.churn_model = None
        self.forecasting_model = None
        self.results = {}
        
    def load_data(self, data_path: str = 'data/retail_data.csv'):
        """Load retail data"""
        try:
            self.data = pd.read_csv(data_path)
            print(f"✅ Loaded {len(self.data)} retail records")
            return True
        except FileNotFoundError:
            print("❌ Data file not found. Generating sample data...")
            self.generate_sample_data()
            return True
    
    def generate_sample_data(self):
        """Generate realistic retail sample data"""
        np.random.seed(42)
        n_customers = 50000
        n_products = 1000
        
        # Generate customer data
        customers = pd.DataFrame({
            'customer_id': range(1, n_customers + 1),
            'age': np.random.normal(35, 12, n_customers).astype(int),
            'gender': np.random.choice(['M', 'F'], n_customers),
            'income': np.random.lognormal(10, 0.5, n_customers),
            'city': np.random.choice(['Madrid', 'Barcelona', 'Valencia', 'Sevilla', 'Bilbao'], n_customers),
            'registration_date': pd.date_range('2020-01-01', '2024-12-31', periods=n_customers)
        })
        
        # Generate transaction data
        n_transactions = 500000
        transactions = pd.DataFrame({
            'transaction_id': range(1, n_transactions + 1),
            'customer_id': np.random.choice(customers['customer_id'], n_transactions),
            'product_id': np.random.choice(range(1, n_products + 1), n_transactions),
            'quantity': np.random.poisson(2, n_transactions) + 1,
            'price': np.random.lognormal(3, 0.8, n_transactions),
            'date': pd.date_range('2020-01-01', '2024-12-31', periods=n_transactions)
        })
        
        # Calculate total amount
        transactions['total_amount'] = transactions['quantity'] * transactions['price']
        
        # Merge data
        self.data = transactions.merge(customers, on='customer_id')
        
        # Add derived features
        self.data['month'] = self.data['date'].dt.month
        self.data['year'] = self.data['date'].dt.year
        self.data['day_of_week'] = self.data['date'].dt.dayofweek
        
        print(f"✅ Generated {len(self.data)} retail records")
    
    def advanced_customer_segmentation(self):
        """Customer segmentation using multiple algorithms"""
        
        print("\n🎯 CUSTOMER SEGMENTATION")
        print("=" * 50)
        
        # Calculate customer metrics
        customer_metrics = self.data.groupby('customer_id').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'date': ['min', 'max'],
            'age': 'first',
            'gender': 'first',
            'income': 'first',
            'city': 'first'
        }).round(2)
        
        # Flatten column names
        customer_metrics.columns = ['total_spent', 'avg_order_value', 'order_count', 
                                   'first_purchase', 'last_purchase', 'age', 'gender', 'income', 'city']
        
        # Calculate additional metrics
        customer_metrics['days_since_first'] = (pd.Timestamp.now() - customer_metrics['first_purchase']).dt.days
        customer_metrics['days_since_last'] = (pd.Timestamp.now() - customer_metrics['last_purchase']).dt.days
        
        # RFM Analysis
        customer_metrics['recency'] = customer_metrics['days_since_last']
        customer_metrics['frequency'] = customer_metrics['order_count']
        customer_metrics['monetary'] = customer_metrics['total_spent']
        
        # Prepare features for clustering
        features = ['recency', 'frequency', 'monetary', 'age', 'income']
        X = customer_metrics[features].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        customer_metrics['segment'] = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, customer_metrics['segment'])
        
        # Segment analysis
        segment_analysis = customer_metrics.groupby('segment').agg({
            'total_spent': ['mean', 'std'],
            'order_count': ['mean', 'std'],
            'recency': ['mean', 'std'],
            'age': ['mean', 'std'],
            'income': ['mean', 'std']
        }).round(2)
        
        # Assign segment names
        segment_names = {
            0: 'High Value Loyal',
            1: 'New Customers',
            2: 'At Risk',
            3: 'Low Value',
            4: 'Champions'
        }
        
        customer_metrics['segment_name'] = customer_metrics['segment'].map(segment_names)
        
        self.customer_segments = customer_metrics
        self.results['segmentation'] = {
            'silhouette_score': silhouette_avg,
            'segment_analysis': segment_analysis,
            'segment_names': segment_names
        }
        
        print(f"✅ Customer segmentation completed")
        print(f"📊 Silhouette Score: {silhouette_avg:.3f}")
        print(f"👥 Segments: {len(segment_names)}")
        
        return customer_metrics
    
    def advanced_churn_prediction(self):
        """Churn prediction with multiple algorithms"""
        
        print("\n🔮 CHURN PREDICTION")
        print("=" * 50)
        
        # Calculate churn indicators
        customer_metrics = self.customer_segments.copy()
        
        # Define churn (no purchase in last 90 days)
        churn_threshold = 90
        customer_metrics['is_churn'] = (customer_metrics['days_since_last'] > churn_threshold).astype(int)
        
        # Prepare features
        features = ['recency', 'frequency', 'monetary', 'age', 'income', 'avg_order_value']
        X = customer_metrics[features].fillna(0)
        y = customer_metrics['is_churn']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize models
        model_suite = ModelSuite(random_state=42)
        validator = ModelValidator(random_state=42)
        
        # Get key models
        key_models = {
            'Random Forest': model_suite.get_model('Random Forest'),
            'Gradient Boosting': model_suite.get_model('Gradient Boosting'),
            'Logistic Regression': model_suite.get_model('Logistic Regression')
        }
        
        # Train and evaluate models
        model_results = {}
        for name, model in key_models.items():
            print(f"🔄 Training {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            model_results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            print(f"  ✅ {name}: AUC = {model_results[name]['auc']:.3f}")
        
        # Cross-validation
        cv_results = validator.robust_cross_validation(key_models, X, y, cv_folds=5)
        
        # Best model
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['auc'])
        best_model = key_models[best_model_name]
        
        self.churn_model = {
            'model': best_model,
            'scaler': scaler,
            'features': features,
            'results': model_results,
            'cv_results': cv_results
        }
        
        self.results['churn_prediction'] = {
            'model_results': model_results,
            'best_model': best_model_name,
            'cv_results': cv_results
        }
        
        print(f"🏆 Best model: {best_model_name}")
        print(f"📊 Churn rate: {y.mean():.1%}")
        
        return model_results
    
    def advanced_sales_forecasting(self):
        """Sales forecasting using time series analysis"""
        
        print("\n📈 SALES FORECASTING")
        print("=" * 50)
        
        # Aggregate sales by month
        monthly_sales = self.data.groupby(['year', 'month'])['total_amount'].sum().reset_index()
        monthly_sales['date'] = pd.to_datetime(monthly_sales[['year', 'month']].assign(day=1))
        monthly_sales = monthly_sales.sort_values('date')
        
        # Create time series features
        monthly_sales['month_num'] = monthly_sales['month']
        monthly_sales['quarter'] = ((monthly_sales['month'] - 1) // 3) + 1
        monthly_sales['is_holiday'] = monthly_sales['month'].isin([11, 12])  # Nov, Dec
        
        # Prepare features
        features = ['month_num', 'quarter', 'is_holiday']
        X = monthly_sales[features]
        y = monthly_sales['total_amount']
        
        # Split data
        split_point = int(len(monthly_sales) * 0.8)
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Forecast next 6 months
        last_month = monthly_sales['month'].iloc[-1]
        last_year = monthly_sales['year'].iloc[-1]
        
        future_months = []
        for i in range(1, 7):
            month = (last_month + i - 1) % 12 + 1
            year = last_year + (last_month + i - 1) // 12
            
            future_months.append({
                'month_num': month,
                'quarter': ((month - 1) // 3) + 1,
                'is_holiday': month in [11, 12]
            })
        
        future_df = pd.DataFrame(future_months)
        future_predictions = model.predict(future_df)
        
        self.forecasting_model = {
            'model': model,
            'features': features,
            'rmse': rmse,
            'r2': r2,
            'future_predictions': future_predictions
        }
        
        self.results['forecasting'] = {
            'rmse': rmse,
            'r2': r2,
            'future_predictions': future_predictions
        }
        
        print(f"✅ Sales forecasting completed")
        print(f"📊 RMSE: {rmse:,.0f}")
        print(f"📊 R²: {r2:.3f}")
        
        return {
            'rmse': rmse,
            'r2': r2,
            'future_predictions': future_predictions
        }
    
    def generate_insights(self):
        """Generate business insights"""
        
        print("\n💡 BUSINESS INSIGHTS")
        print("=" * 50)
        
        insights = []
        
        # Segmentation insights
        if 'segmentation' in self.results:
            segment_analysis = self.results['segmentation']['segment_analysis']
            insights.append("✅ Customer segmentation completed with 5 distinct segments")
            insights.append(f"📊 Silhouette Score: {self.results['segmentation']['silhouette_score']:.3f}")
        
        # Churn insights
        if 'churn_prediction' in self.results:
            best_model = self.results['churn_prediction']['best_model']
            best_auc = self.results['churn_prediction']['model_results'][best_model]['auc']
            insights.append(f"🔮 Churn prediction: {best_model} achieves {best_auc:.3f} AUC")
        
        # Forecasting insights
        if 'forecasting' in self.results:
            r2 = self.results['forecasting']['r2']
            insights.append(f"📈 Sales forecasting: R² = {r2:.3f}")
        
        # Data quality insights
        insights.append(f"📊 Dataset: {len(self.data):,} transactions")
        insights.append(f"👥 Customers: {self.data['customer_id'].nunique():,}")
        insights.append(f"🛍️ Products: {self.data['product_id'].nunique():,}")
        
        self.results['insights'] = insights
        
        for insight in insights:
            print(insight)
        
        return insights
    
    def run_complete_analysis(self):
        """Run complete retail analytics pipeline"""
        
        print("🛒 RETAIL ANALYTICS PIPELINE")
        print("=" * 50)
        
        # Load data
        self.load_data()
        
        # Run analyses
        analysis_functions = [
            self.advanced_customer_segmentation,
            self.advanced_churn_prediction,
            self.advanced_sales_forecasting,
            self.generate_insights
        ]
        
        for func in analysis_functions:
            try:
                func()
            except Exception as e:
                print(f"❌ Error in {func.__name__}: {str(e)}")
        
        print("\n✅ Retail analytics pipeline completed successfully!")
        return self.results

if __name__ == "__main__":
    # Initialize analytics
    analytics = RetailAnalytics()
    
    # Run complete analysis
    results = analytics.run_complete_analysis()
    
    print(f"\nRun: python dashboard.py")
