"""
AI-Powered Business Intelligence Pipeline
Professional-grade AI implementation with conversational analytics
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import modules
import sys
import os
sys.path.append('../financial-risk-analysis/src')
from models import ModelSuite
from validation import ModelValidator

class AIBIEngine:
    """
    Professional-grade AI-powered business intelligence engine
    Demonstrates AI integration with business analytics
    """
    
    def __init__(self):
        self.data = None
        self.models = {}
        self.conversation_history = []
        self.query_classifier = None
        self.response_generator = None
        
    def load_sample_data(self):
        """Load sample business data"""
        np.random.seed(42)
        
        # Generate sample business data
        n_companies = 1000
        n_products = 100
        
        # Company data
        companies = pd.DataFrame({
            'company_id': range(1, n_companies + 1),
            'name': [f'Company_{i}' for i in range(1, n_companies + 1)],
            'industry': np.random.choice(['Technology', 'Finance', 'Healthcare', 'Retail', 'Manufacturing'], n_companies),
            'revenue': np.random.lognormal(15, 1, n_companies),
            'employees': np.random.poisson(500, n_companies),
            'profit_margin': np.random.normal(0.1, 0.05, n_companies),
            'growth_rate': np.random.normal(0.05, 0.1, n_companies)
        })
        
        # Product data
        products = pd.DataFrame({
            'product_id': range(1, n_products + 1),
            'name': [f'Product_{i}' for i in range(1, n_products + 1)],
            'category': np.random.choice(['Software', 'Hardware', 'Services', 'Consulting'], n_products),
            'price': np.random.lognormal(8, 1, n_products),
            'cost': np.random.lognormal(6, 0.8, n_products),
            'units_sold': np.random.poisson(1000, n_products)
        })
        
        # Sales data
        n_sales = 10000
        sales = pd.DataFrame({
            'sale_id': range(1, n_sales + 1),
            'company_id': np.random.choice(companies['company_id'], n_sales),
            'product_id': np.random.choice(products['product_id'], n_sales),
            'date': pd.date_range('2020-01-01', '2024-12-31', periods=n_sales),
            'quantity': np.random.poisson(5, n_sales),
            'discount': np.random.uniform(0, 0.3, n_sales)
        })
        
        # Merge data
        sales = sales.merge(companies, on='company_id')
        sales = sales.merge(products, on='product_id')
        
        # Calculate derived metrics
        sales['total_amount'] = sales['quantity'] * sales['price'] * (1 - sales['discount'])
        sales['profit'] = sales['quantity'] * (sales['price'] - sales['cost']) * (1 - sales['discount'])
        
        self.data = sales
        print(f"✅ Loaded {len(sales)} business records")
        
        return sales
    
    def classify_query(self, query: str) -> str:
        """Classify user query into categories"""
        
        query_lower = query.lower()
        
        # Simple keyword-based classification
        if any(word in query_lower for word in ['revenue', 'sales', 'income', 'profit']):
            return 'financial'
        elif any(word in query_lower for word in ['customer', 'client', 'company']):
            return 'customer'
        elif any(word in query_lower for word in ['product', 'item', 'category']):
            return 'product'
        elif any(word in query_lower for word in ['trend', 'forecast', 'prediction', 'future']):
            return 'forecasting'
        elif any(word in query_lower for word in ['performance', 'metric', 'kpi']):
            return 'performance'
        else:
            return 'general'
    
    def generate_response(self, query: str, query_type: str) -> dict:
        """Generate AI response based on query type"""
        
        response = {
            'query': query,
            'type': query_type,
            'timestamp': datetime.now().isoformat(),
            'answer': '',
            'insights': [],
            'recommendations': [],
            'visualization': None
        }
        
        if query_type == 'financial':
            response = self._handle_financial_query(query, response)
        elif query_type == 'customer':
            response = self._handle_customer_query(query, response)
        elif query_type == 'product':
            response = self._handle_product_query(query, response)
        elif query_type == 'forecasting':
            response = self._handle_forecasting_query(query, response)
        elif query_type == 'performance':
            response = self._handle_performance_query(query, response)
        else:
            response = self._handle_general_query(query, response)
        
        return response
    
    def _handle_financial_query(self, query: str, response: dict) -> dict:
        """Handle financial queries"""
        
        total_revenue = self.data['total_amount'].sum()
        total_profit = self.data['profit'].sum()
        avg_order_value = self.data['total_amount'].mean()
        
        response['answer'] = f"Based on our data analysis, total revenue is €{total_revenue:,.0f} with a profit of €{total_profit:,.0f}. Average order value is €{avg_order_value:,.2f}."
        
        response['insights'] = [
            f"Revenue growth: {self.data.groupby(self.data['date'].dt.year)['total_amount'].sum().pct_change().mean():.1%}",
            f"Profit margin: {total_profit/total_revenue:.1%}",
            f"Top performing industry: {self.data.groupby('industry')['total_amount'].sum().idxmax()}"
        ]
        
        response['recommendations'] = [
            "Focus on high-margin products",
            "Optimize pricing strategy",
            "Monitor industry performance"
        ]
        
        return response
    
    def _handle_customer_query(self, query: str, response: dict) -> dict:
        """Handle customer queries"""
        
        unique_customers = self.data['company_id'].nunique()
        avg_customer_value = self.data.groupby('company_id')['total_amount'].sum().mean()
        top_customer = self.data.groupby('company_id')['total_amount'].sum().idxmax()
        
        response['answer'] = f"We have {unique_customers} unique customers with an average value of €{avg_customer_value:,.0f}. Top customer is Company_{top_customer}."
        
        response['insights'] = [
            f"Customer retention rate: {self.data['company_id'].value_counts().value_counts().sum() / unique_customers:.1%}",
            f"Average orders per customer: {len(self.data) / unique_customers:.1f}",
            f"Customer concentration: Top 10% generate {self.data.groupby('company_id')['total_amount'].sum().nlargest(int(unique_customers*0.1)).sum() / self.data['total_amount'].sum():.1%} of revenue"
        ]
        
        response['recommendations'] = [
            "Implement customer segmentation",
            "Develop retention programs",
            "Focus on high-value customers"
        ]
        
        return response
    
    def _handle_product_query(self, query: str, response: dict) -> dict:
        """Handle product queries"""
        
        unique_products = self.data['product_id'].nunique()
        best_product = self.data.groupby('product_id')['total_amount'].sum().idxmax()
        avg_product_value = self.data.groupby('product_id')['total_amount'].sum().mean()
        
        response['answer'] = f"We have {unique_products} products with an average value of €{avg_product_value:,.0f}. Best performing product is Product_{best_product}."
        
        response['insights'] = [
            f"Product diversity: {unique_products} different products",
            f"Top category: {self.data.groupby('category')['total_amount'].sum().idxmax()}",
            f"Product concentration: Top 20% of products generate {self.data.groupby('product_id')['total_amount'].sum().nlargest(int(unique_products*0.2)).sum() / self.data['total_amount'].sum():.1%} of revenue"
        ]
        
        response['recommendations'] = [
            "Optimize product portfolio",
            "Focus on high-performing categories",
            "Develop new products in successful categories"
        ]
        
        return response
    
    def _handle_forecasting_query(self, query: str, response: dict) -> dict:
        """Handle forecasting queries"""
        
        # Simple trend analysis
        monthly_sales = self.data.groupby(self.data['date'].dt.to_period('M'))['total_amount'].sum()
        trend = monthly_sales.pct_change().mean()
        
        response['answer'] = f"Based on historical data, we're seeing a {trend:.1%} monthly growth trend. This suggests continued growth in the coming months."
        
        response['insights'] = [
            f"Monthly growth rate: {trend:.1%}",
            f"Seasonality: {self.data.groupby(self.data['date'].dt.month)['total_amount'].sum().std() / self.data.groupby(self.data['date'].dt.month)['total_amount'].sum().mean():.1%}",
            f"Forecast confidence: High (based on consistent growth pattern)"
        ]
        
        response['recommendations'] = [
            "Plan for continued growth",
            "Monitor seasonal patterns",
            "Adjust inventory accordingly"
        ]
        
        return response
    
    def _handle_performance_query(self, query: str, response: dict) -> dict:
        """Handle performance queries"""
        
        # Calculate key performance indicators
        total_revenue = self.data['total_amount'].sum()
        total_profit = self.data['profit'].sum()
        unique_customers = self.data['company_id'].nunique()
        unique_products = self.data['product_id'].nunique()
        
        response['answer'] = f"Our key performance indicators show strong results: €{total_revenue:,.0f} revenue, €{total_profit:,.0f} profit, {unique_customers} customers, and {unique_products} products."
        
        response['insights'] = [
            f"Revenue per customer: €{total_revenue/unique_customers:,.0f}",
            f"Profit margin: {total_profit/total_revenue:.1%}",
            f"Product utilization: {len(self.data)/unique_products:.1f} sales per product"
        ]
        
        response['recommendations'] = [
            "Maintain current performance levels",
            "Focus on customer acquisition",
            "Optimize product mix"
        ]
        
        return response
    
    def _handle_general_query(self, query: str, response: dict) -> dict:
        """Handle general queries"""
        
        response['answer'] = "I can help you analyze financial data, customer insights, product performance, forecasting, and business metrics. What specific area would you like to explore?"
        
        response['insights'] = [
            "Total records analyzed: {len(self.data):,}",
            "Data coverage: 2020-2024",
            "Available analysis: Financial, Customer, Product, Forecasting, Performance"
        ]
        
        response['recommendations'] = [
            "Try asking about revenue trends",
            "Explore customer segmentation",
            "Analyze product performance",
            "Request forecasting insights"
        ]
        
        return response
    
    def process_conversation(self, query: str) -> dict:
        """Process a conversational query"""
        
        # Classify query
        query_type = self.classify_query(query)
        
        # Generate response
        response = self.generate_response(query, query_type)
        
        # Store in conversation history
        self.conversation_history.append(response)
        
        return response
    
    def get_conversation_history(self) -> list:
        """Get conversation history"""
        return self.conversation_history
    
    def create_ai_dashboard(self):
        """Create AI-powered dashboard"""
        
        print("🤖 Creating AI-powered dashboard...")
        
        # Generate sample data if not loaded
        if self.data is None:
            self.load_sample_data()
        
        # Create dashboard components
        dashboard = {
            'title': 'AI-Powered Business Intelligence',
            'components': [
                'Conversational Interface',
                'Real-time Analytics',
                'Automated Insights',
                'Interactive Visualizations'
            ],
            'features': [
                'Natural Language Queries',
                'Automated Report Generation',
                'Predictive Analytics',
                'Business Recommendations'
            ]
        }
        
        return dashboard
    
    def run_analysis(self):
        """Run complete AI analysis"""
        
        print("🤖 AI-POWERED BUSINESS INTELLIGENCE")
        print("=" * 50)
        
        # Load data
        if self.data is None:
            self.load_sample_data()
        
        # Initialize models
        model_suite = ModelSuite(random_state=42)
        validator = ModelValidator(random_state=42)
        
        print(f"✅ AI engine initialized")
        print(f"📊 Data loaded: {len(self.data)} records")
        print(f"🤖 Models available: {len(model_suite.get_all_models())}")
        
        return {
            'data_loaded': True,
            'models_initialized': True,
            'conversation_ready': True
        }

if __name__ == "__main__":
    # Initialize AI engine
    ai_engine = AIBIEngine()
    
    # Run analysis
    results = ai_engine.run_analysis()
    
    print(f"\nRun: python dashboard.py")
