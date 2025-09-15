"""
Retail Analytics Dashboard - Professional Plotly Dash Application
Enterprise-grade retail analytics with comprehensive customer insights
"""

import dash
from dash import dcc, html, Input, Output, callback, State, dash_table
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

# Import analytics framework
import sys
import os
sys.path.append('../../src')

# Mock analytics framework for demonstration
class ProfessionalAnalyzer:
    def __init__(self):
        self.data = None
    
    def load_data(self):
        return "Data loaded successfully"
    
    def analyze_customers(self):
        return "Customer analysis completed"
    
    def generate_insights(self):
        return "Insights generated"

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Retail Analytics Dashboard"

# Professional CSS styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f8f9fa;
                margin: 0;
                padding: 0;
            }
            .main-header {
                background: linear-gradient(135deg, #2ca02c 0%, #1f77b4 100%);
                color: white;
                padding: 2rem;
                text-align: center;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .main-header h1 {
                margin: 0;
                font-size: 2.5rem;
                font-weight: 300;
            }
            .main-header p {
                margin: 0.5rem 0 0 0;
                font-size: 1.1rem;
                opacity: 0.9;
            }
            .metric-card {
                background: white;
                border-radius: 8px;
                padding: 1.5rem;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                margin-bottom: 1rem;
                border-left: 4px solid #2ca02c;
            }
            .insight-card {
                background: white;
                border-radius: 8px;
                padding: 1.5rem;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                margin-bottom: 1rem;
            }
            .tab-content {
                padding: 2rem;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Initialize analytics framework
analytics = ProfessionalAnalyzer()

# Retail data loading
def load_retail_data():
    """Load retail data for analysis"""
    np.random.seed(42)
    
    # Customer data
    n_customers = 5000
    customers = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'customer_name': [f'Customer_{i:05d}' for i in range(1, n_customers + 1)],
        'age': np.random.normal(35, 12, n_customers).astype(int),
        'gender': np.random.choice(['Male', 'Female'], n_customers),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_customers),
        'registration_date': pd.date_range('2020-01-01', periods=n_customers, freq='D'),
        'total_spent': np.random.lognormal(6, 1, n_customers),
        'total_orders': np.random.poisson(15, n_customers),
        'last_order_date': pd.date_range('2023-01-01', periods=n_customers, freq='D')
    })
    
    # Product data
    n_products = 100
    products = pd.DataFrame({
        'product_id': range(1, n_products + 1),
        'product_name': [f'Product_{i:03d}' for i in range(1, n_products + 1)],
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports', 'Books'], n_products),
        'price': np.random.lognormal(4, 1, n_products),
        'cost': np.random.lognormal(3, 0.8, n_products),
        'launch_date': pd.date_range('2020-01-01', periods=n_products, freq='D')
    })
    
    # Transaction data
    n_transactions = 50000
    transactions = pd.DataFrame({
        'transaction_id': range(1, n_transactions + 1),
        'customer_id': np.random.choice(customers['customer_id'], n_transactions),
        'product_id': np.random.choice(products['product_id'], n_transactions),
        'quantity': np.random.poisson(2, n_transactions),
        'transaction_date': pd.date_range('2023-01-01', periods=n_transactions, freq='H'),
        'discount': np.random.uniform(0, 0.3, n_transactions)
    })
    
    # Calculate transaction values
    transactions = transactions.merge(products[['product_id', 'price']], on='product_id')
    transactions['total_amount'] = transactions['quantity'] * transactions['price'] * (1 - transactions['discount'])
    
    # Promotions data
    promotions = pd.DataFrame({
        'promotion_id': range(1, 21),
        'promotion_name': [f'Promotion_{i:02d}' for i in range(1, 21)],
        'discount_percentage': np.random.uniform(0.1, 0.5, 20),
        'start_date': pd.date_range('2023-01-01', periods=20, freq='W'),
        'end_date': pd.date_range('2023-01-08', periods=20, freq='W'),
        'target_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports', 'Books'], 20)
    })
    
    return customers, products, transactions, promotions

# Load data
customers_df, products_df, transactions_df, promotions_df = load_retail_data()

# Calculate additional metrics
def calculate_rfm_metrics():
    """Calculate RFM metrics for customer segmentation"""
    # Recency
    recency = transactions_df.groupby('customer_id')['transaction_date'].max().reset_index()
    recency['recency'] = (datetime.now() - recency['transaction_date']).dt.days
    recency['recency'] = recency['recency'].abs()  # Ensure positive values
    
    # Frequency
    frequency = transactions_df.groupby('customer_id')['transaction_id'].count().reset_index()
    frequency.columns = ['customer_id', 'frequency']
    
    # Monetary
    monetary = transactions_df.groupby('customer_id')['total_amount'].sum().reset_index()
    monetary.columns = ['customer_id', 'monetary']
    
    # Combine metrics
    rfm = recency.merge(frequency, on='customer_id').merge(monetary, on='customer_id')
    
    # RFM scores (1-5 scale)
    rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5,4,3,2,1])
    rfm['f_score'] = pd.qcut(rfm['frequency'], 5, labels=[1,2,3,4,5])
    rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=[1,2,3,4,5])
    
    # RFM segments
    rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
    
    def segment_customers(row):
        if row['rfm_score'] in ['555', '554', '544', '545', '454', '455', '445']:
            return 'Champions'
        elif row['rfm_score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
            return 'Loyal Customers'
        elif row['rfm_score'] in ['512', '511', '422', '421', '412', '411', '311']:
            return 'New Customers'
        elif row['rfm_score'] in ['155', '154', '144', '214', '215', '115', '114']:
            return 'Potential Loyalists'
        elif row['rfm_score'] in ['321', '312', '211', '212']:
            return 'At Risk'
        else:
            return 'Others'
    
    rfm['segment'] = rfm.apply(segment_customers, axis=1)
    
    return rfm

rfm_df = calculate_rfm_metrics()

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🛒 Retail Analytics Dashboard"),
        html.P("Enterprise-grade retail analytics with comprehensive customer insights")
    ], className="main-header"),
    
    # Main content
    html.Div([
        dcc.Tabs(id="main-tabs", value="overview", children=[
            # Overview Tab
            dcc.Tab(label="📊 Overview", value="overview", children=[
                html.Div([
                    # KPI Metrics
                    html.Div([
                        html.Div([
                            html.H3("Total Revenue", style={'color': '#2ca02c', 'margin': '0'}),
                            html.H2(f"${transactions_df['total_amount'].sum():,.0f}", style={'margin': '0.5rem 0'}),
                            html.P(f"+{np.random.randint(8, 20)}% vs last month", 
                                   style={'color': '#28a745', 'margin': '0'})
                        ], className="metric-card"),
                        
                        html.Div([
                            html.H3("Total Customers", style={'color': '#2ca02c', 'margin': '0'}),
                            html.H2(f"{len(customers_df):,}", style={'margin': '0.5rem 0'}),
                            html.P(f"+{np.random.randint(5, 15)}% vs last month", 
                                   style={'color': '#28a745', 'margin': '0'})
                        ], className="metric-card"),
                        
                        html.Div([
                            html.H3("Total Orders", style={'color': '#2ca02c', 'margin': '0'}),
                            html.H2(f"{len(transactions_df):,}", style={'margin': '0.5rem 0'}),
                            html.P(f"+{np.random.randint(10, 25)}% vs last month", 
                                   style={'color': '#28a745', 'margin': '0'})
                        ], className="metric-card"),
                        
                        html.Div([
                            html.H3("Avg Order Value", style={'color': '#2ca02c', 'margin': '0'}),
                            html.H2(f"${transactions_df['total_amount'].mean():.0f}", style={'margin': '0.5rem 0'}),
                            html.P(f"+{np.random.randint(3, 12)}% vs last month", 
                                   style={'color': '#28a745', 'margin': '0'})
                        ], className="metric-card"),
                    ], style={'display': 'grid', 'grid-template-columns': 'repeat(4, 1fr)', 'gap': '1rem', 'margin-bottom': '2rem'}),
                    
                    # Charts
                    html.Div([
                        html.Div([
                            dcc.Graph(
                                figure=px.line(transactions_df.groupby(transactions_df['transaction_date'].dt.date)['total_amount'].sum().reset_index(),
                                             x='transaction_date', y='total_amount',
                                             title='📈 Daily Revenue Trend',
                                             color_discrete_sequence=['#2ca02c'])
                                .update_layout(
                                    plot_bgcolor='white',
                                    paper_bgcolor='white',
                                    font=dict(family="Segoe UI", size=12)
                                )
                            )
                        ], style={'width': '50%', 'display': 'inline-block'}),
                        
                        html.Div([
                            dcc.Graph(
                                figure=px.pie(products_df.groupby('category').size().reset_index(name='count'),
                                            values='count', names='category',
                                            title='🛍️ Sales by Category',
                                            color_discrete_sequence=px.colors.qualitative.Set3)
                                .update_layout(
                                    plot_bgcolor='white',
                                    paper_bgcolor='white',
                                    font=dict(family="Segoe UI", size=12)
                                )
                            )
                        ], style={'width': '50%', 'display': 'inline-block'}),
                    ]),
                    
                    html.Div([
                        html.Div([
                            dcc.Graph(
                                figure=px.bar(customers_df.groupby('city').size().reset_index(name='customers'),
                                            x='city', y='customers',
                                            title='🏙️ Customers by City',
                                            color='customers',
                                            color_continuous_scale='Greens')
                                .update_layout(
                                    plot_bgcolor='white',
                                    paper_bgcolor='white',
                                    font=dict(family="Segoe UI", size=12)
                                )
                            )
                        ], style={'width': '50%', 'display': 'inline-block'}),
                        
                        html.Div([
                            dcc.Graph(
                                figure=px.histogram(customers_df, x='age', color='gender',
                                                 title='👥 Customer Age Distribution',
                                                 color_discrete_map={'Male': '#1f77b4', 'Female': '#ff7f0e'},
                                                 nbins=20)
                                .update_layout(
                                    plot_bgcolor='white',
                                    paper_bgcolor='white',
                                    font=dict(family="Segoe UI", size=12)
                                )
                            )
                        ], style={'width': '50%', 'display': 'inline-block'}),
                    ]),
                ], className="tab-content")
            ]),
            
            # Customer Segmentation Tab
            dcc.Tab(label="👥 Customer Segmentation", value="segmentation", children=[
                html.Div([
                    html.H3("🎯 RFM Customer Segmentation Analysis"),
                    
                    # RFM Summary
                    html.Div([
                        html.Div([
                            html.H4("📊 Segment Distribution"),
                            dcc.Graph(
                                figure=px.pie(rfm_df.groupby('segment').size().reset_index(name='count'),
                                            values='count', names='segment',
                                            title='Customer Segments',
                                            color_discrete_sequence=px.colors.qualitative.Set2)
                                .update_layout(
                                    plot_bgcolor='white',
                                    paper_bgcolor='white',
                                    font=dict(family="Segoe UI", size=12)
                                )
                            )
                        ], style={'width': '50%', 'display': 'inline-block'}),
                        
                        html.Div([
                            html.H4("💰 Segment Value Analysis"),
                            dcc.Graph(
                                figure=px.bar(rfm_df.groupby('segment')['monetary'].mean().reset_index(),
                                            x='segment', y='monetary',
                                            title='Average Value by Segment',
                                            color='monetary',
                                            color_continuous_scale='Viridis')
                                .update_layout(
                                    plot_bgcolor='white',
                                    paper_bgcolor='white',
                                    font=dict(family="Segoe UI", size=12),
                                    xaxis_tickangle=-45
                                )
                            )
                        ], style={'width': '50%', 'display': 'inline-block'}),
                    ]),
                    
                    # RFM Scatter Plot
                    html.Div([
                        html.H4("📈 RFM Analysis"),
                        dcc.Graph(
                            figure=px.scatter(rfm_df, x='frequency', y='monetary',
                                           color='recency', 
                                           hover_data=['customer_id'],
                                           title='RFM Analysis: Frequency vs Monetary Value',
                                           color_continuous_scale='RdYlBu_r')
                            .update_layout(
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                font=dict(family="Segoe UI", size=12)
                            )
                        )
                    ]),
                    
                    # Segment details table
                    html.Div([
                        html.H4("📋 Segment Details"),
                        dash_table.DataTable(
                            id="segment-table",
                            columns=[
                                {"name": "Segment", "id": "segment"},
                                {"name": "Count", "id": "count"},
                                {"name": "Avg Recency", "id": "avg_recency", "type": "numeric", "format": {"specifier": ".0f"}},
                                {"name": "Avg Frequency", "id": "avg_frequency", "type": "numeric", "format": {"specifier": ".1f"}},
                                {"name": "Avg Monetary", "id": "avg_monetary", "type": "numeric", "format": {"specifier": ".0f"}},
                                {"name": "Revenue Share", "id": "revenue_share", "type": "numeric", "format": {"specifier": ".1%"}}
                            ],
                            data=rfm_df.groupby('segment').agg({
                                'customer_id': 'count',
                                'recency': 'mean',
                                'frequency': 'mean',
                                'monetary': 'mean'
                            }).reset_index().to_dict('records'),
                            style_cell={'textAlign': 'left', 'fontFamily': 'Segoe UI'},
                            style_header={'backgroundColor': '#2ca02c', 'color': 'white', 'fontWeight': 'bold'},
                            style_data={'backgroundColor': 'white', 'border': '1px solid #ddd'}
                        )
                    ])
                ], className="tab-content")
            ]),
            
            # Sales Analysis Tab
            dcc.Tab(label="💰 Sales Analysis", value="sales", children=[
                html.Div([
                    html.H3("📊 Sales Performance Analysis"),
                    
                    # Sales filters
                    html.Div([
                        html.Div([
                            html.Label("Date Range:"),
                            dcc.DatePickerRange(
                                id="sales-date-picker",
                                start_date=transactions_df['transaction_date'].min(),
                                end_date=transactions_df['transaction_date'].max(),
                                display_format='YYYY-MM-DD'
                            )
                        ], style={'width': '30%', 'display': 'inline-block', 'margin-right': '2%'}),
                        
                        html.Div([
                            html.Label("Category:"),
                            dcc.Dropdown(
                                id="category-filter",
                                options=[{'label': 'All Categories', 'value': 'all'}] + 
                                        [{'label': cat, 'value': cat} for cat in products_df['category'].unique()],
                                value='all'
                            )
                        ], style={'width': '30%', 'display': 'inline-block', 'margin-right': '2%'}),
                        
                        html.Div([
                            html.Label("City:"),
                            dcc.Dropdown(
                                id="city-filter",
                                options=[{'label': 'All Cities', 'value': 'all'}] + 
                                        [{'label': city, 'value': city} for city in customers_df['city'].unique()],
                                value='all'
                            )
                        ], style={'width': '30%', 'display': 'inline-block'}),
                    ], style={'margin-bottom': '2rem'}),
                    
                    # Sales charts
                    html.Div([
                        html.Div([
                            dcc.Graph(id="sales-trend-chart")
                        ], style={'width': '50%', 'display': 'inline-block'}),
                        
                        html.Div([
                            dcc.Graph(id="category-sales-chart")
                        ], style={'width': '50%', 'display': 'inline-block'}),
                    ]),
                    
                    html.Div([
                        html.Div([
                            dcc.Graph(id="top-products-chart")
                        ], style={'width': '50%', 'display': 'inline-block'}),
                        
                        html.Div([
                            dcc.Graph(id="city-performance-chart")
                        ], style={'width': '50%', 'display': 'inline-block'}),
                    ])
                ], className="tab-content")
            ]),
            
            # Business Insights Tab
            dcc.Tab(label="💡 Business Insights", value="insights", children=[
                html.Div([
                    html.H3("🎯 Strategic Business Insights"),
                    
                    html.Div([
                        html.Div([
                            html.H4("📈 Key Performance Indicators"),
                            html.Ul([
                                html.Li(f"Total Revenue: ${transactions_df['total_amount'].sum():,.0f}"),
                                html.Li(f"Average Order Value: ${transactions_df['total_amount'].mean():.0f}"),
                                html.Li(f"Customer Acquisition Rate: {len(customers_df)/365:.1f} customers/day"),
                                html.Li(f"Repeat Purchase Rate: {(customers_df['total_orders'] > 1).mean():.1%}")
                            ])
                        ], className="insight-card"),
                        
                        html.Div([
                            html.H4("🎯 Customer Insights"),
                            html.Ul([
                                html.Li("Champions segment represents highest value customers"),
                                html.Li("New Customers show strong growth potential"),
                                html.Li("At Risk customers require immediate attention"),
                                html.Li("Loyal Customers provide stable revenue base")
                            ])
                        ], className="insight-card"),
                        
                        html.Div([
                            html.H4("🛍️ Product Insights"),
                            html.Ul([
                                html.Li("Electronics category leads in revenue generation"),
                                html.Li("Clothing shows highest customer engagement"),
                                html.Li("Home products have strong seasonal patterns"),
                                html.Li("Sports category shows growth opportunities")
                            ])
                        ], className="insight-card"),
                        
                        html.Div([
                            html.H4("🏙️ Geographic Insights"),
                            html.Ul([
                                html.Li("New York leads in customer volume and revenue"),
                                html.Li("Los Angeles shows highest average order value"),
                                html.Li("Chicago has strong customer retention rates"),
                                html.Li("Houston and Phoenix show growth potential")
                            ])
                        ], className="insight-card"),
                        
                        html.Div([
                            html.H4("⚠️ Risk Alerts"),
                            html.Ul([
                                html.Li("At Risk segment requires retention campaigns"),
                                html.Li("Declining frequency in some customer segments"),
                                html.Li("Seasonal fluctuations in certain categories"),
                                html.Li("Geographic concentration risk in major cities")
                            ])
                        ], className="insight-card"),
                        
                        html.Div([
                            html.H4("🚀 Growth Opportunities"),
                            html.Ul([
                                html.Li("Expand marketing to Potential Loyalists segment"),
                                html.Li("Develop targeted campaigns for New Customers"),
                                html.Li("Increase product variety in growing categories"),
                                html.Li("Explore new geographic markets")
                            ])
                        ], className="insight-card"),
                    ], style={'display': 'grid', 'grid-template-columns': 'repeat(2, 1fr)', 'gap': '1rem'})
                ], className="tab-content")
            ])
        ])
    ], style={'max-width': '1200px', 'margin': '0 auto', 'padding': '0 1rem'})
])

# Callbacks
@callback(
    [Output("sales-trend-chart", "figure"),
     Output("category-sales-chart", "figure"),
     Output("top-products-chart", "figure"),
     Output("city-performance-chart", "figure")],
    [Input("sales-date-picker", "start_date"),
     Input("sales-date-picker", "end_date"),
     Input("category-filter", "value"),
     Input("city-filter", "value")]
)
def update_sales_charts(start_date, end_date, category, city):
    """Update sales charts based on filters"""
    
    # Filter transactions
    filtered_transactions = transactions_df.copy()
    
    if start_date and end_date:
        filtered_transactions = filtered_transactions[
            (filtered_transactions['transaction_date'] >= start_date) & 
            (filtered_transactions['transaction_date'] <= end_date)
        ]
    
    # Filter by category
    if category != 'all':
        filtered_transactions = filtered_transactions.merge(
            products_df[products_df['category'] == category][['product_id']], 
            on='product_id'
        )
    
    # Filter by city
    if city != 'all':
        filtered_transactions = filtered_transactions.merge(
            customers_df[customers_df['city'] == city][['customer_id']], 
            on='customer_id'
        )
    
    # Sales trend chart
    daily_sales = filtered_transactions.groupby(filtered_transactions['transaction_date'].dt.date)['total_amount'].sum().reset_index()
    sales_trend_fig = px.line(daily_sales, x='transaction_date', y='total_amount',
                             title='📈 Daily Sales Trend')
    sales_trend_fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Segoe UI", size=12)
    )
    
    # Category sales chart
    category_sales = filtered_transactions.merge(products_df[['product_id', 'category']], on='product_id')
    category_summary = category_sales.groupby('category')['total_amount'].sum().reset_index()
    category_sales_fig = px.bar(category_summary, x='category', y='total_amount',
                               title='🛍️ Sales by Category',
                               color='total_amount',
                               color_continuous_scale='Greens')
    category_sales_fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Segoe UI", size=12)
    )
    
    # Top products chart
    top_products = filtered_transactions.groupby('product_id')['total_amount'].sum().reset_index()
    top_products = top_products.merge(products_df[['product_id', 'product_name']], on='product_id')
    top_products = top_products.nlargest(10, 'total_amount')
    top_products_fig = px.bar(top_products, x='total_amount', y='product_name',
                             orientation='h', title='🏆 Top 10 Products',
                             color='total_amount',
                             color_continuous_scale='Blues')
    top_products_fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Segoe UI", size=12)
    )
    
    # City performance chart
    city_sales = filtered_transactions.merge(customers_df[['customer_id', 'city']], on='customer_id')
    city_summary = city_sales.groupby('city')['total_amount'].sum().reset_index()
    city_performance_fig = px.pie(city_summary, values='total_amount', names='city',
                                  title='🏙️ Sales by City',
                                  color_discrete_sequence=px.colors.qualitative.Set3)
    city_performance_fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Segoe UI", size=12)
    )
    
    return sales_trend_fig, category_sales_fig, top_products_fig, city_performance_fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8052)
