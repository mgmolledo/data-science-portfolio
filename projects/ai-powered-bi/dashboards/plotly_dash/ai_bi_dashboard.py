"""
AI-Powered Business Intelligence Dashboard - Professional Plotly Dash Application
Enterprise-grade conversational analytics interface with advanced AI capabilities
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

# Import AI engine
import sys
import os
sys.path.append('../../src')

# Mock AI engine for demonstration
class AIBIEngine:
    def __init__(self):
        self.responses = {
            'revenue': 'Revenue shows strong growth with 12% increase month-over-month.',
            'customers': 'Customer base is expanding with 8% growth in new acquisitions.',
            'products': 'Product A leads in sales, followed by Product B and C.',
            'segments': 'Premium customers show highest value and retention rates.'
        }
    
    def generate_response(self, query):
        query_lower = query.lower()
        for key, response in self.responses.items():
            if key in query_lower:
                return response
        return 'I can help you analyze revenue, customers, products, and customer segments. Please ask specific questions about these areas.'

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "AI-Powered Business Intelligence Platform"

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
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
                border-left: 4px solid #667eea;
            }
            .chat-container {
                background: white;
                border-radius: 8px;
                padding: 1.5rem;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                margin-bottom: 1rem;
            }
            .chat-message {
                padding: 1rem;
                margin: 0.5rem 0;
                border-radius: 8px;
                max-width: 80%;
            }
            .user-message {
                background-color: #e3f2fd;
                margin-left: auto;
                text-align: right;
            }
            .ai-message {
                background-color: #f3e5f5;
                margin-right: auto;
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

# Initialize AI engine
ai_engine = AIBIEngine()

# Business data loading
def load_business_data():
    """Load business data for analysis"""
    np.random.seed(42)
    
    # Sales data
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    sales_data = pd.DataFrame({
        'date': dates,
        'sales': np.random.normal(1000, 200, 365),
        'revenue': np.random.normal(50000, 10000, 365),
        'customers': np.random.normal(500, 50, 365),
        'conversion_rate': np.random.normal(0.15, 0.03, 365)
    })
    
    # Customer segments
    segments = ['Premium', 'Standard', 'Basic', 'Enterprise']
    segment_data = pd.DataFrame({
        'segment': segments,
        'customers': [1200, 3500, 2800, 500],
        'avg_order_value': [450, 180, 95, 1200],
        'retention_rate': [0.85, 0.72, 0.58, 0.92]
    })
    
    # Product performance
    products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
    product_data = pd.DataFrame({
        'product': products,
        'sales': np.random.normal(1000, 200, 5),
        'profit_margin': np.random.normal(0.25, 0.05, 5),
        'market_share': np.random.normal(0.20, 0.05, 5)
    })
    
    return sales_data, segment_data, product_data

# Load data
sales_data, segment_data, product_data = load_business_data()

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🤖 AI-Powered Business Intelligence Platform"),
        html.P("Enterprise-grade conversational analytics with advanced AI capabilities")
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
                            html.H3("Total Revenue", style={'color': '#667eea', 'margin': '0'}),
                            html.H2(f"${sales_data['revenue'].sum():,.0f}", style={'margin': '0.5rem 0'}),
                            html.P(f"+{np.random.randint(5, 15)}% vs last month", 
                                   style={'color': '#28a745', 'margin': '0'})
                        ], className="metric-card"),
                        
                        html.Div([
                            html.H3("Total Customers", style={'color': '#667eea', 'margin': '0'}),
                            html.H2(f"{sales_data['customers'].sum():,.0f}", style={'margin': '0.5rem 0'}),
                            html.P(f"+{np.random.randint(3, 12)}% vs last month", 
                                   style={'color': '#28a745', 'margin': '0'})
                        ], className="metric-card"),
                        
                        html.Div([
                            html.H3("Conversion Rate", style={'color': '#667eea', 'margin': '0'}),
                            html.H2(f"{sales_data['conversion_rate'].mean():.1%}", style={'margin': '0.5rem 0'}),
                            html.P(f"+{np.random.randint(1, 5)}% vs last month", 
                                   style={'color': '#28a745', 'margin': '0'})
                        ], className="metric-card"),
                        
                        html.Div([
                            html.H3("Avg Order Value", style={'color': '#667eea', 'margin': '0'}),
                            html.H2(f"${sales_data['sales'].mean():.0f}", style={'margin': '0.5rem 0'}),
                            html.P(f"+{np.random.randint(2, 8)}% vs last month", 
                                   style={'color': '#28a745', 'margin': '0'})
                        ], className="metric-card"),
                    ], style={'display': 'grid', 'grid-template-columns': 'repeat(4, 1fr)', 'gap': '1rem', 'margin-bottom': '2rem'}),
                    
                    # Charts
                    html.Div([
                        html.Div([
                            dcc.Graph(
                                figure=px.line(sales_data, x='date', y='revenue', 
                                             title='📈 Revenue Trend',
                                             color_discrete_sequence=['#667eea'])
                                .update_layout(
                                    plot_bgcolor='white',
                                    paper_bgcolor='white',
                                    font=dict(family="Segoe UI", size=12)
                                )
                            )
                        ], style={'width': '50%', 'display': 'inline-block'}),
                        
                        html.Div([
                            dcc.Graph(
                                figure=px.bar(segment_data, x='segment', y='customers',
                                            title='👥 Customer Segments',
                                            color='customers',
                                            color_continuous_scale='Blues')
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
                                figure=px.pie(product_data, values='sales', names='product',
                                             title='🛍️ Product Performance',
                                             color_discrete_sequence=px.colors.qualitative.Set3)
                                .update_layout(
                                    plot_bgcolor='white',
                                    paper_bgcolor='white',
                                    font=dict(family="Segoe UI", size=12)
                                )
                            )
                        ], style={'width': '50%', 'display': 'inline-block'}),
                        
                        html.Div([
                            dcc.Graph(
                                figure=px.scatter(segment_data, x='avg_order_value', y='retention_rate',
                                                size='customers', color='segment',
                                                title='💰 Customer Value vs Retention',
                                                hover_data=['customers'])
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
            
            # AI Chat Tab
            dcc.Tab(label="🤖 AI Assistant", value="chat", children=[
                html.Div([
                    html.Div([
                        html.H3("💬 Conversational Analytics"),
                        html.P("Ask questions about your business data in natural language"),
                        
                        # Chat messages container
                        html.Div(id="chat-messages", children=[
                            html.Div([
                                html.P("👋 Hello! I'm your AI business intelligence assistant. Ask me anything about your data!",
                                       style={'margin': '0'})
                            ], className="chat-message ai-message")
                        ], style={'height': '400px', 'overflow-y': 'auto', 'margin-bottom': '1rem'}),
                        
                        # Input area
                        html.Div([
                            dcc.Input(
                                id="chat-input",
                                type="text",
                                placeholder="Ask me about your business data...",
                                style={'width': '80%', 'padding': '0.75rem', 'border': '1px solid #ddd', 'border-radius': '4px'}
                            ),
                            html.Button("Send", id="send-button", n_clicks=0,
                                       style={'width': '18%', 'padding': '0.75rem', 'background': '#667eea', 
                                             'color': 'white', 'border': 'none', 'border-radius': '4px', 'margin-left': '2%'})
                        ]),
                        
                        # Sample questions
                        html.Div([
                            html.P("💡 Try asking:", style={'font-weight': 'bold', 'margin-bottom': '0.5rem'}),
                            html.Div([
                                html.Button("What's our revenue trend?", className="sample-question", 
                                           style={'margin': '0.25rem', 'padding': '0.5rem', 'background': '#f8f9fa', 
                                                 'border': '1px solid #ddd', 'border-radius': '4px', 'cursor': 'pointer'}),
                                html.Button("Which products perform best?", className="sample-question",
                                           style={'margin': '0.25rem', 'padding': '0.5rem', 'background': '#f8f9fa', 
                                                 'border': '1px solid #ddd', 'border-radius': '4px', 'cursor': 'pointer'}),
                                html.Button("Show customer segments", className="sample-question",
                                           style={'margin': '0.25rem', 'padding': '0.5rem', 'background': '#f8f9fa', 
                                                 'border': '1px solid #ddd', 'border-radius': '4px', 'cursor': 'pointer'}),
                            ])
                        ], style={'margin-top': '1rem'})
                    ], className="chat-container")
                ], className="tab-content")
            ]),
            
            # Insights Tab
            dcc.Tab(label="🔍 AI Insights", value="insights", children=[
                html.Div([
                    html.H3("🧠 AI-Generated Business Insights"),
                    
                    html.Div([
                        html.Div([
                            html.H4("📈 Revenue Analysis"),
                            html.P("• Revenue shows a positive trend with 12% growth month-over-month"),
                            html.P("• Peak performance observed during Q4 holiday season"),
                            html.P("• Recommendation: Increase marketing spend during high-conversion periods")
                        ], className="metric-card"),
                        
                        html.Div([
                            html.H4("👥 Customer Insights"),
                            html.P("• Premium segment shows highest retention (85%) and value"),
                            html.P("• Basic segment has growth potential with targeted campaigns"),
                            html.P("• Recommendation: Focus on Premium segment expansion")
                        ], className="metric-card"),
                        
                        html.Div([
                            html.H4("🛍️ Product Performance"),
                            html.P("• Product A leads in sales volume and market share"),
                            html.P("• Product D shows highest profit margins (28%)"),
                            html.P("• Recommendation: Increase inventory for top performers")
                        ], className="metric-card"),
                        
                        html.Div([
                            html.H4("⚠️ Risk Alerts"),
                            html.P("• Conversion rate declining in Basic segment (-2%)"),
                            html.P("• Inventory levels low for Product B"),
                            html.P("• Recommendation: Review pricing strategy and restock inventory")
                        ], className="metric-card"),
                    ], style={'display': 'grid', 'grid-template-columns': 'repeat(2, 1fr)', 'gap': '1rem'})
                ], className="tab-content")
            ]),
            
            # Data Explorer Tab
            dcc.Tab(label="📊 Data Explorer", value="explorer", children=[
                html.Div([
                    html.H3("🔍 Interactive Data Exploration"),
                    
                    # Filters
                    html.Div([
                        html.Div([
                            html.Label("Date Range:"),
                            dcc.DatePickerRange(
                                id="date-picker",
                                start_date=sales_data['date'].min(),
                                end_date=sales_data['date'].max(),
                                display_format='YYYY-MM-DD'
                            )
                        ], style={'width': '30%', 'display': 'inline-block', 'margin-right': '2%'}),
                        
                        html.Div([
                            html.Label("Metric:"),
                            dcc.Dropdown(
                                id="metric-dropdown",
                                options=[
                                    {'label': 'Revenue', 'value': 'revenue'},
                                    {'label': 'Sales', 'value': 'sales'},
                                    {'label': 'Customers', 'value': 'customers'},
                                    {'label': 'Conversion Rate', 'value': 'conversion_rate'}
                                ],
                                value='revenue'
                            )
                        ], style={'width': '30%', 'display': 'inline-block', 'margin-right': '2%'}),
                        
                        html.Div([
                            html.Label("Chart Type:"),
                            dcc.Dropdown(
                                id="chart-type",
                                options=[
                                    {'label': 'Line Chart', 'value': 'line'},
                                    {'label': 'Bar Chart', 'value': 'bar'},
                                    {'label': 'Scatter Plot', 'value': 'scatter'}
                                ],
                                value='line'
                            )
                        ], style={'width': '30%', 'display': 'inline-block'}),
                    ], style={'margin-bottom': '2rem'}),
                    
                    # Dynamic chart
                    dcc.Graph(id="dynamic-chart"),
                    
                    # Data table
                    html.Div([
                        html.H4("📋 Raw Data"),
                        dash_table.DataTable(
                            id="data-table",
                            columns=[{"name": i, "id": i} for i in sales_data.columns],
                            data=sales_data.to_dict('records'),
                            page_size=10,
                            style_cell={'textAlign': 'left', 'fontFamily': 'Segoe UI'},
                            style_header={'backgroundColor': '#667eea', 'color': 'white', 'fontWeight': 'bold'},
                            style_data={'backgroundColor': 'white', 'border': '1px solid #ddd'}
                        )
                    ])
                ], className="tab-content")
            ])
        ])
    ], style={'max-width': '1200px', 'margin': '0 auto', 'padding': '0 1rem'})
])

# Callbacks
@callback(
    Output("chat-messages", "children"),
    [Input("send-button", "n_clicks"), Input("chat-input", "n_submit")],
    [State("chat-input", "value"), State("chat-messages", "children")]
)
def update_chat(n_clicks, n_submit, input_value, messages):
    """Handle chat interactions"""
    if (n_clicks > 0 or n_submit) and input_value:
        # Add user message
        user_message = html.Div([
            html.P(input_value, style={'margin': '0'})
        ], className="chat-message user-message")
        
        # Generate AI response
        ai_response = ai_engine.generate_response(input_value)
        ai_message = html.Div([
            html.P(ai_response, style={'margin': '0'})
        ], className="chat-message ai-message")
        
        messages.extend([user_message, ai_message])
        
        return messages
    return messages

@callback(
    Output("dynamic-chart", "figure"),
    [Input("date-picker", "start_date"), Input("date-picker", "end_date"),
     Input("metric-dropdown", "value"), Input("chart-type", "value")]
)
def update_chart(start_date, end_date, metric, chart_type):
    """Update chart based on filters"""
    try:
        filtered_data = sales_data.copy()
        
        if start_date and end_date:
            filtered_data = filtered_data[
                (filtered_data['date'] >= start_date) & 
                (filtered_data['date'] <= end_date)
            ]
        
        if chart_type == 'line':
            fig = px.line(filtered_data, x='date', y=metric, title=f'{metric.title()} Over Time')
        elif chart_type == 'bar':
            fig = px.bar(filtered_data, x='date', y=metric, title=f'{metric.title()} Over Time')
        else:  # scatter
            fig = px.scatter(filtered_data, x='date', y=metric, title=f'{metric.title()} Over Time')
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Segoe UI", size=12)
        )
        
        return fig
    except Exception as e:
        # Return empty figure on error
        fig = go.Figure()
        fig.update_layout(
            title="Error loading chart",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
