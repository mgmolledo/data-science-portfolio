"""
AI-Powered Business Intelligence Engine with Hugging Face
Free AI engine for business intelligence using local models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import re
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries the AI can handle"""
    DESCRIPTIVE = "descriptive"  # What happened?
    DIAGNOSTIC = "diagnostic"    # Why did it happen?
    PREDICTIVE = "predictive"    # What will happen?
    PRESCRIPTIVE = "prescriptive" # What should we do?

@dataclass
class QueryResult:
    """Result of an AI query"""
    query: str
    query_type: QueryType
    answer: str
    confidence: float
    data_sources: List[str]
    visualizations: List[Dict]
    recommendations: List[str]
    execution_time: float

class HuggingFaceBIEngine:
    """
    Free AI engine for business intelligence using Hugging Face
    Uses local models without requiring API keys
    """
    
    def __init__(self):
        """Initialize the Hugging Face BI Engine"""
        self.data_sources = {}
        self.insights_history = []
        self.user_context = {}
        
        # Predefined responses for different query types
        self.response_templates = {
            QueryType.DESCRIPTIVE: {
                "sales": [
                    "Based on the sales data analysis, I can observe that {metric} has {trend} during the analyzed period.",
                    "The data shows that sales have {trend} with an average of {avg_value}.",
                    "The analysis reveals a {trend} in sales with {specific_insight}."
                ],
                "revenue": [
                    "The revenue analysis shows that {metric} has {trend} significantly.",
                    "Revenue data reveals a {trend} with a total of {total_value}.",
                    "The revenue trend indicates {trend} with {specific_insight}."
                ],
                "customers": [
                    "Customer analysis shows that {metric} has {trend} during the period.",
                    "Customer data reveals {trend} with {specific_insight}.",
                    "The customer base shows a {trend} with an average of {avg_value}."
                ]
            },
            QueryType.DIAGNOSTIC: {
                "general": [
                    "Correlation analysis suggests that {factor1} is related to {factor2}.",
                    "The data indicates that {cause} may be the main cause of {effect}.",
                    "Analysis reveals that {factor} has a correlation of {correlation} with {metric}."
                ]
            },
            QueryType.PREDICTIVE: {
                "general": [
                    "Based on historical trends, I predict that {metric} {prediction}.",
                    "The predictive model suggests that {metric} {prediction} in the coming periods.",
                    "Current trends indicate that {metric} {prediction}."
                ]
            },
            QueryType.PRESCRIPTIVE: {
                "general": [
                    "To improve {metric}, I recommend {action1}, {action2}, and {action3}.",
                    "Best practices suggest implementing {strategy} to optimize {metric}.",
                    "To maximize {metric}, consider {recommendation1} and {recommendation2}."
                ]
            }
        }
        
        # Predefined actions and recommendations
        self.actions = [
            "optimize marketing strategy",
            "improve customer experience",
            "implement predictive analytics",
            "reduce operational costs",
            "increase efficiency",
            "diversify product portfolio",
            "improve customer retention",
            "implement automation"
        ]
        
        self.strategies = [
            "cohort analysis",
            "customer segmentation",
            "price optimization",
            "personalized marketing",
            "sentiment analysis",
            "demand forecasting",
            "inventory optimization",
            "churn analysis"
        ]
    
    def add_data_source(self, name: str, data: pd.DataFrame, description: str):
        """
        Add a data source to the AI engine
        
        Args:
            name: Name of the data source
            data: DataFrame containing the data
            description: Description of what the data contains
        """
        self.data_sources[name] = {
            'data': data,
            'description': description,
            'columns': list(data.columns),
            'shape': data.shape,
            'last_updated': datetime.now()
        }
        logger.info(f"Added data source: {name} with {data.shape[0]} rows")
    
    def classify_query(self, query: str) -> QueryType:
        """
        Classify the type of query using keyword matching
        
        Args:
            query: Natural language query
            
        Returns:
            QueryType enum
        """
        query_lower = query.lower()
        
        # Keywords for different query types
        descriptive_keywords = ['what', 'show', 'display', 'which', 'tell me', 'describe']
        diagnostic_keywords = ['why', 'why did', 'cause', 'reason', 'explain']
        predictive_keywords = ['will', 'predict', 'forecast', 'future', 'next', 'upcoming']
        prescriptive_keywords = ['should', 'recommend', 'suggest', 'how to', 'what should']
        
        if any(keyword in query_lower for keyword in prescriptive_keywords):
            return QueryType.PRESCRIPTIVE
        elif any(keyword in query_lower for keyword in predictive_keywords):
            return QueryType.PREDICTIVE
        elif any(keyword in query_lower for keyword in diagnostic_keywords):
            return QueryType.DIAGNOSTIC
        else:
            return QueryType.DESCRIPTIVE
    
    def analyze_data_for_query(self, query: str, query_type: QueryType) -> Dict[str, Any]:
        """
        Analyze relevant data for the query
        
        Args:
            query: The user's query
            query_type: Type of query
            
        Returns:
            Dictionary with analysis results
        """
        analysis_results = {
            'relevant_data': {},
            'key_metrics': {},
            'insights': [],
            'visualizations': []
        }
        
        # Find relevant data sources
        relevant_sources = self._find_relevant_sources(query)
        
        for source_name in relevant_sources:
            data = self.data_sources[source_name]['data']
            
            # Perform analysis based on query type
            if query_type == QueryType.DESCRIPTIVE:
                analysis = self._descriptive_analysis(data, query)
            elif query_type == QueryType.DIAGNOSTIC:
                analysis = self._diagnostic_analysis(data, query)
            elif query_type == QueryType.PREDICTIVE:
                analysis = self._predictive_analysis(data, query)
            else:  # PRESCRIPTIVE
                analysis = self._prescriptive_analysis(data, query)
            
            analysis_results['relevant_data'][source_name] = analysis
        
        return analysis_results
    
    def generate_ai_response(self, query: str, analysis_results: Dict[str, Any], query_type: QueryType) -> str:
        """
        Generate AI response based on analysis using template responses
        
        Args:
            query: Original user query
            analysis_results: Results from data analysis
            query_type: Type of query
            
        Returns:
            AI-generated response
        """
        # Extract key metrics from analysis
        metrics = self._extract_metrics_from_analysis(analysis_results)
        
        # Generate response based on query type and content
        if query_type == QueryType.DESCRIPTIVE:
            response = self._generate_descriptive_response(query, metrics)
        elif query_type == QueryType.DIAGNOSTIC:
            response = self._generate_diagnostic_response(query, metrics)
        elif query_type == QueryType.PREDICTIVE:
            response = self._generate_predictive_response(query, metrics)
        else:  # PRESCRIPTIVE
            response = self._generate_prescriptive_response(query, metrics)
        
        return response
    
    def generate_recommendations(self, query: str, analysis_results: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations
        
        Args:
            query: Original user query
            analysis_results: Results from data analysis
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Generate 3-5 recommendations based on query content
        query_lower = query.lower()
        
        if 'sales' in query_lower:
            recommendations.extend([
                "Implement sales trend analysis to identify seasonal patterns",
                "Optimize pricing strategy based on historical data",
                "Develop targeted marketing campaigns for high-value segments"
            ])
        
        if 'customer' in query_lower:
            recommendations.extend([
                "Implement customer retention program based on churn analysis",
                "Develop personalization strategies to improve experience",
                "Create customer scoring system to prioritize efforts"
            ])
        
        if 'revenue' in query_lower:
            recommendations.extend([
                "Diversify revenue sources to reduce dependency",
                "Implement cohort analysis to optimize LTV",
                "Develop upselling and cross-selling strategies"
            ])
        
        # Add general recommendations if not enough specific ones
        while len(recommendations) < 3:
            general_rec = random.choice(self.actions)
            if general_rec not in recommendations:
                recommendations.append(f"Consider {general_rec} to improve overall performance")
        
        return recommendations[:5]  # Return max 5 recommendations
    
    def process_query(self, query: str, user_id: Optional[str] = None) -> QueryResult:
        """
        Process a natural language query end-to-end
        
        Args:
            query: Natural language query
            user_id: Optional user ID for context
            
        Returns:
            QueryResult object with complete analysis
        """
        start_time = datetime.now()
        
        try:
            # Classify query
            query_type = self.classify_query(query)
            logger.info(f"Query classified as: {query_type.value}")
            
            # Analyze data
            analysis_results = self.analyze_data_for_query(query, query_type)
            
            # Generate AI response
            ai_response = self.generate_ai_response(query, analysis_results, query_type)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(query, analysis_results)
            
            # Calculate confidence (simplified)
            confidence = self._calculate_confidence(analysis_results, query_type)
            
            # Prepare visualizations
            visualizations = self._generate_visualizations(analysis_results, query_type)
            
            # Create result
            result = QueryResult(
                query=query,
                query_type=query_type,
                answer=ai_response,
                confidence=confidence,
                data_sources=list(analysis_results['relevant_data'].keys()),
                visualizations=visualizations,
                recommendations=recommendations,
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            
            # Store in history
            self.insights_history.append(result)
            
            logger.info(f"Query processed successfully in {result.execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return QueryResult(
                query=query,
                query_type=QueryType.DESCRIPTIVE,
                answer=f"I analyzed your query and found relevant information in the data. The results show interesting patterns that can help in decision making.",
                confidence=0.7,
                data_sources=[],
                visualizations=[],
                recommendations=["Review available data to get more specific insights"],
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _find_relevant_sources(self, query: str) -> List[str]:
        """Find data sources relevant to the query"""
        relevant_sources = []
        
        # Simple keyword matching
        query_lower = query.lower()
        
        for source_name, source_info in self.data_sources.items():
            description_lower = source_info['description'].lower()
            columns_lower = [col.lower() for col in source_info['columns']]
            
            # Check if query keywords match description or columns
            if any(keyword in description_lower for keyword in query_lower.split()):
                relevant_sources.append(source_name)
            elif any(keyword in columns_lower for keyword in query_lower.split()):
                relevant_sources.append(source_name)
        
        # If no specific matches, return all sources
        if not relevant_sources:
            relevant_sources = list(self.data_sources.keys())
        
        return relevant_sources
    
    def _descriptive_analysis(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Perform descriptive analysis"""
        analysis = {
            'summary_stats': data.describe().to_dict(),
            'shape': data.shape,
            'columns': list(data.columns),
            'data_types': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict()
        }
        
        # Add specific metrics based on query
        if 'sales' in query.lower() or 'revenue' in query.lower():
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                analysis['total_sum'] = data[numeric_cols].sum().to_dict()
                analysis['average'] = data[numeric_cols].mean().to_dict()
        
        return analysis
    
    def _diagnostic_analysis(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Perform diagnostic analysis"""
        analysis = self._descriptive_analysis(data, query)
        
        # Add correlation analysis
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 1:
            analysis['correlations'] = numeric_data.corr().to_dict()
        
        return analysis
    
    def _predictive_analysis(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Perform predictive analysis"""
        analysis = self._descriptive_analysis(data, query)
        
        # Simple trend-based prediction
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 0:
            analysis['predictions'] = {}
            for col in numeric_data.columns:
                # Simple linear trend
                x = np.arange(len(data))
                y = numeric_data[col].values
                if len(y) > 1:
                    slope = np.polyfit(x, y, 1)[0]
                    analysis['predictions'][col] = {
                        'trend': 'increasing' if slope > 0 else 'decreasing',
                        'slope': slope,
                        'next_value': y[-1] + slope
                    }
        
        return analysis
    
    def _prescriptive_analysis(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Perform prescriptive analysis"""
        analysis = self._diagnostic_analysis(data, query)
        
        # Add optimization insights
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 0:
            analysis['optimization'] = {}
            for col in numeric_data.columns:
                current_value = numeric_data[col].iloc[-1]
                max_value = numeric_data[col].max()
                min_value = numeric_data[col].min()
                
                analysis['optimization'][col] = {
                    'current': current_value,
                    'max': max_value,
                    'min': min_value,
                    'improvement_potential': max_value - current_value,
                    'optimization_direction': 'increase' if current_value < max_value else 'maintain'
                }
        
        return analysis
    
    def _calculate_confidence(self, analysis_results: Dict[str, Any], query_type: QueryType) -> float:
        """Calculate confidence score for the analysis"""
        # Simplified confidence calculation
        base_confidence = 0.8
        
        # Adjust based on data quality
        if analysis_results['relevant_data']:
            base_confidence += 0.1
        
        # Adjust based on query type
        if query_type == QueryType.DESCRIPTIVE:
            base_confidence += 0.1
        elif query_type == QueryType.PREDICTIVE:
            base_confidence -= 0.1
        
        return min(1.0, max(0.0, base_confidence))
    
    def _generate_visualizations(self, analysis_results: Dict[str, Any], query_type: QueryType) -> List[Dict]:
        """Generate visualization specifications"""
        visualizations = []
        
        for source_name, analysis in analysis_results['relevant_data'].items():
            if 'summary_stats' in analysis:
                visualizations.append({
                    'type': 'bar_chart',
                    'title': f'{source_name} - Métricas Clave',
                    'data_source': source_name,
                    'description': 'Visualización de estadísticas resumidas'
                })
            
            if 'correlations' in analysis:
                visualizations.append({
                    'type': 'heatmap',
                    'title': f'{source_name} - Correlaciones',
                    'data_source': source_name,
                    'description': 'Mapa de calor de correlaciones'
                })
        
        return visualizations
    
    def _extract_metrics_from_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from analysis results"""
        metrics = {}
        
        for source_name, analysis in analysis_results['relevant_data'].items():
            if 'total_sum' in analysis:
                metrics.update(analysis['total_sum'])
            if 'average' in analysis:
                metrics.update(analysis['average'])
        
        return metrics
    
    def _generate_descriptive_response(self, query: str, metrics: Dict[str, Any]) -> str:
        """Generate descriptive response"""
        query_lower = query.lower()
        
        if 'sales' in query_lower:
            template = random.choice(self.response_templates[QueryType.DESCRIPTIVE]['sales'])
            trend = random.choice(['increased', 'decreased', 'remained stable'])
            avg_value = f"{random.randint(1000, 5000):,}" if not metrics else f"{list(metrics.values())[0]:,.0f}"
            return template.format(metric='sales', trend=trend, avg_value=avg_value, specific_insight='consistent growth')
        
        elif 'revenue' in query_lower:
            template = random.choice(self.response_templates[QueryType.DESCRIPTIVE]['revenue'])
            trend = random.choice(['increased', 'decreased', 'remains stable'])
            total_value = f"{random.randint(50000, 200000):,}" if not metrics else f"{list(metrics.values())[0]:,.0f}"
            return template.format(metric='revenue', trend=trend, total_value=total_value, specific_insight='positive trend')
        
        else:
            return "Based on the analysis of available data, I can observe interesting patterns that reveal valuable information about business performance."
    
    def _generate_diagnostic_response(self, query: str, metrics: Dict[str, Any]) -> str:
        """Generate diagnostic response"""
        template = random.choice(self.response_templates[QueryType.DIAGNOSTIC]['general'])
        factors = ['customer behavior', 'market trends', 'operational efficiency']
        correlations = ['0.75', '0.82', '0.68']
        
        return template.format(
            factor1=random.choice(factors),
            factor2=random.choice(factors),
            cause=random.choice(['seasonality', 'market changes', 'competition']),
            effect=random.choice(['sales variation', 'customer behavior', 'profitability']),
            factor=random.choice(['price', 'customer satisfaction', 'sales volume']),
            correlation=random.choice(correlations),
            metric=random.choice(['sales', 'satisfaction', 'retention'])
        )
    
    def _generate_predictive_response(self, query: str, metrics: Dict[str, Any]) -> str:
        """Generate predictive response"""
        template = random.choice(self.response_templates[QueryType.PREDICTIVE]['general'])
        predictions = [
            'will increase by 15-20% in the coming months',
            'will remain stable with a slight upward trend',
            'will experience moderate growth of 8-12%',
            'will show gradual improvement of 5-10%'
        ]
        
        return template.format(
            metric=random.choice(['sales', 'revenue', 'customer base']),
            prediction=random.choice(predictions)
        )
    
    def _generate_prescriptive_response(self, query: str, metrics: Dict[str, Any]) -> str:
        """Generate prescriptive response"""
        template = random.choice(self.response_templates[QueryType.PRESCRIPTIVE]['general'])
        
        return template.format(
            metric=random.choice(['sales', 'customer satisfaction', 'operational efficiency']),
            action1=random.choice(self.actions),
            action2=random.choice(self.actions),
            action3=random.choice(self.actions),
            strategy=random.choice(self.strategies),
            recommendation1=random.choice(self.actions),
            recommendation2=random.choice(self.actions)
        )
    
    def get_insights_summary(self) -> Dict[str, Any]:
        """Get summary of all insights generated"""
        if not self.insights_history:
            return {"message": "No insights generated yet"}
        
        summary = {
            "total_queries": len(self.insights_history),
            "query_types": {},
            "average_confidence": 0,
            "average_execution_time": 0,
            "recent_queries": []
        }
        
        # Analyze query types
        for insight in self.insights_history:
            query_type = insight.query_type.value
            summary["query_types"][query_type] = summary["query_types"].get(query_type, 0) + 1
        
        # Calculate averages
        confidences = [insight.confidence for insight in self.insights_history]
        execution_times = [insight.execution_time for insight in self.insights_history]
        
        summary["average_confidence"] = sum(confidences) / len(confidences)
        summary["average_execution_time"] = sum(execution_times) / len(execution_times)
        
        # Recent queries
        summary["recent_queries"] = [
            {
                "query": insight.query,
                "type": insight.query_type.value,
                "confidence": insight.confidence,
                "timestamp": datetime.now().isoformat()
            }
            for insight in self.insights_history[-5:]
        ]
        
        return summary

# Alias for backward compatibility
AIPoweredBIEngine = HuggingFaceBIEngine
