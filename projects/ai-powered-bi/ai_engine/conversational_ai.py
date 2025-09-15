"""
AI-Powered Business Intelligence Engine
Revolutionary conversational AI for business intelligence
"""

import openai
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
import asyncio
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

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

class AIPoweredBIEngine:
    """
    Revolutionary AI engine for business intelligence
    Handles natural language queries and generates intelligent insights
    """
    
    def __init__(self, api_key: str, model: str = "gpt-5"):
        """
        Initialize the AI BI Engine
        
        Args:
            api_key: OpenAI API key
            model: Model to use for AI processing
        """
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)
        
        # Initialize data sources
        self.data_sources = {}
        self.insights_history = []
        self.user_context = {}
        
        # AI prompts for different query types
        self.prompts = {
            QueryType.DESCRIPTIVE: """
            You are an expert business analyst. Analyze the following data and provide a clear, 
            executive-level summary of what happened. Focus on key metrics, trends, and patterns.
            Use business language that executives can understand.
            """,
            
            QueryType.DIAGNOSTIC: """
            You are a senior business analyst. Analyze the data to explain WHY something happened.
            Look for root causes, correlations, and contributing factors. Provide actionable insights
            about what drove the results.
            """,
            
            QueryType.PREDICTIVE: """
            You are a data scientist and business forecaster. Based on the historical data,
            predict what will happen in the future. Include confidence levels, assumptions,
            and potential scenarios. Be specific about timeframes and metrics.
            """,
            
            QueryType.PRESCRIPTIVE: """
            You are a strategic business consultant. Based on the analysis, provide specific,
            actionable recommendations for what the business should do. Include priorities,
            expected outcomes, and implementation considerations.
            """
        }
    
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
        Classify the type of query using AI
        
        Args:
            query: Natural language query
            
        Returns:
            QueryType enum
        """
        classification_prompt = f"""
        Classify this business query into one of these categories:
        1. DESCRIPTIVE: "What happened?" - asking about past events, trends, current state
        2. DIAGNOSTIC: "Why did it happen?" - asking for explanations, root causes
        3. PREDICTIVE: "What will happen?" - asking for forecasts, predictions
        4. PRESCRIPTIVE: "What should we do?" - asking for recommendations, actions
        
        Query: "{query}"
        
        Respond with only the category name (DESCRIPTIVE, DIAGNOSTIC, PREDICTIVE, or PRESCRIPTIVE).
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": classification_prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            classification = response.choices[0].message.content.strip().upper()
            return QueryType(classification)
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
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
        Generate AI response based on analysis
        
        Args:
            query: Original user query
            analysis_results: Results from data analysis
            query_type: Type of query
            
        Returns:
            AI-generated response
        """
        # Prepare context for AI
        context = self._prepare_context(query, analysis_results, query_type)
        
        # Get appropriate prompt
        base_prompt = self.prompts[query_type]
        
        full_prompt = f"""
        {base_prompt}
        
        User Query: "{query}"
        
        Data Analysis Results:
        {json.dumps(analysis_results, indent=2, default=str)}
        
        Context:
        {context}
        
        Provide a comprehensive, executive-level response that directly answers the user's question.
        Include specific data points, insights, and actionable recommendations where appropriate.
        Use clear, professional business language.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return "I apologize, but I encountered an error processing your query. Please try again."
    
    def generate_recommendations(self, query: str, analysis_results: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations
        
        Args:
            query: Original user query
            analysis_results: Results from data analysis
            
        Returns:
            List of recommendations
        """
        recommendations_prompt = f"""
        Based on the following analysis, provide 3-5 specific, actionable recommendations:
        
        Query: "{query}"
        Analysis: {json.dumps(analysis_results, indent=2, default=str)}
        
        Each recommendation should:
        1. Be specific and actionable
        2. Include expected impact
        3. Be prioritized by importance
        4. Include implementation considerations
        
        Format as a numbered list.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": recommendations_prompt}],
                max_tokens=500,
                temperature=0.4
            )
            
            recommendations_text = response.choices[0].message.content.strip()
            # Parse recommendations into list
            recommendations = [rec.strip() for rec in recommendations_text.split('\n') if rec.strip()]
            return recommendations
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations at this time."]
    
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
                answer=f"I encountered an error processing your query: {str(e)}",
                confidence=0.0,
                data_sources=[],
                visualizations=[],
                recommendations=[],
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _find_relevant_sources(self, query: str) -> List[str]:
        """Find data sources relevant to the query"""
        relevant_sources = []
        
        # Simple keyword matching (can be enhanced with embeddings)
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
        
        # Add trend analysis
        if 'date' in data.columns or 'time' in data.columns:
            date_col = 'date' if 'date' in data.columns else 'time'
            analysis['trends'] = self._analyze_trends(data, date_col)
        
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
                # Add chart specifications
                visualizations.append({
                    'type': 'bar_chart',
                    'title': f'{source_name} - Key Metrics',
                    'data_source': source_name,
                    'description': 'Summary statistics visualization'
                })
            
            if 'correlations' in analysis:
                visualizations.append({
                    'type': 'heatmap',
                    'title': f'{source_name} - Correlations',
                    'data_source': source_name,
                    'description': 'Correlation matrix heatmap'
                })
        
        return visualizations
    
    def _analyze_trends(self, data: pd.DataFrame, date_col: str) -> Dict[str, Any]:
        """Analyze trends in time series data"""
        try:
            data_sorted = data.sort_values(date_col)
            numeric_cols = data_sorted.select_dtypes(include=[np.number]).columns
            
            trends = {}
            for col in numeric_cols:
                if len(data_sorted) > 1:
                    x = np.arange(len(data_sorted))
                    y = data_sorted[col].values
                    slope = np.polyfit(x, y, 1)[0]
                    trends[col] = {
                        'slope': slope,
                        'direction': 'increasing' if slope > 0 else 'decreasing',
                        'strength': abs(slope)
                    }
            
            return trends
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {}
    
    def _prepare_context(self, query: str, analysis_results: Dict[str, Any], query_type: QueryType) -> str:
        """Prepare context for AI response generation"""
        context_parts = [
            f"Query Type: {query_type.value}",
            f"Data Sources Available: {len(self.data_sources)}",
            f"Analysis Timestamp: {datetime.now().isoformat()}"
        ]
        
        if self.insights_history:
            recent_insights = self.insights_history[-3:]  # Last 3 insights
            context_parts.append("Recent Insights:")
            for insight in recent_insights:
                context_parts.append(f"- {insight.query}: {insight.query_type.value}")
        
        return "\n".join(context_parts)
    
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

# Example usage and testing
if __name__ == "__main__":
    # Initialize the AI BI Engine
    engine = AIPoweredBIEngine(api_key="your-api-key-here")
    
    # Add sample data
    sample_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100),
        'sales': np.random.normal(1000, 200, 100),
        'customers': np.random.normal(500, 50, 100),
        'revenue': np.random.normal(50000, 10000, 100)
    })
    
    engine.add_data_source(
        name="sales_data",
        data=sample_data,
        description="Daily sales, customer, and revenue data"
    )
    
    # Example queries
    test_queries = [
        "What were our sales trends last quarter?",
        "Why did revenue drop in March?",
        "What will our sales be next month?",
        "What should we do to increase customer satisfaction?"
    ]
    
    print("🤖 AI-Powered Business Intelligence Engine")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        result = engine.process_query(query)
        print(f"🎯 Type: {result.query_type.value}")
        print(f"📊 Answer: {result.answer}")
        print(f"🎯 Confidence: {result.confidence:.2f}")
        print(f"⏱️ Execution Time: {result.execution_time:.2f}s")
        print("-" * 50)
