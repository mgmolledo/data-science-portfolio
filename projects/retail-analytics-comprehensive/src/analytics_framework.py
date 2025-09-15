"""
Professional Data Analytics Notebook Template
This template ensures consistent, comprehensive analysis across all projects
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Professional styling configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

class ProfessionalAnalyzer:
    """
    Professional data analysis framework
    Ensures consistent, comprehensive analysis across all projects
    """
    
    def __init__(self, project_name, business_context):
        self.project_name = project_name
        self.business_context = business_context
        self.data_quality_report = {}
        self.insights = []
        self.recommendations = []
        
    def print_header(self, title, subtitle=""):
        """Print professional section header"""
        print("=" * 80)
        print(f"📊 {title}")
        if subtitle:
            print(f"   {subtitle}")
        print("=" * 80)
    
    def data_quality_assessment(self, df, dataset_name):
        """Comprehensive data quality assessment"""
        self.print_header(f"Data Quality Assessment - {dataset_name}")
        
        # Basic information
        print(f"📋 Dataset Overview:")
        print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        print(f"   Data types: {df.dtypes.value_counts().to_dict()}")
        
        # Missing values analysis
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        print(f"\n🔍 Missing Values Analysis:")
        if missing_data.sum() > 0:
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing_Count': missing_data.values,
                'Missing_Percentage': missing_percent.values
            }).sort_values('Missing_Percentage', ascending=False)
            
            print(missing_df[missing_df['Missing_Count'] > 0].to_string(index=False))
        else:
            print("   ✅ No missing values found")
        
        # Duplicate analysis
        duplicates = df.duplicated().sum()
        print(f"\n🔄 Duplicate Analysis:")
        print(f"   Duplicate rows: {duplicates:,} ({duplicates/len(df)*100:.1f}%)")
        
        # Data quality score
        quality_score = self._calculate_quality_score(df)
        print(f"\n📊 Data Quality Score: {quality_score:.1f}/100")
        
        # Store quality report
        self.data_quality_report[dataset_name] = {
            'shape': df.shape,
            'missing_values': missing_data.sum(),
            'duplicates': duplicates,
            'quality_score': quality_score
        }
        
        return quality_score
    
    def _calculate_quality_score(self, df):
        """Calculate overall data quality score"""
        score = 100
        
        # Deduct for missing values
        missing_percent = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        score -= missing_percent * 2
        
        # Deduct for duplicates
        duplicate_percent = (df.duplicated().sum() / len(df)) * 100
        score -= duplicate_percent * 1
        
        # Deduct for inconsistent data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < df.shape[1] * 0.3:  # Less than 30% numeric
            score -= 10
        
        return max(0, score)
    
    def statistical_summary(self, df, dataset_name):
        """Comprehensive statistical summary"""
        self.print_header(f"Statistical Summary - {dataset_name}")
        
        # Descriptive statistics
        print("📈 Descriptive Statistics:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            stats = df[numeric_cols].describe()
            print(stats.round(2))
            
            # Additional statistics
            print(f"\n📊 Additional Statistics:")
            for col in numeric_cols[:5]:  # Show first 5 columns
                print(f"   {col}:")
                print(f"     Skewness: {df[col].skew():.3f}")
                print(f"     Kurtosis: {df[col].kurtosis():.3f}")
                print(f"     Coefficient of Variation: {df[col].std()/df[col].mean():.3f}")
        
        # Categorical analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            print(f"\n📋 Categorical Analysis:")
            for col in categorical_cols[:3]:  # Show first 3 columns
                print(f"   {col}:")
                print(f"     Unique values: {df[col].nunique()}")
                print(f"     Most frequent: {df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'}")
                print(f"     Frequency: {df[col].value_counts().iloc[0] if not df[col].empty else 0}")
    
    def create_professional_visualizations(self, df, dataset_name):
        """Create comprehensive visualization suite"""
        self.print_header(f"Professional Visualizations - {dataset_name}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # 1. Distribution analysis
        if len(numeric_cols) > 0:
            self._create_distribution_plots(df, numeric_cols[:4])  # First 4 numeric columns
        
        # 2. Correlation analysis
        if len(numeric_cols) > 1:
            self._create_correlation_analysis(df, numeric_cols)
        
        # 3. Categorical analysis
        if len(categorical_cols) > 0:
            self._create_categorical_plots(df, categorical_cols[:3])  # First 3 categorical columns
        
        # 4. Time series analysis (if date column exists)
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            self._create_time_series_plots(df, date_cols[0], numeric_cols[0] if len(numeric_cols) > 0 else None)
    
    def _create_distribution_plots(self, df, numeric_cols):
        """Create distribution plots for numeric columns"""
        n_cols = min(2, len(numeric_cols))
        n_rows = (len(numeric_cols) + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                # Histogram with KDE
                axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
                
                # Add KDE
                from scipy import stats
                kde = stats.gaussian_kde(df[col].dropna())
                x_range = np.linspace(df[col].min(), df[col].max(), 100)
                axes[i].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
                
                axes[i].set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Density')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def _create_correlation_analysis(self, df, numeric_cols):
        """Create correlation analysis"""
        # Correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Scatter plot matrix for top correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # High correlation threshold
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if high_corr_pairs:
            print(f"\n🔗 High Correlation Pairs (|r| > 0.5):")
            for col1, col2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]:
                print(f"   {col1} ↔ {col2}: {corr:.3f}")
    
    def _create_categorical_plots(self, df, categorical_cols):
        """Create categorical analysis plots"""
        for col in categorical_cols:
            # Value counts
            value_counts = df[col].value_counts().head(10)
            
            # Bar plot
            plt.figure(figsize=(12, 6))
            value_counts.plot(kind='bar', color='lightcoral', edgecolor='black')
            plt.title(f'Top 10 Values in {col}', fontsize=14, fontweight='bold')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            print(f"\n📊 {col} Analysis:")
            print(f"   Unique values: {df[col].nunique()}")
            print(f"   Most frequent: {value_counts.index[0]} ({value_counts.iloc[0]} occurrences)")
    
    def _create_time_series_plots(self, df, date_col, value_col):
        """Create time series analysis plots"""
        if value_col is None:
            return
        
        # Group by date and aggregate
        time_series = df.groupby(df[date_col].dt.date)[value_col].sum().reset_index()
        time_series[date_col] = pd.to_datetime(time_series[date_col])
        
        # Time series plot
        plt.figure(figsize=(15, 6))
        plt.plot(time_series[date_col], time_series[value_col], linewidth=2, color='darkblue')
        plt.title(f'{value_col} Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel(value_col)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Seasonal decomposition (if enough data)
        if len(time_series) > 30:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Set date as index
            ts_data = time_series.set_index(date_col)[value_col]
            
            # Decomposition
            decomposition = seasonal_decompose(ts_data, model='additive', period=30)
            
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            decomposition.observed.plot(ax=axes[0], title='Original')
            decomposition.trend.plot(ax=axes[1], title='Trend')
            decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
            decomposition.resid.plot(ax=axes[3], title='Residual')
            
            plt.tight_layout()
            plt.show()
    
    def generate_business_insights(self, df, dataset_name):
        """Generate business-focused insights"""
        self.print_header(f"Business Insights - {dataset_name}")
        
        insights = []
        
        # Data quality insights
        quality_score = self.data_quality_report.get(dataset_name, {}).get('quality_score', 0)
        if quality_score >= 90:
            insights.append("✅ Excellent data quality - ready for advanced analytics")
        elif quality_score >= 70:
            insights.append("⚠️ Good data quality - minor cleaning recommended")
        else:
            insights.append("❌ Data quality issues detected - cleaning required")
        
        # Statistical insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:3]:
                mean_val = df[col].mean()
                std_val = df[col].std()
                cv = std_val / mean_val if mean_val != 0 else 0
                
                if cv > 1:
                    insights.append(f"📊 High variability in {col} (CV: {cv:.2f})")
                elif cv < 0.1:
                    insights.append(f"📊 Low variability in {col} (CV: {cv:.2f})")
        
        # Categorical insights
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols[:2]:
                unique_count = df[col].nunique()
                total_count = len(df)
                diversity = unique_count / total_count
                
                if diversity > 0.8:
                    insights.append(f"📋 High diversity in {col} ({unique_count} unique values)")
                elif diversity < 0.1:
                    insights.append(f"📋 Low diversity in {col} ({unique_count} unique values)")
        
        # Print insights
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        self.insights.extend(insights)
    
    def generate_recommendations(self, df, dataset_name):
        """Generate actionable recommendations"""
        self.print_header(f"Executive Recommendations - {dataset_name}")
        
        recommendations = []
        
        # Data quality recommendations
        quality_score = self.data_quality_report.get(dataset_name, {}).get('quality_score', 0)
        if quality_score < 80:
            recommendations.append("🔧 Implement data quality monitoring system")
            recommendations.append("📊 Establish data validation rules")
        
        # Analysis recommendations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 5:
            recommendations.append("🎯 Focus on top 5 most important variables for initial analysis")
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            recommendations.append("📈 Consider customer segmentation analysis")
        
        # Business recommendations
        recommendations.append("💼 Establish regular data review meetings")
        recommendations.append("📊 Create executive dashboard for key metrics")
        recommendations.append("🔄 Implement automated reporting system")
        
        # Print recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        self.recommendations.extend(recommendations)
    
    def create_executive_summary(self):
        """Create executive summary of analysis"""
        self.print_header("Executive Summary")
        
        print(f"📊 Project: {self.project_name}")
        print(f"🎯 Business Context: {self.business_context}")
        print(f"📈 Datasets Analyzed: {len(self.data_quality_report)}")
        
        # Overall quality score
        avg_quality = np.mean([report['quality_score'] for report in self.data_quality_report.values()])
        print(f"📊 Overall Data Quality: {avg_quality:.1f}/100")
        
        # Key insights
        print(f"\n💡 Key Insights ({len(self.insights)}):")
        for i, insight in enumerate(self.insights[:5], 1):  # Top 5 insights
            print(f"   {i}. {insight}")
        
        # Recommendations
        print(f"\n🎯 Recommendations ({len(self.recommendations)}):")
        for i, rec in enumerate(self.recommendations[:5], 1):  # Top 5 recommendations
            print(f"   {i}. {rec}")
        
        print("\n" + "=" * 80)

# Example usage
def demonstrate_framework():
    """Demonstrate the professional analysis framework"""
    
    # Initialize analyzer
    analyzer = ProfessionalAnalyzer(
        project_name="Retail Analytics Comprehensive",
        business_context="Multi-channel retail sales analysis for executive decision making"
    )
    
    print("🚀 Professional Data Analytics Framework Initialized")
    print("=" * 80)
    print("📊 This framework ensures consistent, comprehensive analysis across all projects")
    print("🎯 Demonstrates mastery of the complete data analytics ecosystem")
    print("=" * 80)
    
    return analyzer

# Initialize the framework
analyzer = demonstrate_framework()
