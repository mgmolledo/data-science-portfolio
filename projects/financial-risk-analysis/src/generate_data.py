"""
Generate Realistic Financial Dataset for Testing
Creates a comprehensive dataset that mimics the real bankruptcy prediction dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_realistic_bankruptcy_dataset(n_companies=2000):
    """
    Generate a realistic financial dataset for bankruptcy prediction
    Based on the structure of the real Kaggle dataset
    """
    
    np.random.seed(42)  # For reproducible results
    
    print(f"🏭 Generating realistic financial dataset for {n_companies:,} companies...")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Generate realistic financial data
    data = {}
    
    # 1. Profitability Ratios
    data['Net Income to Total Assets'] = np.random.normal(0.02, 0.15, n_companies)
    data['Operating Profit Rate'] = np.random.normal(0.05, 0.20, n_companies)
    data['Pre-tax Net Interest Rate'] = np.random.normal(0.03, 0.10, n_companies)
    data['After-tax Net Interest Rate'] = np.random.normal(0.02, 0.08, n_companies)
    data['Net Profit Rate'] = np.random.normal(0.03, 0.12, n_companies)
    
    # 2. Liquidity Ratios
    data['Current Ratio'] = np.random.lognormal(0.5, 0.8, n_companies)
    data['Quick Ratio'] = np.random.lognormal(0.3, 0.7, n_companies)
    data['Cash Ratio'] = np.random.lognormal(0.1, 0.6, n_companies)
    data['Working Capital to Total Assets'] = np.random.normal(0.1, 0.25, n_companies)
    
    # 3. Leverage Ratios
    data['Total debt/Total net worth'] = np.random.lognormal(0.5, 1.0, n_companies)
    data['Debt ratio %'] = np.random.beta(2, 3, n_companies) * 100
    data['Net worth to Assets'] = np.random.beta(3, 2, n_companies)
    data['Long-term liability to Assets'] = np.random.beta(1, 4, n_companies)
    
    # 4. Activity/Efficiency Ratios
    data['Total Asset Turnover'] = np.random.lognormal(0.2, 0.6, n_companies)
    data['Inventory Turnover Rate'] = np.random.lognormal(1.0, 0.8, n_companies)
    data['Receivables Turnover Rate'] = np.random.lognormal(1.5, 0.7, n_companies)
    data['Fixed Assets Turnover Rate'] = np.random.lognormal(0.5, 0.6, n_companies)
    
    # 5. Growth Ratios
    data['Revenue Growth Rate'] = np.random.normal(0.05, 0.30, n_companies)
    data['Net Income Growth Rate'] = np.random.normal(0.03, 0.40, n_companies)
    data['Total Asset Growth Rate'] = np.random.normal(0.08, 0.25, n_companies)
    
    # 6. Size and Scale
    data['Total Assets'] = np.random.lognormal(15, 2, n_companies)  # Log scale for assets
    data['Total Revenue'] = np.random.lognormal(14, 2, n_companies)
    data['Total Employees'] = np.random.lognormal(3, 1.5, n_companies)
    
    # 7. Industry-specific ratios
    data['Research and development expense rate'] = np.random.exponential(0.05, n_companies)
    data['Marketing Expense Rate'] = np.random.exponential(0.08, n_companies)
    data['Administrative Expense Rate'] = np.random.exponential(0.12, n_companies)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Introduce realistic correlations and bankruptcy patterns
    # Companies with negative profitability are more likely to go bankrupt
    profitability_score = (
        df['Net Income to Total Assets'] * 0.3 +
        df['Operating Profit Rate'] * 0.3 +
        df['Net Profit Rate'] * 0.4
    )
    
    # Companies with high leverage are more likely to go bankrupt
    leverage_score = (
        df['Total debt/Total net worth'] * 0.4 +
        df['Debt ratio %'] * 0.3 +
        (1 - df['Net worth to Assets']) * 0.3
    )
    
    # Companies with poor liquidity are more likely to go bankrupt
    liquidity_score = (
        (2 - df['Current Ratio']) * 0.4 +
        (1 - df['Quick Ratio']) * 0.3 +
        (0.2 - df['Working Capital to Total Assets']) * 0.3
    )
    
    # Calculate bankruptcy probability
    bankruptcy_prob = (
        np.clip(-profitability_score, 0, 1) * 0.4 +
        np.clip(leverage_score - 1, 0, 1) * 0.3 +
        np.clip(liquidity_score, 0, 1) * 0.3
    )
    
    # Add some randomness
    bankruptcy_prob += np.random.normal(0, 0.1, n_companies)
    bankruptcy_prob = np.clip(bankruptcy_prob, 0, 1)
    
    # Generate bankruptcy labels
    df['Bankrupt'] = (bankruptcy_prob > 0.3).astype(int)
    
    # Ensure realistic bankruptcy rate (around 15-20%)
    actual_bankruptcy_rate = df['Bankrupt'].mean()
    target_rate = 0.18
    
    if actual_bankruptcy_rate < target_rate:
        # Increase bankruptcy rate by adjusting threshold
        threshold = np.percentile(bankruptcy_prob, (1 - target_rate) * 100)
        df['Bankrupt'] = (bankruptcy_prob > threshold).astype(int)
    elif actual_bankruptcy_rate > target_rate * 1.5:
        # Decrease bankruptcy rate
        threshold = np.percentile(bankruptcy_prob, (1 - target_rate) * 100)
        df['Bankrupt'] = (bankruptcy_prob > threshold).astype(int)
    
    # Add some noise to make it more realistic
    for col in df.columns:
        if col != 'Bankrupt':
            noise = np.random.normal(0, 0.05, n_companies)
            df[col] += noise * df[col].std()
    
    # Handle extreme outliers
    for col in df.columns:
        if col != 'Bankrupt':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
    
    # Save dataset
    df.to_csv('data/data.csv', index=False)
    
    # Print summary
    print(f"✅ Dataset generated successfully!")
    print(f"📊 Shape: {df.shape}")
    print(f"🏢 Companies: {len(df):,}")
    print(f"📈 Features: {len(df.columns)-1}")
    print(f"💥 Bankruptcy rate: {df['Bankrupt'].mean():.1%}")
    print(f"💾 Saved to: data/data.csv")
    
    # Print feature summary
    print(f"\n📋 Feature Summary:")
    print(f"   Profitability ratios: 5")
    print(f"   Liquidity ratios: 4")
    print(f"   Leverage ratios: 4")
    print(f"   Activity ratios: 4")
    print(f"   Growth ratios: 3")
    print(f"   Size metrics: 3")
    print(f"   Industry ratios: 3")
    
    return df

if __name__ == "__main__":
    df = generate_realistic_bankruptcy_dataset(2000)
    
    # Show sample of the data
    print(f"\n📊 Sample of generated data:")
    print(df.head().round(3))
    
    # Show bankruptcy distribution
    print(f"\n💥 Bankruptcy distribution:")
    print(df['Bankrupt'].value_counts())
    print(f"Bankruptcy rate: {df['Bankrupt'].mean():.1%}")
