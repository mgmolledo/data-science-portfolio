# Financial Risk Analysis Dashboard Deployment Guide

## Quick Deploy to Heroku

1. Go to [Heroku](https://heroku.com/)
2. Create a new app
3. Connect your GitHub repository
4. Set buildpack to Python
5. Main file path: `projects/financial-risk-analysis/src/dashboard.py`
6. App URL: `financial-risk-analysis`

## Configuration
- The app will automatically use the requirements.txt
- No additional secrets needed
- The app loads financial data automatically if data files are not found

## Features
- ✅ Professional financial risk analysis
- ✅ Machine learning model performance metrics
- ✅ Interactive risk visualizations
- ✅ Executive-level insights
- ✅ Robust error handling
- ✅ Optimized for cloud deployment

## Data Sources
The app will look for data files in the following order:
1. `data/enhanced_dataset.csv`
2. `../data/enhanced_dataset.csv`
3. `./data/enhanced_dataset.csv`

If no data files are found, it will load realistic financial data automatically.