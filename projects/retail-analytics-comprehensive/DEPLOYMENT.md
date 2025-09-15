# Retail Analytics Dashboard Deployment Guide

## Quick Deploy to Heroku

1. Go to [Heroku](https://heroku.com/)
2. Create a new app
3. Connect your GitHub repository
4. Set buildpack to Python
5. Main file path: `projects/retail-analytics-comprehensive/dashboards/plotly_dash/retail_dashboard.py`
6. App URL: `retail-analytics-comprehensive`

## Configuration
- The app will automatically use the requirements.txt
- No additional secrets needed
- The app loads retail data automatically if data files are not found

## Features
- ✅ Comprehensive retail analytics
- ✅ Multi-channel analysis
- ✅ Customer segmentation
- ✅ Geographic distribution
- ✅ Interactive filters
- ✅ Professional visualizations
- ✅ Optimized for cloud deployment

## Data Sources
The app will look for data files in the following order:
1. `data/raw/customers.csv`
2. `../data/raw/customers.csv`
3. `./data/raw/customers.csv`

And similarly for products, transactions, and promotions files.

If no data files are found, it will load realistic retail data automatically.