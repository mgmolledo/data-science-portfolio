# AI-Powered BI Dashboard Deployment Guide

## Quick Deploy to Heroku

### Option 1: Optimized Version (Recommended)
1. Go to [Heroku](https://heroku.com/)
2. Create a new app
3. Connect your GitHub repository
4. Set buildpack to Python
5. Main file path: `projects/ai-powered-bi/dashboards/plotly_dash/ai_bi_dashboard.py`
6. App URL: `ai-powered-bi-optimized`

### Option 2: Full Version
1. Go to [Heroku](https://heroku.com/)
2. Create a new app
3. Connect your GitHub repository
4. Set buildpack to Python
3. Main file path: `projects/ai-powered-bi/dashboards/plotly_dash/ai_bi_dashboard_full.py`
4. App URL: `ai-powered-bi-full`

### Option 3: Simple Version
1. Go to [Heroku](https://heroku.com/)
2. Create a new app
3. Connect your GitHub repository
4. Set buildpack to Python
3. Main file path: `projects/ai-powered-bi/dashboards/plotly_dash/ai_bi_dashboard_simple.py`
4. App URL: `ai-powered-bi-simple`

## Configuration
- The app will automatically use the optimized requirements.txt
- No additional secrets needed for the optimized version
- The app loads business data automatically

## Features
- ✅ Conversational AI interface
- ✅ Interactive data exploration
- ✅ Automated insights generation
- ✅ Professional styling
- ✅ Optimized for cloud deployment
- ✅ Fast loading times