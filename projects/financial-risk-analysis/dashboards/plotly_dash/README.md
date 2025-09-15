# Financial Risk Analysis Dashboard - Plotly Dash Deployment

## Quick Deploy to Heroku

### Prerequisites
- Heroku CLI installed
- Git repository connected
- Python 3.8+ environment

### Deployment Steps

1. **Create Heroku App**
   ```bash
   heroku create financial-risk-dashboard
   ```

2. **Set Buildpack**
   ```bash
   heroku buildpacks:set heroku/python
   ```

3. **Configure Environment**
   ```bash
   heroku config:set PYTHONPATH=/app
   ```

4. **Deploy**
   ```bash
   git push heroku main
   ```

5. **Open App**
   ```bash
   heroku open
   ```

### Local Development

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Dashboard**
   ```bash
   python projects/financial-risk-analysis/dashboards/plotly_dash/dashboard.py
   ```

3. **Access Dashboard**
   - URL: http://localhost:8051

### Features
- ✅ Professional financial risk analysis
- ✅ Machine learning model performance metrics
- ✅ Interactive risk visualizations
- ✅ Executive-level insights
- ✅ Robust error handling
- ✅ Optimized for cloud deployment

### Configuration
- Port: 8051
- Host: 0.0.0.0
- Debug: True (development)
- Debug: False (production)
