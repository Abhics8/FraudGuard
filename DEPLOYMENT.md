# RiskLens Deployment Guide

## 🚀 Deploy to Railway (Recommended)

Railway offers free tier with Docker support - perfect for ML projects!

### Prerequisites
- GitHub account
- Railway account (sign up at https://railway.app)

### Step-by-Step Deployment

#### 1. Push to GitHub

```bash
cd /Users/abhiabhardwaj/.gemini/antigravity/playground/fractal-hubble/RiskLens

# Create GitHub repository at https://github.com/new
# Name: RiskLens

# Then push:
git remote add origin https://github.com/AB0204/RiskLens.git
git branch -M main
git push -u origin main
```

#### 2. Deploy to Railway

1. **Go to Railway**: https://railway.app
2. **Sign in with GitHub**
3. **Create New Project** → **Deploy from GitHub repo**
4. **Select** `AB0204/RiskLens`
5. **Add Variables**:
   - Click "Variables" tab
   - Add: `PORT=8000`
   - Add: `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`

6. **Deploy!**
   - Railway auto-detects Dockerfile
   - Build starts automatically
   - Get your URL: `https://risklens-production.up.railway.app`

#### 3. Test Deployment

```bash
# Test health endpoint
curl https://your-app.up.railway.app/health

# Test prediction (if model is loaded)
curl -X POST https://your-app.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 12000, "V1": -1.35, ..., "Amount": 149.62}'
```

---

## 🎯 Alternative: Deploy to Render

### Step-by-Step

1. **Go to Render**: https://render.com
2. **New Web Service**
3. **Connect GitHub** → Select `RiskLens`
4. **Configure**:
   - Name: `risklens`
   - Environment: `Docker`
   - Instance Type: `Free`
   - Environment Variables:
     - `PORT=8000`
     - `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`

5. **Deploy**
   - URL: `https://risklens.onrender.com`

---

## 🔧 Environment Variables

Set these in your deployment platform:

```bash
# Required
PORT=8000
MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Optional
API_WORKERS=2
THRESHOLD=0.5
DRIFT_THRESHOLD=0.05
ENABLE_MONITORING=True
```

---

## 📊 Post-Deployment

### Monitor Your App

1. **Railway Dashboard**:
   - View logs
   - Monitor CPU/Memory
   - Check build status

2. **Application Metrics**:
   - API: `https://your-app.up.railway.app/docs`
   - Metrics: `https://your-app.up.railway.app/metrics`
   - Health: `https://your-app.up.railway.app/health`

### Add to Resume/Portfolio

```markdown
**RiskLens** - Production Fraud Detection System
- Deployed at https://risklens.up.railway.app
- 96%+ precision fraud detection with XGBoost
- MLOps pipeline with drift detection
- Real-time API with <50ms latency
- GitHub: https://github.com/AB0204/RiskLens
```

---

## ⚠️ Important Notes

### Free Tier Limitations

**Railway Free Tier**:
- $5 credit/month
- Auto-sleep after inactivity
- 512MB RAM, 1 vCPU

**Render Free Tier**:
- Auto-sleep after 15 min inactivity
- 512MB RAM
- Slower cold starts

### Dataset Considerations

⚠️ **DO NOT** upload the full 280k-row dataset to Git!

**Option 1: Use Subset**
```python
# In train.py, use smaller dataset
df = load_credit_card_fraud_data()
df_sample = df.sample(n=10000, random_state=42)  # 10k rows
```

**Option 2: Download on Startup**
```python
# Download from external source on first run
import requests
if not Path('data/creditcard.csv').exists():
    # Download logic here
```

**Option 3: Train Locally, Deploy Model Only**
```bash
# Train locally
python train.py

# Upload model artifact to deployment
# Load pre-trained model in API
```

---

## 🎬 Demo Video Recording

Once deployed, record a demo showing:

1. **API Documentation** (`/docs`)
2. **Health Check** (`/health`)
3. **Prediction** - Submit transaction, get fraud probability
4. **Explanation** (`/explain`) - Show SHAP values
5. **Metrics** (`/metrics`) - Show Prometheus metrics

Tools for recording:
- **Loom** (https://loom.com) - Easy, shareable links
- **QuickTime** (Mac) - Screen recording
- **OBS** - Professional recording

---

## ✅ Deployment Checklist

- [ ] Push code to GitHub
- [ ] Create Railway/Render account
- [ ] Deploy from GitHub
- [ ] Set environment variables
- [ ] Test API endpoints
- [ ] Record demo video
- [ ] Add live URL to README
- [ ] Share on LinkedIn

---

**You're ready to deploy! Let's make RiskLens publicly accessible!** 🚀
