# Deploy FraudGuard to Hugging Face Spaces

## 🚀 Quick Deploy Guide

### Step 1: Create Hugging Face Account (1 min)
1. Go to https://huggingface.co/join
2. Sign up (no credit card needed!)
3. Verify your email

### Step 2: Create a New Space (2 min)
1. Go to https://huggingface.co/new-space
2. Fill in details:
   - **Owner**: Your username
   - **Space name**: `FraudGuard`
   - **License**: MIT
   - **Select SDK**: **Gradio**
   - **Space hardware**: **CPU basic** (free tier)
   - **Visibility**: Public

3. Click **"Create Space"**

### Step 3: Upload Files (3 min)

You'll be on your Space's page. Click **"Files"** tab, then upload:

**Required files:**
1. `app.py` (the Gradio interface)
2. `README_HF.md` → Rename to `README.md` when uploading
3. `requirements_hf.txt` → Rename to `requirements.txt` when uploading

**How to upload:**
- Click "Add file" → "Upload files"
- Drag and drop the 3 files
- **Important**: Rename `README_HF.md` to `README.md` and `requirements_hf.txt` to `requirements.txt`
- Commit the files

### Step 4: Wait for Build (2-3 min)

Hugging Face will automatically:
- ✅ Install dependencies
- ✅ Build the Gradio app
- ✅ Deploy it live

Watch the **"Build logs"** - when you see ✅ "Running on public URL", you're live!

### Step 5: Get Your Live URL

Your app is now live at:
```
https://huggingface.co/spaces/YOUR_USERNAME/FraudGuard
```

---

## 📝 Files to Upload

### File 1: `app.py`
Location: `/Users/abhiabhardwaj/.gemini/antigravity/playground/fractal-hubble/FraudGuard/app.py`

### File 2: `README.md` 
Upload: `README_HF.md` as `README.md`
Location: `/Users/abhiabhardwaj/.gemini/antigravity/playground/fractal-hubble/FraudGuard/README_HF.md`

### File 3: `requirements.txt`
Upload: `requirements_hf.txt` as `requirements.txt`
Location: `/Users/abhiabhardwaj/.gemini/antigravity/playground/fractal-hubble/FraudGuard/requirements_hf.txt`

---

## ✅ Post-Deployment

1. **Test your app**: Try the examples and sliders
2. **Share your URL**: Add to your resume and LinkedIn
3. **Update GitHub README**: Add live demo link

### Update GitHub README
```bash
cd /Users/abhiabhardwaj/.gemini/antigravity/playground/fractal-hubble/FraudGuard
# Edit README.md - replace live demo section with your HF URL
git add README.md
git commit -m "docs: Add Hugging Face Spaces live demo"
git push origin main
```

---

## 💡 Tips

- **Slow loading?** Normal for free tier on first run (cold start)
- **Want to update?** Just upload new `app.py` file
- **Broken?** Check "Build logs" for errors

---

## 🎉 You're Done!

Your fraud detection demo is now **live and free** on Hugging Face Spaces!

**Example URL**: https://huggingface.co/spaces/Ab0202000/FraudGuard
