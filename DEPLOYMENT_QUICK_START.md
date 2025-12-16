# Quick Deployment Guide 🚀

## Your GitHub Repository
✅ **Already connected**: https://github.com/HemachandRavulapalli/Advanced-ECG

## Option 1: Deploy to Vercel (Frontend) + Railway (Backend) - RECOMMENDED

### Step 1: Deploy Frontend to Vercel (FREE)

1. Go to [vercel.com](https://vercel.com) and sign up/login with GitHub
2. Click "New Project"
3. Import your repository: `HemachandRavulapalli/ECG-Arrhythmia-Classification`
4. Configure:
   - **Root Directory**: `frontend`
   - **Framework Preset**: Vite
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
5. Click "Deploy"
6. **Copy your frontend URL** (e.g., `https://ecg-classification.vercel.app`)

### Step 2: Deploy Backend to Railway (FREE tier available)

1. Go to [railway.app](https://railway.app) and sign up/login with GitHub
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Add a new service → Select "Backend" folder
5. Railway will auto-detect Python
6. Add environment variable (if needed):
   - `PORT`: 8000
7. **Copy your backend URL** (e.g., `https://ecg-backend.railway.app`)

### Step 3: Update Frontend API URL

1. In Vercel dashboard, go to your project → Settings → Environment Variables
2. Add: `VITE_API_URL` = `https://your-backend-url.railway.app`
3. Redeploy frontend

### Step 4: Update Backend CORS

1. In Railway, go to your backend service
2. Open the code editor or use Railway CLI
3. Edit `backend/app.py`, update CORS:
```python
allow_origins=["https://your-frontend.vercel.app"]
```
4. Redeploy backend

## Option 2: Deploy Both on Railway

1. Go to [railway.app](https://railway.app)
2. Create new project from GitHub
3. Add **two services**:
   - Service 1: Frontend (root: `frontend`)
   - Service 2: Backend (root: `backend`)
4. For Frontend service:
   - Build command: `npm install && npm run build`
   - Start command: `npm run preview` (or use a static file server)
5. For Backend service:
   - Start command: `cd backend && python app.py`

## Option 3: Deploy to Render (Alternative)

### Frontend on Render
1. Go to [render.com](https://render.com)
2. New → Static Site
3. Connect GitHub repo
4. Root directory: `frontend`
5. Build command: `npm run build`
6. Publish directory: `dist`

### Backend on Render
1. New → Web Service
2. Connect GitHub repo
3. Root directory: `backend`
4. Build command: `pip install -r requirements.txt`
5. Start command: `python app.py`

## Testing Your Deployment

1. Visit your frontend URL
2. Try uploading an ECG file
3. Check browser console for errors
4. Check backend logs in Railway/Render dashboard

## Important Notes

⚠️ **Model Files**: Your trained models in `backend/src/saved_models/` need to be accessible. Options:
- Include them in the repository (if under GitHub's 100MB limit)
- Upload to cloud storage (S3, etc.) and download during deployment
- Use Railway's persistent storage

⚠️ **Environment Variables**: 
- Frontend: `VITE_API_URL` (your backend URL)
- Backend: Update CORS origins to include frontend URL

⚠️ **File Size Limits**:
- Vercel: 100MB per file
- Railway: Check their limits
- Consider compressing models or using external storage

## Quick Commands

```bash
# Push latest changes to GitHub
git add .
git commit -m "Your message"
git push origin main

# Deploy will happen automatically if connected to Vercel/Railway
```

## Need Help?

- Vercel Docs: https://vercel.com/docs
- Railway Docs: https://docs.railway.app
- Check `DEPLOYMENT.md` for detailed instructions

