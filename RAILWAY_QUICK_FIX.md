# Railway Deployment - Quick Fix Guide

## The Problem
Railway couldn't detect the language because it was looking at the root directory which contains both frontend and backend.

## The Solution
You need to create **TWO SEPARATE SERVICES** in Railway, each with its own root directory.

## Step-by-Step Fix

### 1. Delete Current Service (if exists)
- In Railway dashboard, delete the current service that's failing

### 2. Create Backend Service

1. Click **"+ New"** → **"GitHub Repo"**
2. Select your repository: `HemachandRavulapalli/ECG-Arrhythmia-Classification`
3. **IMPORTANT**: Click on the service → **Settings** → **Root Directory**
4. Set Root Directory to: `backend`
5. Railway will now detect Python automatically
6. It will use:
   - `backend/requirements.txt` for dependencies
   - `backend/Procfile` for start command
   - `backend/app.py` as the entry point

### 3. Create Frontend Service

1. In the same project, click **"+ New"** → **"GitHub Repo"**
2. Select the same repository again
3. **IMPORTANT**: Click on the service → **Settings** → **Root Directory**
4. Set Root Directory to: `frontend`
5. Railway will detect Node.js automatically
6. Go to **Settings** → **Deploy** → **Start Command**
7. Set to: `npm run preview` or `npx serve dist -p $PORT`

### 4. Configure Frontend Build

1. Go to frontend service → **Settings** → **Build**
2. Build Command: `npm install && npm run build`
3. Output Directory: `dist` (if asked)

### 5. Set Environment Variables

**Frontend Service:**
- `VITE_API_URL`: Your backend URL (get it from backend service → Generate Domain)

**Backend Service:**
- `PORT`: Railway sets this automatically (no need to add)

### 6. Update CORS in Backend

After you get your frontend URL, update `backend/app.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-frontend-service.up.railway.app",  # Your Railway frontend URL
        "http://localhost:3000"  # Keep for local dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Then commit and push - Railway will auto-deploy.

## Visual Guide

```
Railway Project
├── Service 1: Backend
│   └── Root Directory: backend/
│   └── Auto-detects: Python
│   └── Uses: backend/requirements.txt, backend/Procfile
│
└── Service 2: Frontend
    └── Root Directory: frontend/
    └── Auto-detects: Node.js
    └── Uses: frontend/package.json
    └── Start Command: npm run preview
```

## What Changed in Code

✅ `Procfile` moved to `backend/Procfile`
✅ `railway.json` added to both `backend/` and `frontend/`
✅ Backend now reads `PORT` from environment
✅ Frontend `package.json` updated with preview command

## After Deployment

1. Get backend URL: Backend service → Settings → Generate Domain
2. Get frontend URL: Frontend service → Settings → Generate Domain
3. Update `VITE_API_URL` in frontend service variables
4. Update CORS in backend code
5. Redeploy both services

## Still Having Issues?

Check the detailed guide: `RAILWAY_DEPLOYMENT.md`

