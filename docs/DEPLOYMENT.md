# Deployment Guide

## Frontend Deployment (Vercel)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Click "Import Project"
   - Select your GitHub repository
   - Framework: Next.js
   - Root Directory: `frontend`
   - Add environment variable: `NEXT_PUBLIC_API_URL=<your-backend-url>`
   - Click "Deploy"

## Backend Deployment (Railway)

1. **Go to [railway.app](https://railway.app)**
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Add environment variables:
Visit: http://localhost:3000
Deployment
Frontend: Deploy to Vercel (one-click)
Backend: Deploy to Railway/Render/Fly.io
See DEPLOYMENT.md for details.
meta-learning-recommender/
├── frontend/          # Next.js web interface
├── backend/           # FastAPI Python backend
├── desktop/           # Original Tkinter GUI (legacy)
├── data/             # Pre-trained models & datasets
└── docs/             # Documentationnstall -r requirements.txt`
   - Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables (same as Railway)

## Local Development

```bash
# Terminal 1 - Backend
cd backend
pip install -r requirements.txt
python api/main.py

# Terminal 2 - Frontend
cd frontend
npm install
npm run dev
```

Visit: http://localhost:3000

## Environment Variables

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Backend (.env)
```
ENV=development
OPENAI_API_KEY=sk-...
ALLOWED_ORIGINS=http://localhost:3000
```
