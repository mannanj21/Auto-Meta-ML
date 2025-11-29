# Meta-Learning Algorithm Recommender System

🤖 **AI-Powered Algorithm Recommendation System** using Meta-Learning

## Features
- ✅ Upload CSV datasets and get personalized ML algorithm recommendations
- ✅ Meta-learning powered by MLkNN on OpenML-CC18 benchmark
- ✅ Beautiful visualizations and AI-generated explanations
- ✅ Modern web interface (Next.js + FastAPI)
- ✅ Pre-trained models for instant predictions

## Architecture
Frontend (Next.js) → Backend (FastAPI) → ML Pipeline (scikit-learn)
Vercel Railway/Render Pre-trained Models

## Quick Start

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python api/main.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Visit: http://localhost:3000

## Deployment
- **Frontend**: Deploy to Vercel (one-click)
- **Backend**: Deploy to Railway/Render/Fly.io

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for details.

## Project Structure
cd frontend
npm install
npm run dev.js 14, React, TailwindCSS, Recharts
- **Backend**: FastAPI, scikit-learn, scikit-multilearn
- **ML**: MLkNN, PyMFE, OpenML

## License
MIT License

## Authors
Mannan Jain
Dron Garg
