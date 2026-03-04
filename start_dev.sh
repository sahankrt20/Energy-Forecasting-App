#!/bin/bash
# ─── Energy Forecasting App — Local Dev Startup ─────────────────────────────
set -e

echo "⚡ Energy Consumption Forecasting — Dev Setup"
echo "=============================================="

# ── Backend ──────────────────────────────────────────────────────────────────
echo ""
echo "📦 Installing backend dependencies..."
cd backend
pip install -r requirements.txt -q

echo "🚀 Starting Django backend on http://localhost:8000 ..."
DJANGO_SETTINGS_MODULE=energy_api.settings python manage.py runserver 0.0.0.0:8000 &
DJANGO_PID=$!

cd ..

# ── Frontend ─────────────────────────────────────────────────────────────────
echo ""
echo "📦 Installing frontend dependencies..."
cd frontend
npm install --silent

echo "⚛️  Starting React frontend on http://localhost:5173 ..."
npm run dev &
VITE_PID=$!

cd ..

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "✅ Both servers running!"
echo "   Frontend → http://localhost:5173"
echo "   Backend  → http://localhost:8000/api/"
echo ""
echo "Press Ctrl+C to stop all servers."

trap "kill $DJANGO_PID $VITE_PID 2>/dev/null; echo 'Servers stopped.'" EXIT
wait
