FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ─── Backend + final image ────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend
COPY backend/ ./backend/
COPY app.py .
COPY README.md .

# Copy built React frontend
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Add backend to Python path
ENV PYTHONPATH=/app/backend
ENV DJANGO_SETTINGS_MODULE=energy_api.settings
ENV DEBUG=False

EXPOSE 7860

CMD ["python", "app.py"]
