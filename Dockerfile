FROM python:3.11-slim

WORKDIR /app

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend_requirements.txt
COPY frontend/requirements.txt ./frontend_requirements.txt

RUN pip install --no-cache-dir -r backend_requirements.txt -r frontend_requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV API_BASE_URL=http://localhost:8000

EXPOSE 8000
EXPOSE 7860

RUN echo '#!/bin/bash\n\
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &\n\
streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]