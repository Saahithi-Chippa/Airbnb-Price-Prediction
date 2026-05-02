FROM python:3.9-slim

WORKDIR /app

# Install dependencies first (separate layer so it's cached on code-only changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the src package
COPY setup.py .
COPY src/ src/
RUN pip install --no-cache-dir -e .

# Copy application code and static assets
COPY app/ app/
COPY data/ data/
COPY models/ models/

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
