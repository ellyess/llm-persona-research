# Minimal image that runs the simulation against the MOCK model: no API key,
# no network calls, no spend. For a real-model run, pass USE_MOCK=0 and an
# ANTHROPIC_API_KEY at `docker run` time.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the simulation needs (keeps the image lean and never copies
# the local api.py scratch key, which is not selected here).
COPY personas_sim/ ./personas_sim/
COPY tests/ ./tests/

# Offline mock is the default: the container runs end to end with no secrets.
ENV USE_MOCK=1

CMD ["python", "-m", "personas_sim.run"]
