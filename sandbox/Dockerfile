FROM python:3.12-slim

# Non-root user — critical for security
RUN useradd --create-home --uid 1000 sandbox

WORKDIR /sandbox

# Install ONLY pandas and numpy — no app code, no groq, no streamlit
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Runner script
COPY runner.py ./runner.py

# Drop privileges
USER sandbox

# Entry point reads code from stdin
ENTRYPOINT ["python", "runner.py"]