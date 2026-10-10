FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.24 /uv /uvx /usr/local/bin/

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --all-groups --no-install-project

COPY src/ ./src/
COPY tests/ ./tests/
COPY README.md ./
RUN uv sync --frozen --all-groups

# Ownership appuser ko do — cache, venv sab likh sake
RUN chown -R appuser:appuser /app

ENV UV_NO_SYNC=1

USER appuser

CMD ["/app/.venv/bin/pytest", "-v"]