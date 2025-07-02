FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app
COPY pyproject.toml uv.lock ./

RUN uv sync --no-install-project

COPY . .

RUN uv sync
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8501 

CMD ["uv", "run", "streamlit", "run", "app.py"]

