FROM python:3.11-bullseye as stage_base

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        alsa-utils \
        libsndfile1-dev \
        gcc \
        libasound2-dev \
        portaudio19-dev \
        python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install poetry==1.4.2

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root && rm -rf $POETRY_CACHE_DIR

# Install PyAudio separately
RUN pip install pyaudio==0.2.14

FROM python:3.11-bullseye as stage_production

# Copy only necessary runtime dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        alsa-utils \
        libsndfile1 \
        libasound2 \
        libportaudio2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=.

COPY --from=stage_base ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY . /app
WORKDIR /app

CMD ["python", "voice_streaming.py", "--batch-size", "30"]