FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry 

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi

# 7. Copy the rest of your application code
COPY src/ /app/src/
COPY gallery/ /app/gallery/
COPY Results/ /app/Results/

# 8. Define the default command to run when the container starts
CMD ["python", "src/main.py"]