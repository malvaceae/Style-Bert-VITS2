FROM python:3.10-slim

COPY requirements.txt /tmp

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    python -c 'import pyopenjtalk; pyopenjtalk._lazy_init()'

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/ml/code:${PATH}"

WORKDIR /opt/ml/code

COPY . .
COPY --chmod=755 serve .
COPY --chmod=755 train .
