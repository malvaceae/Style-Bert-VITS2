FROM ubuntu:22.04

COPY requirements.txt /tmp

RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3-pip && rm -rf /var/lib/apt/lists/* && \
    pip3 install --no-cache-dir --upgrade pip setuptools && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    python3 -c 'import pyopenjtalk; pyopenjtalk._lazy_init()'

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/ml/code:${PATH}"

WORKDIR /opt/ml/code

COPY . .
COPY --chmod=755 serve .
COPY --chmod=755 train .
