FROM nvidia/cuda:12.1.1-base-ubuntu22.04

COPY requirements.txt /tmp

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s python3 /usr/bin/python && \
    pip list -o --format freeze | \
        grep -v '^-e' | \
        cut -f 1 -d = | \
        xargs pip install --no-cache-dir -U && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    python -c 'import pyopenjtalk; pyopenjtalk._lazy_init()'

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/ml/code:${PATH}"

WORKDIR /opt/ml/code

COPY . .
COPY --chmod=755 serve .
COPY --chmod=755 train .
