FROM python:3.10-bullseye as base

FROM base as pybuilder

RUN set -eux; \
        apt update; \
        apt install -y --no-install-recommends \
        libolm-dev \
        ; \
        rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN pip install -U pip setuptools wheel && pip install --user -r /requirements.txt && rm /requirements.txt


FROM base as runner
RUN set -eux; \
        apt update; \
        apt install -y --no-install-recommends \
        libolm-dev \
        ; \
        rm -rf /var/lib/apt/lists/*
        
COPY --from=pybuilder /root/.local /usr/local
COPY . /app


FROM runner
WORKDIR /app
CMD ["python", "bot.py"]
