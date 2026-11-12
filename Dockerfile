# One image for both services in compose.yaml: the web app and the isolated command runner.
FROM python:3.12-alpine
RUN apk add --no-cache bash ffmpeg sox exiftool ghostscript font-dejavu \
        imagemagick imagemagick-jpeg imagemagick-webp imagemagick-heic imagemagick-tiff imagemagick-pdf imagemagick-svg \
 && adduser -D -u 10001 clip \
 && mkdir -p /jobs /run/clipwright && chown clip /jobs /run/clipwright
WORKDIR /srv
COPY app/requirements.txt app/
RUN pip install --no-cache-dir -r app/requirements.txt
COPY . .
USER clip
ENV HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
