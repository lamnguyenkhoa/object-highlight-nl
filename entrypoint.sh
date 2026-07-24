#!/bin/sh
set -e

if [ ! -f /app/model/.download_complete ]; then
    echo "Model weights not found in volume, downloading..."
    python download_models.py
    touch /app/model/.download_complete
fi

exec python app.py "$@"
