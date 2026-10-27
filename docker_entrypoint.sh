#!/bin/sh
# RunPod pod (JOB_ID set by cloud/job.py): download photos from R2, run, upload the result.
# Otherwise (docker compose run ... <main.py options>): the pipeline on the mounted folders.
set -e
if [ -n "$JOB_ID" ]; then
    exec python3 runpod_job.py
fi
exec python3 main.py "$@"
