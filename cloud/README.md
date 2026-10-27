# Cloud: RunPod + Cloudflare R2

One scan = one pod, started on demand and terminated as soon as its result is in R2.
The pipeline (`pipeline/`, `main.py`) does not depend on this folder; `runpod_job.py` (in the Docker image)
and `cloud/job.py` (on the Pi / laptop) connect the two.

```
Pi / laptop                              R2 (one bucket)                          RunPod pod
───────────                              ───────────────                          ──────────
1. job id, preflight
2. upload photos  ─────────────────────▶ jobs/<id>/input/rig/0|1/*.jpg
3. verify every photo is there
4. temporary credentials (TTL = pod runtime cap, 60 min):
   input/ read-only, output/ read-write
5. create the pod ──────────────────────────────────────────────────────────────▶ 6. download the photos
                                                                                   7. pipeline (main.run_pipeline)
                                         jobs/<id>/output/ ◀──────────────────────  8. output.zip, stats.json,
                                                                                      run_log.txt, status.json (last)
9. wait for status.json, terminate the pod (always)
10. download + extract output.zip -> results/<id>/
11. delete jobs/<id>/ from R2
```

## Files

| File | Role |
|---|---|
| `cloudflare_r2.py` | S3 client for R2, job key layout (`jobs/<id>/input|output/`), uploads (never raise), downloads, prefix deletion, preflight, **temporary prefix-scoped credentials** |
| `runpod.py` | RunPod REST client (`POST/GET/DELETE /v1/pods`), pod request, pod lifecycle with guaranteed termination and bounded retries |
| `job.py` | The whole flow above for one scan (`run_cloud_job`), used by the Pi and by `tools/cloud_scan.py` |
| `../runpod_job.py` | Inside the pod: download, run, upload, `status.json` |
| `../tools/cloud_scan.py` | Runs the flow from the laptop with `input_images/` (not in the Docker image) |

## Keys: who holds what

| Where | Key | Can do |
|---|---|---|
| Pi / laptop (`.env`) | `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY`: R2 key limited to the scanner bucket | upload photos, read results, delete jobs |
| | `CLOUDFLARE_R2_API_TOKEN`: Cloudflare API token with **R2 permissions only** | mint temporary credentials |
| | `RUNPOD_API_KEY` | create / delete pods |
| Pod | `R2_IN_*`: temporary, `object-read-only`, prefix `jobs/<id>/input/` | read the photos of this job |
| | `R2_OUT_*`: temporary, `object-read-write`, prefix `jobs/<id>/output/` | write the result of this job |

The temporary credentials come from Cloudflare's `POST /accounts/{account_id}/r2/temp-access-credentials`
(`parentAccessKeyId` = the bucket key above, so they can never do more than it). They expire after
`RUNPOD_POD_MAX_RUNTIME_S` (default 3600 s), the same cap after which the client terminates the pod: one setting,
so the credentials cannot expire before the pod finishes uploading. The pod never gets the RunPod key and never
stops itself; the client terminates it.

## Choosing the pod

`POST https://rest.runpod.io/v1/pods` with:
- `gpuTypeIds`: acceptable GPUs, cheapest first (`RUNPOD_GPU_TYPES`, default list in `runpod.py`), and
  `gpuTypePriority: "custom"`, so RunPod takes the first one with capacity in our order. GPU speed barely matters
  (2.3 of 30 minutes are GPU compute); host RAM, vCPUs and disk do.
- `minRAMPerGPU` = `ceil(estimate_ram_gb × 1.3)` from `pipeline/resources.py` (photos × resolution),
  `minVCPUPerGPU` (default 8), `containerDiskInGb` from the photo count.
- No data center filter (more capacity), no ports, no volume.

Retries are bounded: a create request rejected for lack of capacity is retried `RUNPOD_MAX_CREATE_ATTEMPTS` times
(free); a second pod is started only if the first reported `ram_insufficient` (asks for the RAM it measured × 1.2)
or died without writing a status (likely OOM, asks for 1.5× RAM). `pipeline_error`, timeout and cancel are not retried.

## status.json (written last by the pod)

```json
{"job_id": "...", "state": "ok | ram_insufficient | pipeline_error | input_error | timeout",
 "error": "...", "required_gb": 0, "available_gb": 0, "resources": {...}, "timings_s": {...}}
```

## Test from the laptop

```bash
cp .env.example .env       # fill in the keys and TURNTABLE_IMAGE
python tools/cloud_scan.py --dry_run    # preflight + estimate + the pod request (secrets redacted); no pod
python tools/cloud_scan.py              # full flow; prints timings, the pod used and the cost estimate
python tools/cloud_scan.py --keep_remote                 # leave jobs/<id>/ in R2
python tools/cloud_scan.py --job_id <id> --skip_upload   # retry with the photos already in R2
```

Before the first real run:
1. Build and push the image (`docker build -t <user>/turntable-studio:v1 . && docker push ...`). Then `python tools/cloud_scan.py --mock` tests image, keys, R2 and pod termination with simulated OpenMVS.
   For a private registry, add its credentials in RunPod (Settings → Container Registry Auth) and put the id in
   `RUNPOD_REGISTRY_AUTH_ID`.
2. Check the GPU ids in `--dry_run` against RunPod's list (some ids contain double spaces there).
3. Optional safety net: an R2 lifecycle rule that expires `jobs/` after a few days (abandoned jobs).

## Upload queue on the Pi

`upload_file(path, bucket, key)` never raises (the queue retries; the photo stays on disk). Keys:
`r2.input_key(job_id, "rig/0/f_1000.jpg")`. At the end of the scan: `verify_keys(...)` and only then
`run_cloud_job(..., skip_upload=True, job_id=job_id)`. `/prepare` no longer empties the bucket: a new job id is enough.
