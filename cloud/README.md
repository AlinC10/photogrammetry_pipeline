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
11. delete the photos (jobs/<id>/input/) from R2; the result stays in jobs/<id>/output/ until the next job
    deletes the older jobs at its start (without that cleanup, e.g. --keep_old_jobs, the whole job goes now)
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
- `gpuTypeIds`: acceptable GPUs, fastest first, only those up to `RUNPOD_MAX_GPU_PRICE` $/h (default 1.10:
  RTX 4090, L40S, RTX 6000 Ada, L40, RTX 3090, ... A4000; the RTX 5090 costs 1.19 on Secure Cloud and runs our
  CUDA 12.2 build only through the driver's PTX JIT, untested). `RUNPOD_GPU_TYPES` replaces the list. With
  `gpuTypePriority: "custom"` RunPod takes the first one with capacity in our order. The GPU speeds up only
  DensifyPointCloud; host RAM, vCPUs and disk matter more. A pod above the cap (prices changed) is logged.
- `minRAMPerGPU` = `ceil(estimate_ram_gb × 1.3)` from `pipeline/resources.py` (photos × resolution),
  `minVCPUPerGPU` (default 8), `containerDiskInGb` from the photo count.
- Asked for first: at least `RUNPOD_PREFERRED_VCPU` (16) and `RUNPOD_PREFERRED_RAM_GB` (48 GB). The price is per
  GPU, yet the same RTX 3090 came with 8 vCPU (6 usable: the 4-level stand timed out) or 32 vCPU. Only when no
  host has them is the same create attempt repeated at once with the minimums above.
- No data center filter (more capacity), no ports, no volume.

Retries are bounded: a create request rejected for lack of capacity is retried `RUNPOD_MAX_CREATE_ATTEMPTS` times
(free); a second pod is started only if the first reported `ram_insufficient` (asks for the RAM it measured × 1.2)
or died without writing a status (likely OOM, asks for 1.5× RAM). `pipeline_error`, timeout and cancel are not retried.

## status.json (written last by the pod)

```json
{"job_id": "...", "state": "ok | ram_insufficient | pipeline_error | input_error | timeout | cancelled",
 "error": "...", "required_gb": 0, "available_gb": 0, "resources": {...}, "timings_s": {...},
 "warnings": [{"code": "not_watertight", "message": "...", ...}], "debug_lite": {"uploaded": false, ...}}
```

`warnings` (with `state` "ok"): the model was delivered but may need a rescan or a check. Codes: `not_watertight`,
`model_tiny`, `model_at_disc_rim`, `most_faces_dropped`, `masks_disagree`, `hull_cone_only`. The `message` is for
people (the Pi / front-end can show it as is).

## progress.json

While it runs the pod writes `jobs/<id>/output/progress.json`: `{"step": "<last finished step>", "at": ...}`, the
step names of `stats.json` ("2. Turntable Poses (Rigid Lock)", ...: the Pi's `pod_step_label` maps them to messages).
In streaming mode, before the pipeline: `{"step": "0. Waiting for photos", "photos": n, "level": L}`.

## Streaming: pod started with the scan (JOB_JSON `"stream": true`)

`plans/PLAN_STREAMING_DAG.md` §2. The pod is requested when the scan starts; it lists `jobs/<id>/input/` every
`STREAM_POLL_S` (2 s), downloads each new photo (to `<name>.part`, then renamed) and pre-computes what depends on the
photos alone (`pipeline/stream_prework.py`: SIFT, background medians, preliminary poses + AI crops of every finished
level). The scanner writes, under `jobs/<id>/input/`:

| file | when | content |
|---|---|---|
| `level_done_<L>.json` | optional, when the photos of level L are uploaded | `{"level": L, "files": ["rig/0/f_1000.jpg", ...]}` |
| `capture_done.json` | after the upload of every photo was verified | `{"n_images": 80, "levels": [1, 2], "files": [...], "step_deg": 18, "finished_at": ...}` |
| `cancel.json` | the scan was cancelled (besides terminating the pod) | `{}` |

On `capture_done.json` the pod downloads what is missing, keeps exactly the listed photos and runs the pipeline with
the caches; the results are those of a normal run (only the time changes). No new photo for `STREAM_IDLE_S` (420 s)
without `capture_done.json` → `status.json` `input_error` ("capture abandoned"); `cancel.json` → `cancelled`.
`status.json` gets `"stream"` (photos, capture time, pre-work per level) and `timings_s.wait_after_capture_s`
(capture_done seen → status.json). Without `"stream"` the job runs as before. The pod RAM is chosen before the number
of photos is known: the launcher plans it for `2 × (360 / step_deg) × max_levels` photos.

## Debug archive (optional, separate bucket)

The pod uploads `jobs/<id>/debug_lite.zip` (raw mesh as Draco, masks at 1/4, visual hull, points the hull pruned /
added, small reports; ~5-40 MB, `pipeline/debug_lite.py`) to the DEBUG bucket, never to the delivery bucket, when the
launcher gives it temporary credentials for it. The launcher (Pi / `cloud/job.py`) adds one line next to
`pod_credentials_env`:

```python
env.update(r2.debug_credentials_env(job_id, ttl))   # {} without R2_DEBUG_BUCKET: nothing changes
```

`R2_DEBUG_BUCKET` (and optionally `R2_DEBUG_ACCESS_KEY_ID`, a token for that bucket only) in `.env`. Without them the
pod skips the upload (`status.json` "debug_lite": {"uploaded": false}). On the PC: `python tools/pull_debug.py`
downloads everything, verifies it and deletes it from R2. Job option `"debug_lite": false` turns it off.
Its large parts are built in the background right after ReconstructMesh; the archive goes up beside `output.zip`,
both before `status.json`.

## Test from the laptop

```bash
cp .env.example .env       # fill in the keys and TURNTABLE_IMAGE
python tools/cloud_scan.py --dry_run    # preflight + estimate + the pod request (secrets redacted); no pod
python tools/cloud_scan.py              # full flow; prints timings, the pod used and the cost estimate
python tools/cloud_scan.py --keep_remote                 # leave jobs/<id>/ in R2
python tools/cloud_scan.py --keep_old_jobs               # keep earlier jobs in R2 (default: all jobs/ folders deleted first)
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
