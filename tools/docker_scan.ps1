# docker_scan.ps1 — One scan with the Docker image on this Windows PC (GPU), photos and results on the local disk.
# Nothing goes through Cloudflare R2: without JOB_ID the container runs main.py on the mounted folders.
#
#   cd D:\photogrammetry_pipeline
#   powershell -ExecutionPolicy Bypass -File tools\docker_scan.ps1 -Name t1-implicit
#   powershell -ExecutionPolicy Bypass -File tools\docker_scan.ps1 -Name t2-fara-masti -Extra "--densify_masks","none"
#   powershell -ExecutionPolicy Bypass -File tools\docker_scan.ps1 -Name fixativ -InputDir D:\scanari\fixativ -Title "Fixativ Taft"
#
# Results: <repo>\teste\<Name>\ (OBJ, GLB, STL, 3MF, object.json, stats.json, run_log.txt).
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [string]$InputDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "input_images"),
    [string]$OutputDir = "",
    [string]$Image = "alinc10/turntable-studio:v4",
    [string]$Title = "",
    [string[]]$Extra = @()
)
if (-not $OutputDir) { $OutputDir = Join-Path (Join-Path (Split-Path $PSScriptRoot -Parent) "teste") $Name }
if (-not (Test-Path (Join-Path $InputDir "rig"))) { Write-Error "No rig\0 and rig\1 in $InputDir"; exit 1 }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
if (-not $Title) { $Title = $Name }

$args_main = @("--input_dir", "/workspace/input_images", "--output_dir", "/workspace/output", "--rig_mode",
               "--estimate_lift", "--title", $Title) + $Extra
Write-Host "Scan $Name : $InputDir -> $OutputDir ($Image) $($Extra -join ' ')"
$t0 = Get-Date
docker run --rm --gpus all `
    -v "${InputDir}:/workspace/input_images:ro" `
    -v "${OutputDir}:/workspace/output" `
    -v turntable_work:/tmp/turntable_work `
    $Image @args_main
$code = $LASTEXITCODE
Write-Host ("Done in {0:N1} min, exit code {1}. Results: {2}" -f ((Get-Date) - $t0).TotalMinutes, $code, $OutputDir)
exit $code
