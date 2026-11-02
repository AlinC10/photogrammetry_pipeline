"""
pull_debug.py — Local only: move the debug data of the scans from its own R2 bucket to this PC.

Downloads every object of the debug bucket (R2_DEBUG_BUCKET, never the bucket the Pi gets its models from) into
--dest, keeping the key layout (jobs/<id>/...), checks each file against the size R2 lists for it and only then
deletes it from the bucket. Objects whose download failed or does not match stay in R2 for the next run.

    python tools/pull_debug.py                 # list, ask, download, verify, delete
    python tools/pull_debug.py --dry-run       # only list what is there and how big it is
    python tools/pull_debug.py --keep-remote   # download without deleting
    python tools/pull_debug.py --yes --dest D:/turntable_debug

Keys come from .env: R2_ENDPOINT_URL plus R2_DEBUG_ACCESS_KEY_ID / R2_DEBUG_SECRET_ACCESS_KEY when set (a token
for the debug bucket only), otherwise R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY.
"""

import argparse
import logging
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from botocore.exceptions import BotoCoreError, ClientError  # noqa: E402

from cloud import cloudflare_r2 as r2  # noqa: E402

logger = logging.getLogger("TurntableStudio.PullDebug")

DELETE_CHUNK = 1000


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def check_bucket(bucket: str, delivery_bucket: str) -> None:
    """The debug bucket must be named and must not be the bucket the Pi downloads its models from."""
    if not (bucket or "").strip():
        raise r2.R2Error("No debug bucket: set R2_DEBUG_BUCKET in .env or pass --bucket.")
    if bucket == delivery_bucket:
        raise r2.R2Error(f"Refusing to empty '{bucket}': it is the delivery bucket (R2_BUCKET). "
                         "Debug data must live in its own bucket.")


def list_objects(client, bucket: str, prefix: str = "") -> Dict[str, int]:
    """key -> size in bytes of every object under prefix (paginated)."""
    out: Dict[str, int] = {}
    try:
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                out[obj["Key"]] = int(obj.get("Size", 0))
    except (ClientError, BotoCoreError) as e:
        raise r2.R2Error(f"Could not list bucket '{bucket}': {e}") from e
    return out


def local_path(dest: Path, key: str) -> Optional[Path]:
    """Where a key goes under dest; None for keys that would escape it (absolute, '..', empty, folder markers)."""
    p = PurePosixPath(key)
    if not key or key.endswith("/") or p.is_absolute() or ".." in p.parts or ":" in p.parts[0]:
        return None
    return dest.joinpath(*p.parts)


def pull(client, bucket: str, dest: Path, objects: Dict[str, int], delete: bool = True) -> Tuple[List[str], List[str]]:
    """
    Download (or reuse an identical local copy), verify the size, then delete the verified keys from the bucket.
    Returns (verified keys, failed keys). Failed keys are never deleted.
    """
    verified: List[str] = []
    failed: List[str] = []
    for i, (key, size) in enumerate(sorted(objects.items()), 1):
        target = local_path(dest, key)
        if target is None:
            logger.warning("Skipped unsafe key %r (left in the bucket).", key)
            failed.append(key)
            continue
        if target.is_file() and target.stat().st_size == size:
            verified.append(key)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        try:
            client.download_file(bucket, key, str(part))
            got = part.stat().st_size
            if got != size:
                raise OSError(f"size {got} B, R2 lists {size} B")
            os.replace(part, target)
        except (ClientError, BotoCoreError, OSError) as e:
            logger.error("Download of %s failed: %s (left in the bucket).", key, e)
            part.unlink(missing_ok=True)
            failed.append(key)
            continue
        verified.append(key)
        if i % 50 == 0:
            logger.info("  %d/%d", i, len(objects))

    if delete and verified:
        for start in range(0, len(verified), DELETE_CHUNK):
            chunk = verified[start:start + DELETE_CHUNK]
            try:
                resp = client.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in chunk],
                                                                    "Quiet": True})
            except (ClientError, BotoCoreError) as e:
                raise r2.R2Error(f"Downloaded, but could not delete from '{bucket}': {e}") from e
            for err in resp.get("Errors") or []:
                logger.error("Delete of %s failed: %s", err.get("Key"), err.get("Message", err.get("Code")))
    return verified, failed


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Download the scan debug data from its R2 bucket, then delete it there.")
    ap.add_argument("--bucket", default=os.environ.get("R2_DEBUG_BUCKET", ""), help="debug bucket (R2_DEBUG_BUCKET)")
    ap.add_argument("--dest", default=os.environ.get("TURNTABLE_DEBUG_DIR", str(Path.home() / "turntable_debug")),
                    help="local folder (default ~/turntable_debug or TURNTABLE_DEBUG_DIR)")
    ap.add_argument("--prefix", default="", help="only keys under this prefix (e.g. jobs/20261101-)")
    ap.add_argument("--dry-run", action="store_true", help="only list")
    ap.add_argument("--keep-remote", action="store_true", help="download without deleting from R2")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        check_bucket(args.bucket, r2.bucket_from_env())
        prefix = "R2_DEBUG_" if os.environ.get("R2_DEBUG_ACCESS_KEY_ID") else "R2_"
        client = r2.client_from_env(prefix)
        objects = list_objects(client, args.bucket, args.prefix)
    except r2.R2Error as e:
        logger.error("%s", e)
        return 2

    jobs = {k.split("/")[1] if k.startswith("jobs/") and k.count("/") > 1 else k.split("/")[0] for k in objects}
    logger.info("Bucket '%s': %d object(s), %s, %d scan(s).", args.bucket, len(objects),
                human(sum(objects.values())), len(jobs))
    if not objects or args.dry_run:
        return 0
    dest = Path(args.dest).expanduser().resolve()
    action = "download to" if args.keep_remote else "download to, then DELETE from R2 ->"
    if not args.yes and input(f"{action} {dest} ? [y/N] ").strip().lower() not in ("y", "yes", "da", "d"):
        logger.info("Nothing done.")
        return 0

    try:
        verified, failed = pull(client, args.bucket, dest, objects, delete=not args.keep_remote)
    except r2.R2Error as e:
        logger.error("%s", e)
        return 2
    logger.info("Done: %d file(s) in %s%s; %d failed (kept in R2).", len(verified), dest,
                "" if args.keep_remote else " and deleted from R2", len(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
