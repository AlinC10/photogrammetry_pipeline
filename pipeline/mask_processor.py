"""
mask_processor.py — Mask management and formatting for OpenMVS.
Ensures masks are named as <image_name>.mask.png in the masks directory.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("TurntableStudio.MaskProcessor")


def discover_masks(
    masks_dir: str,
    images_dir: str,
) -> Dict[str, str]:
    """
    Finds existing masks and associates them with corresponding images.
    Returns mapping of image filename -> mask file path.
    """
    m_dir = Path(masks_dir).resolve()
    i_dir = Path(images_dir).resolve()

    if not m_dir.is_dir():
        logger.info(f"Masks directory not found: {m_dir}")
        return {}

    image_files = [
        f for f in i_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"} and not f.name.endswith(".mask.png")
    ]

    mask_map = {}
    for img in image_files:
        stem = img.stem
        # Look for various naming schemes
        candidates = [
            m_dir / f"{stem}.mask.png",
            m_dir / f"{stem}.png",
            m_dir / f"{img.name}.mask.png",
            m_dir / f"{stem}_mask.png",
            m_dir / f"{img.name}",
        ]
        for c in candidates:
            if c.is_file():
                mask_map[img.name] = str(c.resolve())
                break

    logger.info(f"Discovered {len(mask_map)} / {len(image_files)} masks in {m_dir}")
    return mask_map


def prepare_openmvs_masks(
    mask_map: Dict[str, str],
    target_masks_dir: str,
) -> str:
    """
    Copies or formats masks into target_masks_dir with the strict OpenMVS naming convention:
    `<image_name>.mask.png`.
    Ensures masks are 8-bit single-channel binary images (0 or 255).
    """
    target_dir = Path(target_masks_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    prepared_count = 0
    for img_name, mask_src in mask_map.items():
        dst_name = f"{img_name}.mask.png"
        dst_path = target_dir / dst_name

        mask = cv2.imread(mask_src, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning(f"Failed to read mask image: {mask_src}")
            continue

        # Ensure pure binary mask (0 = background, 255 = object)
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(str(dst_path), binary_mask)
        prepared_count += 1

    logger.info(f"Prepared {prepared_count} OpenMVS masks in {target_dir}")
    return str(target_dir)
