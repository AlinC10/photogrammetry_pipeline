"""
mesh_cleaner.py — Watertight base slicing, prismatic skirt extrusion,
2D Ear-Clipping base capping (-Y normal), and PyMeshLab manifold validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

from .config import MeshCleaningConfig

logger = logging.getLogger("TurntableStudio.MeshCleaner")


def triangulate_2d_polygon(coords_2d: np.ndarray) -> np.ndarray:
    """
    Triangulates a 2D simple polygon using mapbox-earcut or shapely/scipy fallback.
    Returns array of triangle indices with shape (M, 3).
    """
    try:
        import earcut
        # coords_2d is (N, 2)
        flat_coords = coords_2d.flatten().tolist()
        tri_indices = earcut.earcut(flat_coords, dim=2)
        return np.array(tri_indices, dtype=np.int64).reshape(-1, 3)
    except ImportError:
        pass

    try:
        import mapbox_earcut
        tri_indices = mapbox_earcut.triangulate_float64(coords_2d, np.array([len(coords_2d)]))
        return tri_indices.reshape(-1, 3)
    except ImportError:
        pass

    # Shapely polygon triangulation fallback
    try:
        from shapely.geometry import Polygon
        from shapely.ops import triangulate
        poly = Polygon(coords_2d)
        if not poly.is_valid:
            poly = poly.buffer(0)
        triangles = triangulate(poly)
        # Match vertices to indices
        faces = []
        for tri in triangles:
            pts = list(tri.exterior.coords)[:3]
            face_idx = []
            for p in pts:
                dists = np.linalg.norm(coords_2d - np.array([p[0], p[1]]), axis=1)
                face_idx.append(int(np.argmin(dists)))
            faces.append(face_idx)
        return np.array(faces, dtype=np.int64)
    except Exception as e:
        logger.warning(f"Ear-clipping fallback failed: {e}. Using fan triangulation.")
        n = len(coords_2d)
        return np.array([[0, i, i + 1] for i in range(1, n - 1)], dtype=np.int64)


def seal_watertight_pedestal(
    input_mesh: trimesh.Trimesh,
    slice_y: float = 0.003,
    skirt_y: float = 0.000,
    max_radius: float = 0.12,
) -> trimesh.Trimesh:
    """
    Slices mesh at slice_y (dropping noisy turntable points), finds boundary loops,
    extrudes a prismatic skirt down to skirt_y, and triangulates the bottom cap
    with strict -Y normal winding.
    """
    logger.info(f"Slicing mesh at Y = {slice_y * 1000:.1f} mm, skirt down to Y = {skirt_y * 1000:.1f} mm...")

    # 1. Planar slice keeping everything ABOVE slice_y
    # Plane normal points UP (+Y), origin at [0, slice_y, 0]
    sliced = trimesh.intersections.slice_mesh_plane(
        mesh=input_mesh,
        plane_normal=[0.0, 1.0, 0.0],
        plane_origin=[0.0, slice_y, 0.0],
        cap=False,
    )

    if not isinstance(sliced, trimesh.Trimesh) or len(sliced.faces) == 0:
        logger.warning("Planar slice resulted in empty mesh! Returning original.")
        return input_mesh

    # Merge duplicate coincident vertices created by the planar cut
    sliced.merge_vertices()

    # 2. Extract boundary edges (edges with only 1 adjacent face)
    # trimesh edges_unique and edges_face count
    edges = sliced.edges_sorted
    edge_counts = {}
    for e in edges:
        tup = (e[0], e[1])
        edge_counts[tup] = edge_counts.get(tup, 0) + 1

    boundary_edges = [np.array(e) for e, count in edge_counts.items() if count == 1]
    if not boundary_edges:
        logger.info("Mesh has no open boundaries after slicing (already watertight).")
        return sliced

    # Filter boundary edges close to the slice plane (Y ≈ slice_y)
    v = sliced.vertices
    filtered_b_edges = []
    for e in boundary_edges:
        if abs(v[e[0], 1] - slice_y) < 0.002 and abs(v[e[1], 1] - slice_y) < 0.002:
            r0 = np.hypot(v[e[0], 0], v[e[0], 2])
            r1 = np.hypot(v[e[1], 0], v[e[1], 2])
            if r0 <= max_radius and r1 <= max_radius:
                filtered_b_edges.append(e)

    if not filtered_b_edges:
        logger.info("No open boundary loops found near slice plane.")
        return sliced

    # Chain boundary edges into ordered loops
    # Build adjacency dict
    adj = {}
    for e in filtered_b_edges:
        adj.setdefault(e[0], []).append(e[1])
        adj.setdefault(e[1], []).append(e[0])

    visited = set()
    loops = []
    for start_node in adj:
        if start_node in visited:
            continue
        loop = [start_node]
        visited.add(start_node)
        curr = start_node
        while True:
            neighbors = [n for n in adj[curr] if n not in visited]
            if not neighbors:
                break
            nxt = neighbors[0]
            loop.append(nxt)
            visited.add(nxt)
            curr = nxt
        if len(loop) >= 3:
            loops.append(loop)

    if not loops:
        logger.info("Could not form closed boundary loop, returning sliced mesh.")
        return sliced

    # Select largest boundary loop (by vertex count)
    primary_loop = max(loops, key=len)
    loop_v_idx = np.array(primary_loop)
    n_loop = len(loop_v_idx)
    logger.info(f"Identified primary boundary loop with {n_loop} vertices.")

    # 3. Create skirt vertices and faces
    new_vertices = list(sliced.vertices.copy())
    new_faces = list(sliced.faces.copy())

    # Skirt bottom vertices at y = skirt_y
    skirt_bottom_indices = []
    base_coords_2d = []

    for idx in loop_v_idx:
        vx, vy, vz = sliced.vertices[idx]
        skirt_v = np.array([vx, skirt_y, vz], dtype=np.float64)
        new_idx = len(new_vertices)
        new_vertices.append(skirt_v)
        skirt_bottom_indices.append(new_idx)
        base_coords_2d.append([vx, vz])

    skirt_bottom_indices = np.array(skirt_bottom_indices)
    base_coords_2d = np.array(base_coords_2d)

    # Build vertical quad walls between slice_y and skirt_y
    for i in range(n_loop):
        top_curr = loop_v_idx[i]
        top_next = loop_v_idx[(i + 1) % n_loop]
        bot_curr = skirt_bottom_indices[i]
        bot_next = skirt_bottom_indices[(i + 1) % n_loop]

        # Two triangles per quad
        f1 = [top_curr, top_next, bot_next]
        f2 = [top_curr, bot_next, bot_curr]

        # Check normal direction of quad faces (should point radially outwards)
        v0, v1, v2 = new_vertices[f1[0]], new_vertices[f1[1]], new_vertices[f1[2]]
        face_norm = np.cross(v1 - v0, v2 - v0)
        centroid = (v0 + v1 + v2) / 3.0
        # Radially outwards vector: [centroid[0], 0, centroid[2]]
        radial_vec = np.array([centroid[0], 0.0, centroid[2]])
        if np.dot(face_norm, radial_vec) < 0:
            # Swap winding
            f1 = [top_curr, bot_next, top_next]
            f2 = [top_curr, bot_curr, bot_next]

        new_faces.append(f1)
        new_faces.append(f2)

    # 4. Triangulate bottom base cap at y = skirt_y
    tri_2d_faces = triangulate_2d_polygon(base_coords_2d)
    for tri in tri_2d_faces:
        idx0 = skirt_bottom_indices[tri[0]]
        idx1 = skirt_bottom_indices[tri[1]]
        idx2 = skirt_bottom_indices[tri[2]]

        # Normal must point DOWN (-Y)
        v0 = new_vertices[idx0]
        v1 = new_vertices[idx1]
        v2 = new_vertices[idx2]

        norm = np.cross(v1 - v0, v2 - v0)
        if norm[1] > 0:
            # Winding points UP (+Y), swap to point DOWN (-Y)
            new_faces.append([idx0, idx2, idx1])
        else:
            new_faces.append([idx0, idx1, idx2])

    assembled_mesh = trimesh.Trimesh(
        vertices=np.array(new_vertices),
        faces=np.array(new_faces),
        process=True,
    )
    logger.info(
        f"Sealed mesh created: {len(assembled_mesh.vertices):,d} vertices, "
        f"{len(assembled_mesh.faces):,d} faces (Watertight: {assembled_mesh.is_watertight})"
    )
    return assembled_mesh


def clean_and_repair_mesh(
    input_obj_path: str,
    output_obj_path: str,
    config: MeshCleaningConfig,
) -> Dict[str, Any]:
    """
    Cleans raw photogrammetry mesh using trimesh and PyMeshLab:
    1. Removes isolated noisy fragments (keeps largest component).
    2. Slices base and seals with watertight skirt and flat base.
    3. Runs PyMeshLab manifold repair, hole closing, and metric checks.
    """
    in_p = Path(input_obj_path).resolve()
    out_p = Path(output_obj_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading mesh for cleaning: {in_p}")
    raw_mesh = trimesh.load(str(in_p), process=False)

    # 1. Keep largest connected component if requested
    if config.keep_largest_only and hasattr(raw_mesh, "split"):
        components = raw_mesh.split(only_watertight=False)
        if len(components) > 1:
            raw_mesh = max(components, key=lambda m: len(m.faces))
            logger.info(f"Kept largest component with {len(raw_mesh.faces):,d} faces.")

    # 2. Watertight pedestal sealing
    sealed = seal_watertight_pedestal(
        input_mesh=raw_mesh,
        slice_y=config.slice_y_m,
        skirt_y=config.skirt_bottom_y_m,
        max_radius=config.max_radius_m,
    )

    # Export intermediate to temp for PyMeshLab
    temp_obj = out_p.parent / f"_temp_{out_p.name}"
    sealed.export(str(temp_obj))

    # 3. PyMeshLab manifold repair
    is_watertight = sealed.is_watertight
    volume_cm3 = None

    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(temp_obj))

        logger.info("Running PyMeshLab manifold repair filters...")
        if config.remove_duplicate_faces:
            ms.meshing_remove_duplicate_faces()
            ms.meshing_remove_duplicate_vertices()

        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()
        ms.meshing_close_holes(maxholesize=30)

        # Save repaired mesh
        ms.save_current_mesh(str(out_p))

        # Check metrics
        curr = ms.current_mesh()
        v_count = curr.vertex_number()
        f_count = curr.face_number()

        # Re-check watertightness with trimesh
        final_check = trimesh.load(str(out_p), process=False)
        is_watertight = bool(final_check.is_watertight)
        if is_watertight and final_check.volume is not None:
            volume_cm3 = round(float(final_check.volume) * 1e6, 2)

    except Exception as e:
        logger.warning(f"PyMeshLab processing warning: {e}. Falling back to trimesh export.")
        sealed.export(str(out_p))
        v_count = len(sealed.vertices)
        f_count = len(sealed.faces)

    finally:
        if temp_obj.is_file():
            temp_obj.unlink()

    logger.info(
        f"Mesh cleaning completed: {out_p.name} | {v_count:,d} vertices, {f_count:,d} faces | "
        f"Watertight: {is_watertight} | Volume: {volume_cm3} cm³"
    )

    return {
        "output_path": str(out_p),
        "vertex_count": v_count,
        "face_count": f_count,
        "is_watertight": is_watertight,
        "volume_cm3": volume_cm3,
    }
