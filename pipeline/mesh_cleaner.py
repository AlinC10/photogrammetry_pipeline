"""
mesh_cleaner.py — Watertight base slicing, prismatic skirt extrusion,
2D Ear-Clipping base capping (-Y normal), and PyMeshLab manifold validation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


# -----------------------------------------------------------------------------
# Components, boundary loops, base cap
# -----------------------------------------------------------------------------

def component_report(mesh: trimesh.Trimesh, slice_y: float, rim_radius: float) -> List[Dict[str, Any]]:
    """Connected pieces with their size, radius and height (the score = vertices above the cut inside rim_radius)."""
    out = []
    for piece in mesh.split(only_watertight=False):
        v = piece.vertices
        r = np.hypot(v[:, 0], v[:, 2])
        out.append({
            "mesh": piece,
            "faces": int(len(piece.faces)),
            "score": int(((v[:, 1] > slice_y + 0.0005) & (r < rim_radius)).sum()),
            "r_median_mm": round(float(np.median(r)) * 1000, 1),
            "y_min_mm": round(float(v[:, 1].min()) * 1000, 1),
            "y_max_mm": round(float(v[:, 1].max()) * 1000, 1),
        })
    return out


def select_object_components(mesh: trimesh.Trimesh, slice_y: float, rim_radius: float = 0.115,
                             keep_ratio: float = 0.25, label: str = "") -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    """
    The object by position, not by size (G1/P7 of the 01.11 audit): a piece counts by its vertices ABOVE the cut and
    inside `rim_radius`. The disc (flat, under the cut), its rim (radius 118-127 mm) and floating specks score low;
    the piece with the best score is kept, plus any other with >= keep_ratio of it (a second part of the object, or a
    second object on the disc). The orange piece 094944 lost its object this way: the disc was the largest piece.
    Returns (kept mesh, info with the dropped pieces).
    """
    parts = component_report(mesh, slice_y, rim_radius)
    info: Dict[str, Any] = {"pieces": len(parts), "faces_in": int(len(mesh.faces))}
    if len(parts) <= 1:
        info.update(kept=len(parts), faces_kept=int(len(mesh.faces)), dropped=[])
        return mesh, info
    best = max(p["score"] for p in parts)
    if best <= 0:
        keep = [max(parts, key=lambda p: p["faces"])]
        info["no_piece_above_cut"] = True
    else:
        keep = [p for p in parts if p["score"] >= keep_ratio * best]
    dropped = sorted((p for p in parts if not any(p is k for k in keep)), key=lambda p: -p["faces"])
    kept = keep[0]["mesh"] if len(keep) == 1 else trimesh.util.concatenate([k["mesh"] for k in keep])
    info.update(kept=len(keep), faces_kept=int(len(kept.faces)),
                kept_pieces=[{k: p[k] for k in ("faces", "score", "r_median_mm", "y_min_mm", "y_max_mm")} for p in keep],
                dropped=[{k: p[k] for k in ("faces", "score", "r_median_mm", "y_min_mm", "y_max_mm")}
                         for p in dropped[:10]],
                faces_dropped=int(sum(p["faces"] for p in dropped)))
    if dropped:
        big = ", ".join(f"{p['faces']:,d} faces at r {p['r_median_mm']} mm, y {p['y_min_mm']}-{p['y_max_mm']} mm"
                        for p in dropped[:3])
        logger.info(f"{label}kept {len(keep)} of {len(parts)} pieces; dropped {len(dropped)} "
                    f"({info['faces_dropped']:,d} faces; largest: {big}).")
    return kept, info


def boundary_loops(mesh: trimesh.Trimesh) -> Tuple[List[np.ndarray], int]:
    """
    Closed boundary loops as vertex index arrays, chained along the directed boundary half-edges (each boundary edge
    taken in the direction of its face, so a loop never jumps to a neighbouring hole the way "first unvisited
    neighbour" chaining did). Returns (loops, open paths that could not be closed).
    """
    if len(mesh.faces) == 0:
        return [], 0
    groups = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    if len(groups) == 0:
        return [], 0
    he = mesh.edges[np.asarray(groups).ravel()]
    nxt: Dict[int, List[int]] = {}
    for i, (a, _) in enumerate(he):
        nxt.setdefault(int(a), []).append(i)
    used = np.zeros(len(he), bool)
    loops: List[np.ndarray] = []
    open_paths = 0
    for i0 in range(len(he)):
        if used[i0]:
            continue
        start = int(he[i0, 0])
        loop = [start]
        i = i0
        closed = False
        while True:
            used[i] = True
            b = int(he[i, 1])
            if b == start:
                closed = True
                break
            loop.append(b)
            cand = [j for j in nxt.get(b, []) if not used[j]]
            if not cand:
                break
            i = cand[0]
        if closed and len(loop) >= 3:
            loops.append(np.array(loop, dtype=np.int64))
        elif not closed:
            open_paths += 1
    return loops, open_paths


def _triangulate_rings(rings_2d: List[np.ndarray]) -> np.ndarray:
    """Triangles (indices into the stacked rings) of a polygon: rings_2d[0] = outline, the rest = holes."""
    verts = np.vstack(rings_2d).astype(np.float64)
    ends = np.cumsum([len(r) for r in rings_2d]).astype(np.uint32)
    try:
        import mapbox_earcut
        return np.asarray(mapbox_earcut.triangulate_float64(verts, ends), dtype=np.int64).reshape(-1, 3)
    except ImportError:
        if len(rings_2d) == 1:
            return triangulate_2d_polygon(rings_2d[0])
        raise


def plan_base_caps(vertices: np.ndarray, loops: List[np.ndarray], slice_y: float, max_radius: float,
                   tol: float = 1e-6) -> Tuple[List[List[np.ndarray]], Dict[str, int]]:
    """
    Groups the loops lying in the cut plane into polygons with holes (even-odd nesting in x, z): an outline with
    the loops directly inside it as holes, islands inside holes as outlines again (a ring-shaped foot, three legs,
    a mug standing on its rim). Every such loop is capped, not only the largest one (G1). Loops only partly in the
    plane (a hole reaching the cut) are counted, not capped: the hole filling closes them.
    Returns ([[outline, hole, ...], ...] as vertex index arrays, counts).
    """
    from shapely.geometry import Point, Polygon

    flat, partial = [], 0
    for lp in loops:
        on = np.abs(vertices[lp, 1] - slice_y) < tol
        if on.all():
            if np.hypot(vertices[lp, 0], vertices[lp, 2]).max() <= max_radius:
                flat.append(lp)
        elif on.sum() >= 2:
            partial += 1
    polys = []
    for lp in flat:
        pg = Polygon(vertices[lp][:, [0, 2]])
        if not pg.is_valid:
            pg = pg.buffer(0)
        polys.append(pg)
    areas = [p.area for p in polys]
    depth, parent = [], []
    for i, lp in enumerate(flat):
        pt = Point(vertices[lp[0], 0], vertices[lp[0], 2])
        inside = [j for j in range(len(flat)) if j != i and areas[j] > areas[i] and polys[j].contains(pt)]
        depth.append(len(inside))
        parent.append(min(inside, key=lambda j: areas[j]) if inside else -1)
    groups: List[List[np.ndarray]] = []
    index = {}
    for i in sorted(range(len(flat)), key=lambda k: depth[k]):
        if depth[i] % 2 == 0:
            index[i] = len(groups)
            groups.append([flat[i]])
        elif parent[i] in index:
            groups[index[parent[i]]].append(flat[i])
    return groups, {"loops": len(loops), "flat_loops": len(flat), "partial_loops": partial,
                    "polygons": len(groups), "holes_in_polygons": sum(len(g) - 1 for g in groups)}


def seal_watertight_pedestal(
    input_mesh: trimesh.Trimesh,
    slice_y: float = 0.001,
    skirt_y: float = 0.001,
    max_radius: float = 0.12,
    extrude_skirt: Optional[bool] = None,
    base_color_rgb: Tuple[int, int, int] = (25, 25, 25),
    rim_radius: float = 0.115,
    report: Optional[Dict[str, Any]] = None,
) -> trimesh.Trimesh:
    """
    Slices the mesh at slice_y (dropping the turntable), keeps the object pieces (select_object_components),
    optionally extrudes a vertical skirt down to skirt_y under every base loop, and caps all base loops (polygons
    with holes, ear clipping) with the normals strictly along -Y. `report` (optional dict) receives the numbers.
    """
    if extrude_skirt is None:
        extrude_skirt = (skirt_y < (slice_y - 1e-4))
    report = report if report is not None else {}

    logger.info(
        f"Slicing mesh at Y = {slice_y * 1000:.1f} mm "
        f"({'extruding skirt down to Y = ' + str(skirt_y * 1000) + ' mm' if extrude_skirt else 'direct base cap without skirt'})..."
    )

    # 1. Planar slice keeping everything ABOVE slice_y
    sliced = trimesh.intersections.slice_mesh_plane(
        mesh=input_mesh,
        plane_normal=[0.0, 1.0, 0.0],
        plane_origin=[0.0, slice_y, 0.0],
        cap=False,
    )

    if not isinstance(sliced, trimesh.Trimesh) or len(sliced.faces) == 0:
        logger.warning("Planar slice resulted in empty mesh! Returning original.")
        report["empty_after_cut"] = True
        return input_mesh

    # Merge duplicate coincident vertices created by the planar cut; snap the cut vertices onto the plane exactly
    sliced.merge_vertices()
    report["faces_after_cut"] = int(len(sliced.faces))

    # The disc rim rises above the cut: before slicing it hangs on the object through the disc, after it the rim
    # pieces are separate components (multimeter: 7 pieces at radius 118-127 mm)
    sliced, comp = select_object_components(sliced, slice_y, rim_radius, label="After the cut: ")
    report["components_after_cut"] = comp

    # 2. Closed boundary loops in the cut plane, grouped into polygons with holes
    loops, open_paths = boundary_loops(sliced)
    groups, counts = plan_base_caps(sliced.vertices, loops, slice_y, max_radius)
    counts["open_paths"] = open_paths
    report["base"] = counts
    if not groups:
        logger.info(f"No closed boundary loop in the cut plane ({counts}).")
        return sliced
    logger.info(f"Base: {counts['polygons']} polygon(s) with {counts['holes_in_polygons']} hole(s) from "
                f"{counts['flat_loops']} loop(s) in the cut plane ({counts['partial_loops']} loop(s) only partly in it, "
                f"left to the hole filling).")

    vertices = [sliced.vertices.copy()]
    n_vert = len(sliced.vertices)
    new_faces = [sliced.faces.copy()]
    cap_faces = []
    for rings in groups:
        if extrude_skirt and abs(slice_y - skirt_y) > 1e-4:
            bottom_rings = []
            for lp in rings:
                bot = np.arange(n_vert, n_vert + len(lp))
                v = sliced.vertices[lp].copy()
                v[:, 1] = skirt_y
                vertices.append(v)
                n_vert += len(lp)
                # The cut edge a -> b belongs to a face as a -> b: the wall face takes it as b -> a
                a, b = lp, np.roll(lp, -1)
                ab, bb = bot, np.roll(bot, -1)
                new_faces.append(np.stack([b, a, ab], 1))
                new_faces.append(np.stack([b, ab, bb], 1))
                bottom_rings.append(bot)
            cap_rings = bottom_rings
        else:
            cap_rings = rings
        all_v = np.vstack(vertices)
        tri = _triangulate_rings([all_v[r][:, [0, 2]] for r in cap_rings])
        ids = np.concatenate(cap_rings)[tri]
        cap_faces.append(ids)
    all_v = np.vstack(vertices)
    caps = np.vstack(cap_faces) if cap_faces else np.zeros((0, 3), np.int64)
    if len(caps):
        # Normal must point DOWN (-Y)
        n = np.cross(all_v[caps[:, 1]] - all_v[caps[:, 0]], all_v[caps[:, 2]] - all_v[caps[:, 0]])
        flip = n[:, 1] > 0
        caps[flip] = caps[flip][:, [0, 2, 1]]
    assembled_mesh = trimesh.Trimesh(vertices=all_v, faces=np.vstack(new_faces + [caps]), process=True)
    # Ear clipping drops (nearly) collinear points of the cut outline: each leaves a sliver gap of 3-6 vertices
    # in the cap plane. Fill those with a fan taken against the direction of their border (consistent winding).
    cap_y = skirt_y if extrude_skirt and abs(slice_y - skirt_y) > 1e-4 else slice_y
    rest, _ = boundary_loops(assembled_mesh)
    slivers = [lp[::-1] for lp in rest
               if len(lp) <= 8 and np.all(np.abs(assembled_mesh.vertices[lp, 1] - cap_y) < 1e-6)]
    if slivers:
        fans = np.array([[lp[0], lp[i], lp[i + 1]] for lp in slivers for i in range(1, len(lp) - 1)], np.int64)
        assembled_mesh = trimesh.Trimesh(vertices=assembled_mesh.vertices,
                                         faces=np.vstack([assembled_mesh.faces, fans]), process=False)
        report["cap_slivers_filled"] = len(slivers)
    report["cap_faces"] = int(len(caps))
    logger.info(
        f"Sealed mesh created: {len(assembled_mesh.vertices):,d} vertices, "
        f"{len(assembled_mesh.faces):,d} faces (Watertight: {assembled_mesh.is_watertight})"
    )
    return assembled_mesh


def face_target(faces: int, ratio: float, fixed: int = 0, minimum: int = 1000, maximum: int = 0) -> int:
    """
    Face count to decimate to: `fixed` when > 0, otherwise `ratio` x `faces`, capped at `maximum` (> 0), never
    below `minimum` and never above `faces`.
    """
    if fixed and fixed > 0:
        return int(fixed)
    target = round(faces * float(ratio))
    if maximum and maximum > 0:
        target = min(target, maximum)
    return int(min(faces, max(minimum, target)))


def _pml_boundary_edges(ms) -> int:
    try:
        return int(ms.get_topological_measures().get("boundary_edges", 0))
    except Exception:
        return -1


def _pml_non_manifold(ms) -> int:
    try:
        return int(ms.get_topological_measures().get("non_two_manifold_edges", 0))
    except Exception:
        return -1


def _selected_area_mm2(ms) -> float:
    m = ms.current_mesh()
    sel = m.face_selection_array()
    if not sel.any():
        return 0.0
    V, F = m.vertex_matrix(), m.face_matrix()[sel]
    return float(0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum())


def _separate_coincident_vertices(ms, offset_mm: float = 0.02) -> int:
    """
    The non-manifold vertex repair splits a pinch point (two sheets touching) into copies at the same position: the
    mesh is manifold, but an STL stores positions only and every reader merges them back into the pinch. Each copy
    is moved `offset_mm` towards the centroid of its own faces, so the sheets no longer share a point.
    Returns the number of vertices moved.
    """
    import pymeshlab

    m = ms.current_mesh()
    V, F = m.vertex_matrix(), m.face_matrix()
    if not len(V):
        return 0
    _, inv, counts = np.unique(np.round(V, 6), axis=0, return_inverse=True, return_counts=True)
    dup = np.flatnonzero(counts[inv.ravel()] > 1)
    if not len(dup):
        return 0
    centroid = V[F].mean(axis=1)
    acc = np.zeros_like(V)
    n = np.zeros(len(V))
    for k in range(3):
        np.add.at(acc, F[:, k], centroid)
        np.add.at(n, F[:, k], 1)
    target = acc[dup] / np.maximum(n[dup], 1)[:, None]
    d = target - V[dup]
    norm = np.linalg.norm(d, axis=1)
    ok = norm > 1e-9
    V = V.copy()
    V[dup[ok]] += d[ok] / norm[ok, None] * np.minimum(offset_mm, 0.5 * norm[ok])[:, None]
    old = ms.current_mesh_id()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=V, face_matrix=F))
    new = ms.current_mesh_id()
    ms.set_current_mesh(old)
    ms.delete_current_mesh()
    ms.set_current_mesh(new)
    return int(ok.sum())


def fill_holes_with_fairing(ms, max_hole_edges: int = 2000, fairing_steps: int = 5,
                            patch_hook=None) -> Dict[str, Any]:
    """
    Closes the holes of the mesh in `ms` (PyMeshLab, millimetres), once, on the high-poly mesh:
    1. holes up to max_hole_edges edges, with the self-intersection check, the patch remeshed to the median edge
       length of the mesh (only the patch: the refinehole option of close_holes cost ~20 s per call on 0.25 M faces);
    2. what is still open, without the check (large holes at the base stay open with it, 01.11 test: 2/8 models
       watertight with it, 7/8 without);
    after each pass a few Laplacian steps on the new faces only (fairing: the patch follows its border smoothly,
    Liepa 2003) and `patch_hook(ms)` (new faces selected; e.g. snap them to the visual hull). Then non-manifold
    repair and a last pass for the small holes it opens. Poisson was tested and rejected (watertight 3/8, bulges
    up to 34 mm). Returns the boundary edges before / after and the patched area (mm2).
    """
    info: Dict[str, Any] = {"boundary_edges_before": _pml_boundary_edges(ms), "patched_area_mm2": 0.0, "passes": []}
    if info["boundary_edges_before"] == 0:
        info["boundary_edges_after"] = 0
        return info
    # Median edge length of the mesh: the patches are remeshed to it (refinehole=True of close_holes did the same
    # but cost ~20 s per call on 0.25 M faces even with nothing to close, ~7 min on the 2 M-face remote control)
    m0 = ms.current_mesh()
    V0, F0 = m0.vertex_matrix(), m0.face_matrix()
    sample = F0[:: max(1, len(F0) // 200000)]
    edge_len = float(np.median(np.linalg.norm(V0[sample[:, 0]] - V0[sample[:, 1]], axis=1))) if len(sample) else 0.0
    del V0, F0, sample
    for selfint in (True, False):
        ms.set_selection_none()
        t_pass = time.time()
        try:
            ms.meshing_close_holes(maxholesize=int(max_hole_edges), newfaceselected=True,
                                   selfintersection=selfint, refinehole=False)
        except Exception as e:
            logger.warning(f"close_holes (self-intersection check {selfint}) failed: {e}")
            continue
        new = int(ms.current_mesh().selected_face_number())
        area = _selected_area_mm2(ms)
        if new and edge_len > 0:
            # Only the patch is remeshed (its border stays): small triangles the fairing and the hull can shape
            try:
                import pymeshlab
                ms.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pymeshlab.PureValue(edge_len),
                                                       selectedonly=True, checksurfdist=False, featuredeg=180)
            except Exception as e:
                logger.warning(f"Patch refinement failed: {e}")
        if new and fairing_steps > 0:
            try:
                ms.apply_coord_laplacian_smoothing(stepsmoothnum=int(fairing_steps), selected=True)
            except Exception as e:
                logger.warning(f"Patch fairing failed: {e}")
        if new and patch_hook is not None:
            try:
                patch_hook(ms)
            except Exception as e:
                logger.warning(f"Patch hook failed: {e}")
        info["passes"].append({"self_intersection_check": selfint, "new_faces": new, "area_mm2": round(area, 1),
                               "seconds": round(time.time() - t_pass, 1)})
        info["patched_area_mm2"] += area
        if _pml_boundary_edges(ms) == 0:
            break
    # A patch without the self-intersection check can leave non-manifold edges (no open edge, still not a closed
    # solid: the flat phone 093314). Repairing them removes faces and opens small holes: repair and close again,
    # until the mesh is both closed and manifold (at most 3 rounds)
    t_rep = time.time()
    for _ in range(3):
        ms.set_selection_none()
        ms.meshing_remove_duplicate_vertices()     # coincident patch vertices: an STL reader merges them
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()
        if _pml_boundary_edges(ms) != 0:
            ms.meshing_close_holes(maxholesize=100, newfaceselected=False, selfintersection=False)
        if _pml_boundary_edges(ms) == 0 and _pml_non_manifold(ms) == 0:
            break
    ms.set_selection_none()
    info["split_vertices_separated"] = _separate_coincident_vertices(ms)
    # Patches closed without the self-intersection check can come out facing inwards: one coherent orientation,
    # outwards (positive volume)
    try:
        ms.meshing_re_orient_faces_coherently()
        m = ms.current_mesh()
        V, F = m.vertex_matrix(), m.face_matrix()
        if len(F) and np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() < 0:
            ms.meshing_invert_face_orientation(forceflip=True)
    except Exception as e:
        logger.warning(f"Re-orienting the faces failed: {e}")
    info["repair_seconds"] = round(time.time() - t_rep, 1)
    info["non_manifold_edges_after"] = _pml_non_manifold(ms)
    info["boundary_edges_after"] = _pml_boundary_edges(ms)
    info["patched_area_mm2"] = round(info["patched_area_mm2"], 1)
    return info


def _close_base_band_holes(mesh: trimesh.Trimesh, slice_y: float, band_m: float, max_hole_edges: int
                           ) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    """
    Before the cut: closes the holes whose border lies in the band just above it (slice_y .. band_m). A hole that
    reaches the cut plane would otherwise open the base outline into an open path that no cap can close (G2: the
    chord across the base). Cheap: no refinement, no self-intersection check, only holes touching the band.
    """
    info: Dict[str, Any] = {}
    try:
        import pymeshlab
    except ImportError:
        return mesh, info
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(mesh.vertices, np.float64) * 1000.0,
                               face_matrix=np.asarray(mesh.faces, np.int32)))
    lo, hi = slice_y * 1000.0, band_m * 1000.0
    try:
        ms.compute_selection_by_condition_per_face(
            condselect=f"((y0 > {lo}) && (y0 < {hi})) || ((y1 > {lo}) && (y1 < {hi})) || ((y2 > {lo}) && (y2 < {hi}))")
        if ms.current_mesh().selected_face_number() == 0:
            return mesh, info
        before = ms.current_mesh().face_number()
        ms.meshing_close_holes(maxholesize=int(max_hole_edges), selected=True, newfaceselected=False,
                               selfintersection=False, refinehole=False)
        m = ms.current_mesh()
        info["faces_added"] = int(m.face_number() - before)
    except Exception as e:
        logger.warning(f"Closing the holes near the base failed ({e}); cutting the mesh as it is.")
        return mesh, info
    if info["faces_added"] <= 0:
        return mesh, info
    logger.info(f"Closed the holes near the base before the cut: {info['faces_added']:,d} faces added.")
    return trimesh.Trimesh(vertices=m.vertex_matrix() / 1000.0, faces=m.face_matrix(), process=True), info


def mesh_warnings(res: Dict[str, Any], config: MeshCleaningConfig) -> List[Dict[str, Any]]:
    """Problems of the cleaned mesh the client should see (status.json "warnings")."""
    out = []
    if res.get("is_watertight") is False:
        out.append({"code": "not_watertight", "message": "The model is not closed (holes left after the filling): "
                    "the STL / 3MF may need repair in the slicer.",
                    "boundary_edges": (res.get("holes") or {}).get("boundary_edges_after")})
    ext = res.get("extent_mm")
    if ext and max(ext) < 10.0:
        out.append({"code": "model_tiny", "message": f"The model is only {max(ext):.1f} mm: the object was probably "
                    f"lost (masks or reconstruction); rescan.", "extent_mm": ext})
    comp = res.get("components") or {}
    kept = comp.get("kept_pieces") or []
    if kept and min(p["r_median_mm"] for p in kept) > config.rim_radius_m * 1000 - 15:
        out.append({"code": "model_at_disc_rim", "message": "The model lies at the disc rim: it may be a piece of the "
                    "turntable, not the object; rescan with the object in the centre."})
    cut = (res.get("base") or {}).get("faces_after_cut") or 0
    after = ((res.get("base") or {}).get("components_after_cut") or {}).get("faces_kept")
    if cut and after is not None and after < 0.1 * cut:
        out.append({"code": "most_faces_dropped", "message": f"Only {after:,d} of {cut:,d} faces were kept as the "
                    f"object: check the model.", "faces_kept": after, "faces_after_cut": cut})
    return out


def _write_binary_copies(mesh: trimesh.Trimesh, obj_path: Path) -> Dict[str, str]:
    """<obj>.ply and <obj>.npz from the mesh parsed back from the OBJ (see clean_and_repair_mesh binary_copies)."""
    out: Dict[str, str] = {}
    try:
        ply = obj_path.with_suffix(".ply")
        mesh.export(str(ply))                    # before anything is computed on it: same bytes as the conversion
        out["ply_path"] = str(ply)
        npz = obj_path.with_suffix(".npz")
        np.savez(str(npz), vertices=np.asarray(mesh.vertices, np.float64), faces=np.asarray(mesh.faces, np.int64))
        out["arrays_path"] = str(npz)
    except Exception as e:
        logger.warning(f"Binary copies of the cleaned mesh not written ({e}); the OBJ is read instead.")
    return out


def clean_and_repair_mesh(
    input_obj_path: str,
    output_obj_path: str,
    config: MeshCleaningConfig,
    patch_hook=None,
    binary_copies: bool = False,
) -> Dict[str, Any]:
    """
    `binary_copies` (B5, plans/PLAN_STREAMING_DAG.md): the cleaned mesh as parsed back from the OBJ written here is
    also saved as <name>.ply (the exact bytes texrecon_runner's OBJ -> PLY conversion writes) and <name>.npz (float64
    vertices, int64 faces: what texture_prep reads from the OBJ), so the texturing steps skip parsing the OBJ text.
    res["ply_path"] / res["arrays_path"] name them (absent when not written).
    Raw photogrammetry mesh -> closed high-poly mesh (the one texrecon, STL / 3MF and the low-poly decimation use):
    1. the object pieces by position (select_object_components), not the largest piece;
    2. holes near the base closed, then the planar cut, all base loops capped (seal_watertight_pedestal);
    3. PyMeshLab (in millimetres): duplicates, non-manifold repair, decimation to the high-poly target;
    4. the holes closed ONCE, on the high-poly mesh (fill_holes_with_fairing; `patch_hook(ms)` sees each patch);
    5. Taubin smoothing, unit vertex normals. The low poly is a topology-preserving decimation of this mesh.
    The result has "warnings" (mesh_warnings) and the numbers of every step.
    """
    in_p = Path(input_obj_path).resolve()
    out_p = Path(output_obj_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading mesh for cleaning: {in_p}")
    raw_mesh = trimesh.load(str(in_p), process=False)
    rim = getattr(config, "rim_radius_m", 0.115)
    report: Dict[str, Any] = {"raw_faces": int(len(raw_mesh.faces)) if hasattr(raw_mesh, "faces") else None}

    # 1. The object pieces (the old "keep the largest" kept the disc of the orange piece 094944)
    if config.keep_largest_only and hasattr(raw_mesh, "split"):
        raw_mesh, report["components"] = select_object_components(raw_mesh, config.slice_y_m, rim,
                                                                  label="Raw mesh: ")

    # 2. Holes near the base, then cut and cap
    if getattr(config, "max_hole_edges", 0) and getattr(config, "base_band_m", 0) > config.slice_y_m:
        raw_mesh, report["base_band"] = _close_base_band_holes(raw_mesh, config.slice_y_m, config.base_band_m,
                                                               config.max_hole_edges)
    base: Dict[str, Any] = {}
    sealed = seal_watertight_pedestal(
        input_mesh=raw_mesh,
        slice_y=config.slice_y_m,
        skirt_y=config.skirt_bottom_y_m,
        max_radius=config.max_radius_m,
        extrude_skirt=getattr(config, "extrude_skirt", False),
        base_color_rgb=getattr(config, "base_color_rgb", (25, 25, 25)),
        rim_radius=rim,
        report=base,
    )
    report["base"] = base

    # 3. PyMeshLab repair
    is_watertight = sealed.is_watertight
    volume_cm3 = None
    sealed_faces = len(sealed.faces)

    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        # Millimetres inside PyMeshLab: its absolute epsilons (self-intersection, refinement) suit object sizes
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(sealed.vertices, np.float64) * 1000.0,
                                   face_matrix=np.asarray(sealed.faces, np.int32)))

        logger.info("Running PyMeshLab manifold repair filters...")
        if config.remove_duplicate_faces:
            ms.meshing_remove_duplicate_faces()
            ms.meshing_remove_duplicate_vertices()

        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()

        # High-Poly Conditioning: decimate by a share of the sealed mesh (or to a fixed target)
        # (Preserves embossed lettering/logos while eliminating micro-triangle depth noise)
        curr = ms.current_mesh()
        curr_faces = curr.face_number()
        sealed_faces = curr_faces
        target_high = face_target(curr_faces, getattr(config, "high_poly_ratio", 0.75),
                                  getattr(config, "target_high_poly_faces", 0))
        if target_high and curr_faces > target_high:
            logger.info(
                f"Conditioning High-Poly mesh: decimating from {curr_faces:,d} to {target_high:,d} faces "
                f"(preserving boundaries, normal relief, and planar quadrics)..."
            )
            try:
                # Quadric decimation does nothing while faces are selected
                ms.set_selection_none()
                ms.meshing_decimation_quadric_edge_collapse(
                    targetfacenum=target_high,
                    preserveboundary=True,
                    preservenormal=True,
                    preservetopology=True,
                    planarquadric=True,
                    optimalplacement=True,
                )
                decimated = ms.current_mesh().face_number()
                if decimated > target_high * 1.1:
                    logger.warning(f"High-poly decimation stopped at {decimated:,d} faces (target {target_high:,d}).")
                # Edge collapses can still leave a few non-manifold edges
                ms.meshing_repair_non_manifold_edges()
                ms.meshing_repair_non_manifold_vertices()
            except Exception as de:
                logger.warning(f"Quadric edge collapse decimation warning: {de}")

        # 4. Holes closed once, on the high poly (2.8): the low poly inherits the closed mesh
        if getattr(config, "max_hole_edges", 0):
            report["holes"] = fill_holes_with_fairing(ms, config.max_hole_edges,
                                                      getattr(config, "hole_fairing_steps", 5), patch_hook)
            h = report["holes"]
            logger.info(f"Hole filling: {h.get('boundary_edges_before')} -> {h.get('boundary_edges_after')} open "
                        f"edges, {h.get('patched_area_mm2', 0) / 100:.1f} cm2 patched.")
        else:
            ms.meshing_close_holes(maxholesize=30, newfaceselected=False)

        # Volume-preserving Taubin smoothing (eliminates 200um noise ripples without shrinkage)
        smooth_steps = getattr(config, "high_poly_smooth_iterations", 5)
        if smooth_steps > 0:
            logger.info(f"Applying volume-preserving Taubin smoothing ({smooth_steps} steps)...")
            try:
                ms.set_selection_none()
                ms.apply_coord_taubin_smoothing(
                    lambda_=0.5,
                    mu=-0.53,
                    stepsmoothnum=smooth_steps,
                )
            except Exception as se:
                logger.warning(f"Taubin smoothing warning: {se}")

        # Back to metres, then angle-weighted per-vertex normals STRICTLY NORMALIZED
        # (CRITICAL: prevents unnormalized 2pi length normals causing Blender specular frosting!)
        ms.compute_matrix_from_scaling_or_normalization(axisx=0.001, axisy=0.001, axisz=0.001, uniformflag=True,
                                                        scalecenter="origin", freeze=True)
        logger.info("Computing angle-weighted vertex normals and normalizing strictly (||n|| = 1.0000)...")
        try:
            ms.compute_normal_per_vertex(weightmode=1)  # 1 = weighted by angle
            ms.apply_normal_normalization_per_vertex()
        except Exception as ne:
            logger.warning(f"Normal computation/normalization warning: {ne}")

        # Save repaired mesh
        ms.save_current_mesh(str(out_p))

        # Check metrics
        curr = ms.current_mesh()
        v_count = curr.vertex_number()
        f_count = curr.face_number()

        # Re-check watertightness with trimesh
        final_check = trimesh.load(str(out_p), process=False)
        if binary_copies:
            report.update(_write_binary_copies(final_check, out_p))
        is_watertight = bool(final_check.is_watertight)
        if is_watertight and final_check.volume is not None:
            volume_cm3 = round(float(final_check.volume) * 1e6, 2)
        report["extent_mm"] = [round(float(x) * 1000, 1) for x in final_check.extents]

    except Exception as e:
        logger.warning(f"PyMeshLab processing warning: {e}. Falling back to trimesh export.")
        try:
            sealed.fix_normals()
            if hasattr(sealed, "vertex_normals") and len(sealed.vertex_normals) > 0:
                sealed.vertex_normals = trimesh.geometry.unitize(sealed.vertex_normals)
        except Exception:
            pass
        sealed.export(str(out_p))
        v_count = len(sealed.vertices)
        f_count = len(sealed.faces)
        report["extent_mm"] = [round(float(x) * 1000, 1) for x in sealed.extents]

    logger.info(
        f"Mesh cleaning completed: {out_p.name} | {v_count:,d} vertices, {f_count:,d} faces | "
        f"Watertight: {is_watertight} | Volume: {volume_cm3} cm³"
    )

    res = {
        "output_path": str(out_p),
        "sealed_faces": sealed_faces,   # before the high-poly decimation (reference of the face ratios)
        "vertex_count": v_count,
        "face_count": f_count,
        "is_watertight": is_watertight,
        "volume_cm3": volume_cm3,
        "patched_area_cm2": round(float((report.get("holes") or {}).get("patched_area_mm2", 0.0)) / 100.0, 2),
        **report,
    }
    res["warnings"] = mesh_warnings(res, config)
    return res
