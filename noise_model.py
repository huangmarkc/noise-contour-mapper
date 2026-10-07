"""
noise_model.py — Core acoustical calculation functions for source-based noise mapping.

This module implements a hybrid noise-mapping method:

  1. Inverse square attenuation for point sources.
  2. Logarithmic (energy) addition of sound levels from multiple sources.
  3. Source contribution calculation at a receiver point.
  4. Grid-based calculation of predicted sound levels.
  5. Residual correction using measured sound level meter (SLM) readings.
  6. Optional walls and barriers: transmission loss through walls, diffraction
     around wall ends and over partial barriers (Maekawa / Kurze-Anderson), and
     first-order reflections (image sources); measurement influence routed
     around walls instead of through them.

Important acoustical assumptions
--------------------------------
* The inverse square law (Lp2 = Lp1 - 20*log10(r2/r1)) is a baseline approximation
  for point-like sources radiating spherically in a free field.
* Indoor industrial environments may include reflections, reverberation, barriers,
  shielding, and non-point source behavior (line/area sources). The simple model
  will over- or under-predict in those conditions.
* dBA values must be combined using logarithmic energy addition, never arithmetic
  addition: two 90 dBA sources combine to 93 dBA, not 180 dBA.
* Residual correction calibrates the idealized source model against real measured
  SLM readings, absorbing (in aggregate) the effects the physics model ignores.
* Walls (section 9) are 2-D segments. Single-number A-weighted values are used
  throughout: transmission loss as an effective field value, diffraction at a
  representative 500 Hz, one reflection per wall. Multiple reflections, floor and
  ceiling reflections, and reverberant build-up are not modeled.
* The corrected noise grid is intended for visualization and planning — it is NOT
  a replacement for personal noise dosimetry or regulatory exposure assessment.

The wall model in section 9 matches the Noise Contour Mapper app (ui/index.html)
calculation for calculation; keep the two in sync when either changes.

Run this file directly for a complete worked example with synthetic data:

    python noise_model.py
"""

import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Inverse square attenuation
# ---------------------------------------------------------------------------

def attenuate_point_source(source_level_dba, reference_distance_m,
                           receiver_distance_m, min_distance_m=0.5):
    """Predict the sound level at a receiver using inverse square attenuation.

    Uses the point-source free-field relationship:

        Lp2 = Lp1 - 20 * log10(r2 / r1)

    where Lp1 is the source sound pressure level (dBA) at reference distance r1,
    and Lp2 is the predicted level at receiver distance r2.

    Parameters
    ----------
    source_level_dba : float
        Source sound pressure level in dBA at the reference distance.
    reference_distance_m : float
        Distance (m) at which source_level_dba was specified. Must be > 0.
    receiver_distance_m : float or np.ndarray
        Distance(s) (m) from the source to the receiver(s).
    min_distance_m : float, optional
        Minimum receiver distance (default 0.5 m). Distances below this are
        clamped, preventing divide-by-zero and unrealistic near-field levels
        (the far-field inverse square law does not hold very close to a source).

    Returns
    -------
    float or np.ndarray
        Predicted sound pressure level(s) in dBA at the receiver distance(s).
    """
    if reference_distance_m <= 0:
        raise ValueError("reference_distance_m must be greater than zero.")

    # Clamp receiver distance to avoid division by zero / near-field blow-up.
    r2 = np.maximum(np.asarray(receiver_distance_m, dtype=float), min_distance_m)

    return source_level_dba - 20.0 * np.log10(r2 / reference_distance_m)


# ---------------------------------------------------------------------------
# 2. Logarithmic sound level addition
# ---------------------------------------------------------------------------

def add_sound_levels(levels_dba):
    """Combine multiple sound levels using logarithmic energy addition.

    dBA values are logarithmic quantities and must never be added
    arithmetically. Each level is converted to linear acoustic energy,
    the energies are summed, and the sum is converted back to dBA:

        L_total = 10 * log10( sum( 10^(L_i / 10) ) )

    Parameters
    ----------
    levels_dba : list or np.ndarray
        Sound levels in dBA. NaN entries are ignored.

    Returns
    -------
    float
        Combined sound level in dBA, or NaN if the input is empty or
        contains only NaN values.
    """
    levels = np.asarray(levels_dba, dtype=float).ravel()
    levels = levels[~np.isnan(levels)]          # drop NaN values

    if levels.size == 0:
        return float("nan")

    energy_sum = np.sum(10.0 ** (levels / 10.0))  # linear energy sum
    return float(10.0 * np.log10(energy_sum))


# ---------------------------------------------------------------------------
# 3. Distance
# ---------------------------------------------------------------------------

def calculate_distance(x1, y1, x2, y2):
    """Euclidean distance between (x1, y1) and (x2, y2) in meters.

    Accepts scalars or numpy arrays (broadcasting applies), so it can compute
    a single receiver-to-source distance or a whole grid of distances at once.

    Returns
    -------
    float or np.ndarray
        Distance(s) in meters.
    """
    return np.hypot(np.asarray(x2, dtype=float) - np.asarray(x1, dtype=float),
                    np.asarray(y2, dtype=float) - np.asarray(y1, dtype=float))


# ---------------------------------------------------------------------------
# 4. Source contributions at a single receiver point
# ---------------------------------------------------------------------------

def calculate_source_contributions_at_point(receiver_x, receiver_y, noise_sources):
    """Predict the sound level at one receiver point from all noise sources.

    Parameters
    ----------
    receiver_x, receiver_y : float
        Receiver coordinates in meters.
    noise_sources : list of dict
        Each source dict must contain:
            source_id, x, y, source_level_dba, reference_distance_m,
            source_type, description
        (Only point sources are modeled; source_type is carried through
        for future extension to line/area sources.)

    Returns
    -------
    total_predicted_dba : float
        Logarithmic (energy) combination of all source contributions.
    individual_contributions : list of dict
        One entry per source with source_id, description, distance_m,
        and contribution_dba.
    """
    individual_contributions = []

    for src in noise_sources:
        dist = calculate_distance(src["x"], src["y"], receiver_x, receiver_y)
        contribution = attenuate_point_source(
            src["source_level_dba"], src["reference_distance_m"], dist)
        individual_contributions.append({
            "source_id": src["source_id"],
            "description": src.get("description", ""),
            "distance_m": float(dist),
            "contribution_dba": float(contribution),
        })

    # Combine contributions with energy addition — never arithmetic addition.
    total_predicted_dba = add_sound_levels(
        [c["contribution_dba"] for c in individual_contributions])

    return total_predicted_dba, individual_contributions


# ---------------------------------------------------------------------------
# 5. Grid-based source model
# ---------------------------------------------------------------------------

def calculate_source_noise_grid(grid_x, grid_y, noise_sources, walls=None, **wall_options):
    """Predict sound levels across a grid from all noise sources.

    Parameters
    ----------
    grid_x, grid_y : np.ndarray
        Meshgrid arrays of receiver coordinates (m), e.g. from np.meshgrid.
    noise_sources : list of dict
        Same structure as in calculate_source_contributions_at_point.
    walls : list of dict, optional
        Walls and barriers (see make_wall). When given, every cell is evaluated
        with WalledSourceModel (blocking, diffraction, reflections).
    **wall_options
        Passed to WalledSourceModel (block, reflect, source_height_m, ...).

    Returns
    -------
    total_noise_grid : np.ndarray
        Combined predicted sound level (dBA) at every grid cell.
    per_source_grids : dict
        {source_id: np.ndarray} — the individual contribution grid of each
        source, useful for identifying which equipment dominates each zone.
    """
    if walls:
        gx, gy = np.asarray(grid_x, dtype=float), np.asarray(grid_y, dtype=float)
        per_source_grids = {}
        energy_sum = np.zeros_like(gx)
        for src in noise_sources:
            model = WalledSourceModel([src], walls, **wall_options)
            level_grid = np.empty_like(gx)
            for idx in np.ndindex(gx.shape):
                level_grid[idx] = model.level(gx[idx], gy[idx])
            per_source_grids[src["source_id"]] = level_grid
            energy_sum += 10.0 ** (level_grid / 10.0)
        return 10.0 * np.log10(energy_sum), per_source_grids

    per_source_grids = {}
    energy_sum = np.zeros_like(np.asarray(grid_x, dtype=float))

    for src in noise_sources:
        dist_grid = calculate_distance(src["x"], src["y"], grid_x, grid_y)
        level_grid = attenuate_point_source(
            src["source_level_dba"], src["reference_distance_m"], dist_grid)
        per_source_grids[src["source_id"]] = level_grid
        energy_sum += 10.0 ** (level_grid / 10.0)   # accumulate linear energy

    total_noise_grid = 10.0 * np.log10(energy_sum)
    return total_noise_grid, per_source_grids


# ---------------------------------------------------------------------------
# 6. Prediction at measurement locations
# ---------------------------------------------------------------------------

def predict_at_measurement_points(measurements, noise_sources, walls=None, **wall_options):
    """Predict source-model sound levels at each SLM measurement location.

    Parameters
    ----------
    measurements : pd.DataFrame
        Must contain columns: point_id, x, y, measured_dba.
    noise_sources : list of dict
        Same structure as in calculate_source_contributions_at_point.
    walls : list of dict, optional
        Walls and barriers (see make_wall); predictions then include them.
    **wall_options
        Passed to WalledSourceModel.

    Returns
    -------
    pd.DataFrame
        A copy of `measurements` with two added columns:
            predicted_dba — source-model prediction at the point
            residual_dba  — measured_dba - predicted_dba
        Positive residuals mean the model under-predicts (e.g., reflections
        or unmodeled sources); negative means it over-predicts (e.g., barriers
        or shielding).
    """
    result = measurements.copy()
    model = WalledSourceModel(noise_sources, walls, **wall_options) if walls else None

    predicted = []
    for _, row in result.iterrows():
        if model is not None:
            total = model.level(row["x"], row["y"])
        else:
            total, _ = calculate_source_contributions_at_point(
                row["x"], row["y"], noise_sources)
        predicted.append(total)

    result["predicted_dba"] = predicted
    result["residual_dba"] = result["measured_dba"] - result["predicted_dba"]
    return result


# ---------------------------------------------------------------------------
# 7. Residual interpolation (IDW)
# ---------------------------------------------------------------------------

def interpolate_residuals_idw(measurements_with_residuals, grid_x, grid_y,
                              power=2, min_distance_m=0.5, walls=None, **wall_geometry):
    """Interpolate measurement residuals across the grid with inverse
    distance weighting (IDW).

    The residual field captures, in aggregate, everything the point-source
    model ignores (reverberation, barriers, directivity, unmodeled sources),
    calibrating the physics model to the measured SLM readings.

    Parameters
    ----------
    measurements_with_residuals : pd.DataFrame
        Must contain columns: x, y, residual_dba
        (as produced by predict_at_measurement_points).
    grid_x, grid_y : np.ndarray
        Meshgrid arrays of receiver coordinates (m).
    power : float, optional
        IDW power parameter (default 2). Higher values localize the
        influence of each measurement.
    min_distance_m : float, optional
        Distances below this are clamped (default 0.5 m) to avoid division
        by zero at grid cells that coincide with a measurement point.
    walls : list of dict, optional
        When given, distances are measured around full-height walls (e.g. through
        doorways) instead of through them. Cells no measurement can reach get a
        residual of 0 (the physics prediction alone).
    **wall_geometry
        Passed to WallModel (vertex_offset_m, join_tol_m).

    Returns
    -------
    residual_grid : np.ndarray
        Interpolated residual (dB) at every grid cell.
    """
    if walls:
        model = WallModel(walls, **wall_geometry)
        gx, gy = np.asarray(grid_x, dtype=float), np.asarray(grid_y, dtype=float)
        rows = [(row["x"], row["y"], row["residual_dba"])
                for _, row in measurements_with_residuals.iterrows()]
        sites = [model.site(x, y) for x, y, _ in rows] if model.full else None
        residual_grid = np.zeros_like(gx)
        for idx in np.ndindex(gx.shape):
            x, y = gx[idx], gy[idx]
            vis = None
            num = den = 0.0
            for k, (mx, my, res) in enumerate(rows):
                if sites is None:
                    d = math.hypot(x - mx, y - my)
                else:
                    if vis is None and model.blocked(mx, my, x, y):
                        vis = model.visible_vertices(x, y)
                    d = model.path_distance(mx, my, sites[k], x, y, vis)
                if math.isinf(d):
                    continue
                w = 1.0 / max(d, min_distance_m) ** power
                num += w * res
                den += w
            residual_grid[idx] = num / den if den > 0 else 0.0
        return residual_grid

    weight_sum = np.zeros_like(np.asarray(grid_x, dtype=float))
    weighted_residual_sum = np.zeros_like(weight_sum)

    for _, row in measurements_with_residuals.iterrows():
        dist = calculate_distance(row["x"], row["y"], grid_x, grid_y)
        dist = np.maximum(dist, min_distance_m)     # avoid divide-by-zero
        w = 1.0 / dist ** power                     # IDW weights
        weight_sum += w
        weighted_residual_sum += w * row["residual_dba"]

    residual_grid = weighted_residual_sum / weight_sum
    return residual_grid


# ---------------------------------------------------------------------------
# 8. Final corrected grid
# ---------------------------------------------------------------------------

def create_corrected_noise_grid(source_grid, residual_grid):
    """Apply the interpolated residual correction to the source-model grid.

    The residual is added in dBA space (a dB offset is a multiplicative
    correction of acoustic energy), producing a map that matches the SLM
    readings at the measurement points while following the physics-based
    spatial pattern between them.

    Parameters
    ----------
    source_grid : np.ndarray
        Source-model predicted levels (dBA).
    residual_grid : np.ndarray
        Interpolated residuals (dB) from interpolate_residuals_idw.

    Returns
    -------
    final_corrected_grid : np.ndarray
        Corrected predicted sound levels (dBA).
    """
    return source_grid + residual_grid


# ---------------------------------------------------------------------------
# 9. Walls, barriers and reflections
# ---------------------------------------------------------------------------
#
# Walls are 2-D segments with coordinates in meters. A full-height wall
# ("height": None) reduces sound passing through it by its transmission loss
# "tl" (dB) and stops measurement influence; sound also bends around its free
# ends. A partial barrier ("height" in m) only screens sound bending over its
# top. Every wall reflects sound once, reduced by its absorption "alpha".

LAMBDA_M = 343.0 / 500.0       # wavelength at 500 Hz, the usual single-frequency stand-in for dBA
MAX_SCREEN_DB = 20.0           # practical cap on screening by diffraction
CAP_N = 10 ** ((MAX_SCREEN_DB - 5) / 10) / (2 * math.pi)   # Fresnel number where the cap is reached

WALL_TYPES = {
    "concrete": {"name": "Concrete / masonry block",         "tl": 30, "alpha": 0.02, "height": None},
    "panels":   {"name": "Concrete with acoustic panels",    "tl": 30, "alpha": 0.70, "height": None},
    "drywall":  {"name": "Drywall / stud partition",         "tl": 20, "alpha": 0.05, "height": None},
    "metal":    {"name": "Metal panel / sheet steel",        "tl": 20, "alpha": 0.05, "height": None},
    "glass":    {"name": "Glass / window wall",              "tl": 20, "alpha": 0.04, "height": None},
    "curtain":  {"name": "Acoustic curtain",                 "tl": 10, "alpha": 0.50, "height": None},
    "barrier":  {"name": "Partial barrier / machine screen", "tl": 0,  "alpha": 0.05, "height": 2.4},
}


def make_wall(x1, y1, x2, y2, wall_type="concrete", **overrides):
    """Create a wall dict from a preset type; overrides (tl, alpha, height) win."""
    preset = {k: v for k, v in WALL_TYPES[wall_type].items() if k != "name"}
    wall = {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
            "type": wall_type, **preset}
    wall.update(overrides)
    return wall


def screen_db(delta_m):
    """Screening (dB) of sound bending around or over an edge.

    delta_m is the extra path length via the edge (m); pass it negative when the
    edge does not break the line of sight (the "lit" side). Kurze-Anderson form
    of Maekawa's curve with Fresnel number N = 2*delta/lambda at 500 Hz, capped
    at MAX_SCREEN_DB. Continuous through N = 0 (5 dB at the shadow boundary).
    """
    n = 2.0 * delta_m / LAMBDA_M
    if n <= -0.2:
        return 0.0
    if n < 0:
        r = math.sqrt(-2 * math.pi * n)
        a = 5 + 20 * math.log10(r / math.tan(r))
    elif n == 0:
        a = 5.0
    else:
        r = math.sqrt(2 * math.pi * n)
        a = 5 + 20 * math.log10(r / math.tanh(r))
    return min(max(a, 0.0), MAX_SCREEN_DB)


def _cross_t(ax, ay, bx, by, cx, cy, dx, dy):
    """Parameter t where segment a->b crosses segment c->d, or -1.

    The crossed segment's own ends count (paths can't slip through the joint
    between two walls); the path's ends don't (paths may start or end on a wall).
    """
    rx, ry, qx, qy = bx - ax, by - ay, dx - cx, dy - cy
    den = rx * qy - ry * qx
    if abs(den) < 1e-12:
        return -1.0
    wx, wy = cx - ax, cy - ay
    t = (wx * qy - wy * qx) / den
    u = (wx * ry - wy * rx) / den
    return t if (1e-9 < t < 1 - 1e-9 and -1e-9 <= u <= 1 + 1e-9) else -1.0


def _seg_dist(px, py, w):
    ex, ey = w["x2"] - w["x1"], w["y2"] - w["y1"]
    l2 = ex * ex + ey * ey
    t = min(max(((px - w["x1"]) * ex + (py - w["y1"]) * ey) / l2, 0.0), 1.0) if l2 else 0.0
    return math.hypot(px - (w["x1"] + t * ex), py - (w["y1"] + t * ey))


def _mirror_image(src, w):
    ex, ey = w["x2"] - w["x1"], w["y2"] - w["y1"]
    f = ((src["x"] - w["x1"]) * ex + (src["y"] - w["y1"]) * ey) / (ex * ex + ey * ey)
    return 2 * (w["x1"] + f * ex) - src["x"], 2 * (w["y1"] + f * ey) - src["y"]


def _bounce(src, w, ix, iy, rx, ry):
    """Bounce point (px, py, u) of the reflection of src off w's line toward (rx, ry);
    None unless source and receiver are both in front of the wall. u in [0, 1]
    means the bounce lands on the wall itself."""
    ex, ey = w["x2"] - w["x1"], w["y2"] - w["y1"]
    len2 = ex * ex + ey * ey
    cs = ex * (src["y"] - w["y1"]) - ey * (src["x"] - w["x1"])
    cr = ex * (ry - w["y1"]) - ey * (rx - w["x1"])
    if len2 < 1e-9 or cs * cr <= 0:
        return None
    t = cs / (cs + cr)
    px, py = ix + t * (rx - ix), iy + t * (ry - iy)
    u = ((px - w["x1"]) * ex + (py - w["y1"]) * ey) / len2
    return px, py, u


class WallModel:
    """Wall geometry prepared for path finding.

    Shortest routes around full-height walls use a visibility graph whose
    corners sit just past each wall end, on both sides (vertex_offset_m). Wall
    ends within join_tol_m of another full-height wall are joints, not edges.
    """

    def __init__(self, walls, vertex_offset_m=0.05, join_tol_m=0.03):
        self.walls = list(walls)
        self.full = [w for w in self.walls if w.get("height") is None]
        eps = vertex_offset_m
        self.vx, self.vy, seen = [], [], set()
        for w in self.full:
            length = math.hypot(w["x2"] - w["x1"], w["y2"] - w["y1"]) or 1.0
            dx, dy = (w["x2"] - w["x1"]) / length, (w["y2"] - w["y1"]) / length
            for ex, ey, s in ((w["x2"], w["y2"], 1), (w["x1"], w["y1"], -1)):
                for side in (1, -1):
                    px = ex + eps * (s * dx - side * dy)
                    py = ey + eps * (s * dy + side * dx)
                    key = (math.floor(px * 3 / eps + 0.5), math.floor(py * 3 / eps + 0.5))
                    if key not in seen:
                        seen.add(key)
                        self.vx.append(px)
                        self.vy.append(py)
        self.edges = []          # free wall ends: where sound bends around a wall
        for w in self.full:
            for ex, ey in ((w["x1"], w["y1"]), (w["x2"], w["y2"])):
                if not any(o is not w and _seg_dist(ex, ey, o) < join_tol_m for o in self.full):
                    self.edges.append((ex, ey))
        n = len(self.vx)
        self.vd = np.full((n, n), np.inf)    # corner-to-corner distance where visible
        for i in range(n):
            for j in range(i + 1, n):
                if not self.blocked(self.vx[i], self.vy[i], self.vx[j], self.vy[j]):
                    d = math.hypot(self.vx[j] - self.vx[i], self.vy[j] - self.vy[i])
                    self.vd[i, j] = self.vd[j, i] = d

    def blocked(self, ax, ay, bx, by):
        """True if the straight path a->b passes through a full-height wall."""
        x0, x1, y0, y1 = min(ax, bx), max(ax, bx), min(ay, by), max(ay, by)
        for w in self.full:
            if (max(w["x1"], w["x2"]) < x0 or min(w["x1"], w["x2"]) > x1 or
                    max(w["y1"], w["y2"]) < y0 or min(w["y1"], w["y2"]) > y1):
                continue
            if _cross_t(ax, ay, bx, by, w["x1"], w["y1"], w["x2"], w["y2"]) > 0:
                return True
        return False

    def visible_vertices(self, x, y):
        return [v for v in range(len(self.vx))
                if not self.blocked(x, y, self.vx[v], self.vy[v])]

    def _relax(self, dist):
        """Dijkstra over the corner graph, starting from first-hop distances."""
        n = len(self.vx)
        done = np.zeros(n, dtype=bool)
        while True:
            best, bd = -1, math.inf
            for v in range(n):
                if not done[v] and dist[v] < bd:
                    best, bd = v, dist[v]
            if best < 0:
                return dist
            done[best] = True
            for v in range(n):
                nd = bd + self.vd[best, v]
                if not done[v] and nd < dist[v]:
                    dist[v] = nd

    def site(self, x, y):
        """Shortest distance from (x, y) to every corner without crossing a wall."""
        dist = np.full(len(self.vx), np.inf)
        for v in range(len(self.vx)):
            if not self.blocked(x, y, self.vx[v], self.vy[v]):
                dist[v] = math.hypot(self.vx[v] - x, self.vy[v] - y)
        return self._relax(dist)

    def mirror_site(self, src, w):
        """Corner distances from src's mirror image in wall w: the first hop must
        bounce off the wall itself with both legs clear."""
        ix, iy = _mirror_image(src, w)
        dist = np.full(len(self.vx), np.inf)
        for v in range(len(self.vx)):
            b = _bounce(src, w, ix, iy, self.vx[v], self.vy[v])
            if b is None or not 0 <= b[2] <= 1:
                continue
            if (self.blocked(src["x"], src["y"], b[0], b[1]) or
                    self.blocked(b[0], b[1], self.vx[v], self.vy[v])):
                continue
            dist[v] = math.hypot(self.vx[v] - ix, self.vy[v] - iy)
        return (ix, iy), self._relax(dist)

    def via_corners(self, dist, x, y, vis):
        best = math.inf
        for v in vis:
            best = min(best, math.hypot(x - self.vx[v], y - self.vy[v]) + dist[v])
        return best

    def path_distance(self, sx, sy, dist, x, y, vis=None):
        """Shortest distance from site (sx, sy) to (x, y) around full-height walls."""
        if not self.blocked(sx, sy, x, y):
            return math.hypot(x - sx, y - sy)
        return self.via_corners(dist, x, y, vis if vis is not None else self.visible_vertices(x, y))

    def edge_screen(self, ax, ay, bx, by):
        """Lit-side screening (dB) of an unblocked path passing just beside a free
        wall end; fades from 5 dB at the shadow boundary to 0."""
        d = math.hypot(bx - ax, by - ay)
        best = 0.0
        for ex, ey in self.edges:
            delta = math.hypot(ex - ax, ey - ay) + math.hypot(bx - ex, by - ey) - d
            if delta < 0.1 * LAMBDA_M:
                best = max(best, screen_db(-delta))
        return best

    def path_loss(self, ax, ay, bx, by, skip, source_height_m, ear_height_m):
        """Reduction (dB) along the straight path a->b: the tl of every full-height
        wall crossed plus the largest over-the-top screening of any partial
        barrier crossed. skip = index of a wall to ignore. Returns (dB, crosses_full)."""
        tl, scr, full = 0.0, 0.0, False
        d = math.hypot(bx - ax, by - ay)
        hs, hr = source_height_m, ear_height_m
        for k, w in enumerate(self.walls):
            if k == skip:
                continue
            t = _cross_t(ax, ay, bx, by, w["x1"], w["y1"], w["x2"], w["y2"])
            if t < 0:
                continue
            if w.get("height") is None:
                tl += w["tl"]
                full = True
                continue
            a, b, hb = t * d, (1 - t) * d, w["height"]
            over = math.hypot(a, hb - hs) + math.hypot(b, hb - hr) - math.hypot(a + b, hr - hs)
            scr = max(scr, screen_db(over if hb > hs + (hr - hs) * t else -over))
        return tl + scr, full


class WalledSourceModel:
    """Point sources with walls: the same calculation as the app's source model.

    Each source reaches a receiver (1) straight through, minus the tl of walls
    crossed and the screening of partial barriers; (2) when a full-height wall is
    in the way, also around wall ends (e.g. through a doorway), screened by the
    detour; and (3) once off each wall, from its mirror image, reduced by the
    wall's absorption and treated like a real source (blocked, routed around
    walls, faded at wall ends). Paths are combined by energy addition.
    """

    def __init__(self, noise_sources, walls, block=True, reflect=True,
                 source_height_m=1.0, ear_height_m=1.5, **wall_geometry):
        self.sources = list(noise_sources)
        self.walls = list(walls or [])
        self.model = WallModel(self.walls, **wall_geometry)
        self.block = block and bool(self.walls)
        self.reflect = reflect and bool(self.walls)
        self.hs, self.hr = source_height_m, ear_height_m
        paths = self.block and bool(self.model.full)
        self.sites = [self.model.site(s["x"], s["y"]) for s in self.sources] if paths else None
        self.mirrors = None
        if paths and self.reflect:
            self.mirrors = [[self.model.mirror_site(s, w)[1]
                             if w["alpha"] < 1 and math.hypot(w["x2"] - w["x1"], w["y2"] - w["y1"]) >= 1e-6
                             else None for w in self.walls] for s in self.sources]

    def level(self, x, y):
        """Predicted sound level (dBA) at (x, y)."""
        m, energy = self.model, 0.0
        vis = None

        def get_vis():
            nonlocal vis
            if vis is None:
                vis = m.visible_vertices(x, y)
            return vis

        def loss(ax, ay, bx, by, skip):
            return m.path_loss(ax, ay, bx, by, skip, self.hs, self.hr)

        for si, s in enumerate(self.sources):
            def lvl(d, s=s):
                return s["source_level_dba"] - 20 * math.log10(max(d, 0.5) / s["reference_distance_m"])
            d = math.hypot(x - s["x"], y - s["y"])
            direct = lvl(d)
            if self.block:
                db, full = loss(s["x"], s["y"], x, y, -1)
                direct -= db
                if self.sites is not None and full:
                    g = m.path_distance(s["x"], s["y"], self.sites[si], x, y, get_vis())
                    if not math.isinf(g):
                        energy += 10 ** ((lvl(g) - screen_db(g - d)) / 10)
                elif self.sites is not None:
                    direct -= m.edge_screen(s["x"], s["y"], x, y)
            energy += 10 ** (direct / 10)
            if not self.reflect:
                continue
            for wi, w in enumerate(self.walls):
                if w["alpha"] >= 1 or math.hypot(w["x2"] - w["x1"], w["y2"] - w["y1"]) < 1e-6:
                    continue
                ix, iy = _mirror_image(s, w)
                b = _bounce(s, w, ix, iy, x, y)
                if b is None:
                    continue
                px, py, u = b
                absorb = 10 * math.log10(1 - w["alpha"])
                d_img = math.hypot(x - ix, y - iy)
                if 0 <= u <= 1:
                    qx, qy = (w["x1"], w["y1"]) if u < 0.5 else (w["x2"], w["y2"])
                    lr = lvl(d_img) + absorb - screen_db(
                        -(math.hypot(qx - ix, qy - iy) + math.hypot(x - qx, y - qy) - d_img))
                    if self.block:
                        db1, f1 = loss(s["x"], s["y"], px, py, wi)
                        db2, f2 = loss(px, py, x, y, wi)
                        lr -= db1 + db2
                        site = self.mirrors[si][wi] if self.mirrors is not None else None
                        if site is not None and (f1 or f2):
                            g = m.via_corners(site, x, y, get_vis())
                            if not math.isinf(g):
                                energy += 10 ** ((lvl(g) + absorb - screen_db(g - d_img)) / 10)
                        elif site is not None:
                            lr -= max(m.edge_screen(s["x"], s["y"], px, py), m.edge_screen(px, py, x, y))
                    energy += 10 ** (lr / 10)
                else:
                    qx, qy = (w["x1"], w["y1"]) if u < 0 else (w["x2"], w["y2"])
                    g = math.hypot(qx - ix, qy - iy) + math.hypot(x - qx, y - qy)
                    delta = g - d_img
                    if 2 * delta / LAMBDA_M > CAP_N:
                        continue
                    lr = lvl(g) + absorb - screen_db(delta)
                    if self.block:
                        lr -= loss(s["x"], s["y"], qx, qy, wi)[0] + loss(qx, qy, x, y, wi)[0]
                    energy += 10 ** (lr / 10)
        return 10 * math.log10(energy)


# ---------------------------------------------------------------------------
# Complete worked example with synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Synthetic facility: three noise sources ---------------------------
    noise_sources = [
        {"source_id": "S1", "x": 10.0, "y": 5.0, "source_level_dba": 92.0,
         "reference_distance_m": 1.0, "source_type": "point",
         "description": "Air compressor"},
        {"source_id": "S2", "x": 30.0, "y": 20.0, "source_level_dba": 96.0,
         "reference_distance_m": 1.0, "source_type": "point",
         "description": "Stamping press"},
        {"source_id": "S3", "x": 45.0, "y": 8.0, "source_level_dba": 88.0,
         "reference_distance_m": 1.0, "source_type": "point",
         "description": "Dust collector"},
    ]

    # --- Six synthetic SLM measurement points -------------------------------
    # measured_dba values deviate slightly from the ideal model, representing
    # real-world effects (reflections, shielding, background noise).
    measurements = pd.DataFrame({
        "point_id": ["P1", "P2", "P3", "P4", "P5", "P6"],
        "x":            [8.0, 15.0, 28.0, 35.0, 42.0, 25.0],
        "y":            [7.0, 12.0, 18.0, 15.0, 10.0, 5.0],
        "measured_dba": [86.5, 79.0, 88.0, 84.5, 81.0, 78.5],
    })

    # --- Prediction grid (numpy meshgrid), 0–50 m x 0–25 m at 0.5 m spacing -
    x_coords = np.arange(0.0, 50.0 + 0.5, 0.5)
    y_coords = np.arange(0.0, 25.0 + 0.5, 0.5)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    # --- 1) Individual source contributions at one example receiver ---------
    receiver = (20.0, 10.0)
    total_dba, contributions = calculate_source_contributions_at_point(
        receiver[0], receiver[1], noise_sources)

    print("=" * 70)
    print(f"Source contributions at receiver point {receiver}:")
    for c in contributions:
        print(f"  {c['source_id']} ({c['description']}): "
              f"{c['distance_m']:.1f} m -> {c['contribution_dba']:.1f} dBA")
    print(f"Total predicted level (energy sum): {total_dba:.1f} dBA")

    # --- 2) Source-model grid ------------------------------------------------
    source_grid, per_source_grids = calculate_source_noise_grid(
        grid_x, grid_y, noise_sources)

    # --- 3) Prediction + residuals at measurement points ---------------------
    measurements_with_residuals = predict_at_measurement_points(
        measurements, noise_sources)

    print("=" * 70)
    print("Measurements with predictions and residuals:")
    print(measurements_with_residuals.round(2).to_string(index=False))

    # --- 4) Residual interpolation and final corrected grid ------------------
    residual_grid = interpolate_residuals_idw(
        measurements_with_residuals, grid_x, grid_y, power=2)
    final_corrected_grid = create_corrected_noise_grid(source_grid, residual_grid)

    print("=" * 70)
    print(f"source_grid:          min {source_grid.min():.1f} dBA, "
          f"max {source_grid.max():.1f} dBA")
    print(f"final_corrected_grid: min {final_corrected_grid.min():.1f} dBA, "
          f"max {final_corrected_grid.max():.1f} dBA")
    # --- 5) The same facility with walls ------------------------------------
    # A block wall at x = 20 m with a 3 m doorway (y 10-13 m), a partial screen
    # beside the stamping press, and the building's south wall for reflections.
    walls = [
        make_wall(20, 0, 20, 10, "concrete"),
        make_wall(20, 13, 20, 25, "concrete"),
        make_wall(26, 16, 26, 24, "barrier"),
        make_wall(0, 0, 50, 0, "concrete"),
    ]
    walled = WalledSourceModel(noise_sources, walls)
    print("=" * 70)
    print("With walls (blocking, diffraction, reflections):")
    for label, (x, y) in [("behind the block wall", (15.0, 20.0)),
                          ("in line with the doorway", (15.0, 11.5)),
                          ("behind the partial screen", (24.0, 20.0))]:
        free = calculate_source_contributions_at_point(x, y, noise_sources)[0]
        print(f"  {label:28s} ({x:4.1f}, {y:4.1f}): "
              f"{walled.level(x, y):5.1f} dBA  (no walls: {free:5.1f} dBA)")

    print("=" * 70)
    print("Note: this corrected grid is for visualization and planning — not a")
    print("replacement for personal dosimetry or regulatory exposure assessment.")
