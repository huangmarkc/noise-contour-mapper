"""
noise_model.py — Core acoustical calculation functions for source-based noise mapping.

This module implements a hybrid noise-mapping method:

  1. Inverse square attenuation for point sources.
  2. Logarithmic (energy) addition of sound levels from multiple sources.
  3. Source contribution calculation at a receiver point.
  4. Grid-based calculation of predicted sound levels.
  5. Residual correction using measured sound level meter (SLM) readings.

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
* The corrected noise grid is intended for visualization and planning — it is NOT
  a replacement for personal noise dosimetry or regulatory exposure assessment.

Run this file directly for a complete worked example with synthetic data:

    python noise_model.py
"""

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

def calculate_source_noise_grid(grid_x, grid_y, noise_sources):
    """Predict sound levels across a grid from all noise sources.

    Parameters
    ----------
    grid_x, grid_y : np.ndarray
        Meshgrid arrays of receiver coordinates (m), e.g. from np.meshgrid.
    noise_sources : list of dict
        Same structure as in calculate_source_contributions_at_point.

    Returns
    -------
    total_noise_grid : np.ndarray
        Combined predicted sound level (dBA) at every grid cell.
    per_source_grids : dict
        {source_id: np.ndarray} — the individual contribution grid of each
        source, useful for identifying which equipment dominates each zone.
    """
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

def predict_at_measurement_points(measurements, noise_sources):
    """Predict source-model sound levels at each SLM measurement location.

    Parameters
    ----------
    measurements : pd.DataFrame
        Must contain columns: point_id, x, y, measured_dba.
    noise_sources : list of dict
        Same structure as in calculate_source_contributions_at_point.

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

    predicted = []
    for _, row in result.iterrows():
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
                              power=2, min_distance_m=0.5):
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

    Returns
    -------
    residual_grid : np.ndarray
        Interpolated residual (dB) at every grid cell.
    """
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
    print("=" * 70)
    print("Note: this corrected grid is for visualization and planning — not a")
    print("replacement for personal dosimetry or regulatory exposure assessment.")
