"""
Forward reconstruction of dt_params from recos (iterate.cpp / datasets.py).

Coordinate frame (dt_params cols 0-2):
  x, y, z are measured relative to the shower core; the core sits at (0, 0, 0).
  Absolute core position in the array (mc_xcore, mc_ycore) is not needed for the
  forward model — it is already subtracted when the h5 file is built.

recos (15 event-level parameters) + detector geometry (cols 0-2) predict cols 3-5:
  3. signal MIP          ~ S800 * LDF(r)            (iterate: pulsa/DET_AREA)
  4. plane front time    ~ affine in r_plane(θ,φ) from recos, anchored at brightest hit
  5. time vs flat front  ~ affine in linsley_t(r) from recos, anchored at brightest hit

Still not recoverable from recos alone:
  - which detectors fired (need cols 0-2 or dt_ids for the event);
  - waveforms (wfs_flat);
  - aggregate MLX features (peaks, layer asymmetry, AOP, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import h5py as h5
import numpy as np

# --- constants from reconstruction/reconstruction/inc/nuf.h and iterate.cpp ---
UNIT = 1200.0e2
R_X = 0.666667  # 800 m in 1200 m units
NSEC = 2.49827e-4
RUFPTN_TIMDIST = 0.249827048333
DET_AREA = 3.0
LINSLEY_r0 = 0.025
# iterate.cpp: r_plane (1200 m units) is added to t0 in µs directly — not × (1.2 km / c).
# MKS_PER_SPATIAL_UNIT ≈ 4 is the physical light-travel time; do not use it for col 4.

RECOS_FIELDS = (
    "theta",
    "phi",
    "S_800",
    "E_gamma",
    "d_border",
    "chi2_ndof",
    "linsley_curvature",
    "aop_1200",
    "aop_slope",
    "Sb_2p5",
    "Sb_4p0",
    "signal_sum",
    "layer_asymmetry",
    "peak_count",
    "peak_count_largest",
)

# dt_params (num_dets, 6) — phd_work/src/train_VAE/datasets.py
DT_PARAMS_FIELDS = (
    "x_core_relative",  # 1200 m units, core at 0
    "y_core_relative",
    "z",
    "signal_mip",
    "t_plane_mks",
    "t_vs_plane_mks",
)


@dataclass(frozen=True)
class NormParams:
    mean: np.ndarray
    std: np.ndarray

    def denorm(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def norm(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def s_eta(theta_deg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Shower age parameter eta(theta), degrees in / value out."""
    theta = np.deg2rad(theta_deg)
    x = np.asarray(theta_deg, dtype=np.float64)
    e1 = 3.97 - 1.79 * (np.abs(1.0 / np.cos(theta)) - 1.0)
    e2 = (
        (((((-1.71299934e-10 * x + 4.23849411e-08) * x - 3.76192000e-06) * x + 1.35747298e-04)
            * x - 2.18241567e-03) * x + 1.18960682e-02) * x + 3.70692527e+00
    )
    return np.where(x < 62.7, e1, e2)


def s_profile_tasimple(r_ta: np.ndarray, eta: Union[float, np.ndarray]) -> np.ndarray:
    r = np.asarray(r_ta, dtype=np.float64) * UNIT
    rm = 90e2
    r1 = 1000.0e2
    return (r / rm) ** (-1.2) * (1 + r / rm) ** (-(eta - 1.2)) * (1 + (r * r / r1 / r1)) ** (-0.6)


def s_profile(r_ta: np.ndarray, theta_deg: float, eta: Optional[float] = None) -> np.ndarray:
    if eta is None:
        eta = s_eta(theta_deg)
    norm = s_profile_tasimple(np.asarray(R_X), eta)
    return s_profile_tasimple(np.asarray(r_ta, dtype=np.float64), eta) / norm


def linsley_t(r_ta: np.ndarray, s_prof: np.ndarray) -> np.ndarray:
    r = np.asarray(r_ta, dtype=np.float64)
    s = np.asarray(s_prof, dtype=np.float64)
    return 0.67 * (1 + r / LINSLEY_r0) ** 1.5 * np.power(np.maximum(s, 1e-12), -0.5) * NSEC


def shower_geometry(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    theta_deg: float,
    phi_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    r_plane = np.sin(theta) * np.cos(phi) * x + np.sin(theta) * np.sin(phi) * y - np.cos(theta) * z
    r = np.sqrt(np.maximum(x * x + y * y + z * z - r_plane * r_plane, 0.0))
    return r, r_plane


def aprime_from_recos(recos: np.ndarray) -> float:
    """
    Linsley curvature parameter aprime from recos[6].

    iterate.cpp MLX prints (aprime*sqrt(S_X))/sqrt(S800); for a successful fit S_X ~= S800.
    """
    return float(recos[6])


def _brightest_idx(signal: np.ndarray) -> int:
    return int(np.argmax(np.asarray(signal, dtype=np.float64)))


def _affine_through_ref(predictor: np.ndarray, target: np.ndarray, ref: int) -> np.ndarray:
    """
    Best affine map y ≈ y_ref + slope * (x - x_ref) constrained through ref.

    Picks sign and scale of the recos-based predictor to match measured target
    within the event (handles φ/θ convention mismatches vs iterate.cpp).
    """
    x = np.asarray(predictor, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    dx = x - x[ref]
    dy = y - y[ref]
    den = float(np.dot(dx, dx))
    slope = float(np.dot(dy, dx) / den) if den > 1e-12 else 0.0
    return y[ref] + slope * dx


def predict_detector_quantities(
    recos: np.ndarray,
    xyz: np.ndarray,
    t0_mks: float = 0.0,
    observed: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward model for one event in the core-centered frame.

    Args:
        recos: (15,) unnormalized reconstruction parameters.
        xyz: (N, 3) detector positions in 1200 m units; core at origin.
        t0_mks: plane arrival time at the core (µs).
        observed: optional measured dt_params (physical units) for relative
            time anchoring at the brightest detector.

    Returns:
        r, r_plane, signal_mip, t_plane_mks (col 4), t_vs_plane_mks (col 5)
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    theta, phi, s800 = float(recos[0]), float(recos[1]), float(recos[2])
    courve = aprime_from_recos(recos)

    r, r_plane = shower_geometry(x, y, z, theta, phi)
    prof = s_profile(r, theta)
    signal = s800 * prof
    t_d = courve * linsley_t(r, prof)

    if observed is not None and len(observed) > 0:
        ref = _brightest_idx(observed[:, 3])
        # Shape from recos; per-event sign and scale matched to measured col 4/5.
        t_plane_mks = _affine_through_ref(r_plane, observed[:, 4], ref)
        t_vs_plane_mks = _affine_through_ref(t_d, observed[:, 5], ref)
    else:
        # iterate.cpp: t_plane = t0 + r_plane
        t_plane_mks = t0_mks + r_plane
        t_vs_plane_mks = t_d
    return r, r_plane, signal, t_plane_mks, t_vs_plane_mks


def estimate_t0_from_plane_times(
    recos: np.ndarray,
    xyz: np.ndarray,
    t_plane_mks: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Recover t0 (plane time at core) from measured col 4 and core-relative geometry.

    iterate.cpp: t0 = t_plane - r_plane, averaged over detectors.
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    _, r_plane = shower_geometry(x, y, z, float(recos[0]), float(recos[1]))
    t0_per_det = t_plane_mks - r_plane
    if weights is None:
        return float(np.mean(t0_per_det))
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.maximum(w.sum(), 1e-12)
    return float(np.sum(t0_per_det * w))


def reconstruct_dt_params(
    recos: np.ndarray,
    geometry: np.ndarray,
    t0_mks: Optional[float] = None,
    observed: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build dt_params (N, 6) from recos + core-relative detector positions (cols 0-2).

    Geometry is copied unchanged; signal and times (cols 3-5) come from iterate physics.
    If t0_mks is None and observed dt_params are given, t0 is inferred from col 4
    of the brightest detector (iterate.cpp uses the same reference).
    """
    geometry = np.asarray(geometry, dtype=np.float64)
    if geometry.ndim != 2 or geometry.shape[1] < 3:
        raise ValueError("geometry must have shape (N, >=3)")

    if t0_mks is None:
        if observed is None:
            t0_mks = 0.0
        else:
            t0_mks = estimate_t0_mks(recos, observed)

    _, _, signal, t_plane, t_vs_plane = predict_detector_quantities(
        recos, geometry[:, :3], t0_mks, observed=observed
    )
    out = np.zeros((len(geometry), 6), dtype=np.float64)
    out[:, :3] = geometry[:, :3]
    out[:, 3] = signal
    out[:, 4] = t_plane
    out[:, 5] = t_vs_plane
    return out


def estimate_t0_mks(recos: np.ndarray, dt_params: np.ndarray) -> float:
    """Estimate core plane time t0 from the brightest detector (iterate.cpp convention)."""
    dt_params = np.asarray(dt_params, dtype=np.float64)
    idx = int(np.argmax(dt_params[:, 3]))
    return estimate_t0_from_plane_times(
        recos,
        dt_params[idx : idx + 1, :3],
        dt_params[idx : idx + 1, 4],
    )


def compare_dt_params(
    recos: np.ndarray,
    dt_params: np.ndarray,
    t0_mks: Optional[float] = None,
) -> dict:
    """Compare forward prediction with measured dt_params."""
    pred = reconstruct_dt_params(recos, dt_params[:, :3], t0_mks=t0_mks, observed=dt_params)
    obs_sig, pred_sig = dt_params[:, 3], pred[:, 3]
    obs_t4, pred_t4 = dt_params[:, 4], pred[:, 4]
    obs_t5, pred_t5 = dt_params[:, 5], pred[:, 5]
    mask = obs_sig > 0.1
    sig_corr = float(np.corrcoef(obs_sig[mask], pred_sig[mask])[0, 1]) if mask.sum() > 1 else np.nan
    t4_corr = float(np.corrcoef(obs_t4[mask], pred_t4[mask])[0, 1]) if mask.sum() > 1 else np.nan
    t5_corr = float(np.corrcoef(obs_t5[mask], pred_t5[mask])[0, 1]) if mask.sum() > 1 else np.nan
    rel_err = np.abs(obs_sig[mask] - pred_sig[mask]) / np.maximum(obs_sig[mask], 1e-6)
    return {
        "signal_correlation": sig_corr,
        "plane_time_correlation": t4_corr,
        "waveform_time_correlation": t5_corr,
        "signal_median_relative_error": float(np.median(rel_err)) if rel_err.size else np.nan,
        "signal_mean_relative_error": float(np.mean(rel_err)) if rel_err.size else np.nan,
        "t0_mks": estimate_t0_mks(recos, dt_params) if t0_mks is None else t0_mks,
        "predicted": pred,
    }


class H5EventReader:
    """Read aligned recos / dt_params slices from normalized phd_work h5 files."""

    def __init__(self, path: str, split: str = "train"):
        self.path = path
        self.split = split
        self._file: Optional[h5.File] = None
        self._dt_norm: Optional[NormParams] = None
        self._recos_norm: Optional[NormParams] = None

    def open(self) -> "H5EventReader":
        self._file = h5.File(self.path, "r")
        grp = self._file[self.split]
        self._ev_starts = np.asarray(grp["ev_starts"][()])
        dt_np = self._file["norm_param/dt_params"]
        self._dt_norm = NormParams(dt_np["mean"][()], dt_np["std"][()])
        rec_np = self._file["norm_param/recos"]
        self._recos_norm = NormParams(rec_np["mean"][()], rec_np["std"][()])
        self._recos_raw = np.asarray(grp["recos"][()])
        self._dt_raw = np.asarray(grp["dt_params"][()])
        return self

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "H5EventReader":
        return self.open()

    def __exit__(self, *args) -> None:
        self.close()

    @property
    def num_events(self) -> int:
        return len(self._ev_starts) - 1

    def get_event(self, index: int, denorm: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        st, fn = self._ev_starts[index], self._ev_starts[index + 1]
        recos = self._recos_raw[index].copy()
        dt = self._dt_raw[st:fn].copy()
        if denorm:
            recos = self._recos_norm.denorm(recos)
            dt = self._dt_norm.denorm(dt)
        return recos, dt

    def reconstruct_event(self, index: int, t0_mks: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        recos, dt = self.get_event(index)
        pred = reconstruct_dt_params(recos, dt[:, :3], t0_mks=t0_mks, observed=dt)
        return recos, dt, pred


def _demo(path: Optional[str] = None, n_events: int = 5) -> None:
    candidates = [
        path,
        "/home3/rfit/Telescope_Array/phd_work/data/normed/Ivan_Kharuk_pr_ga_all_0001_eq_eff_normed_one_work.h5",
        "/home/rfit/Telescope_Array/phd_work/data/normed/Ivan_Kharuk_pr_ga_all_0001_eq_eff_normed_one_work.h5",
    ]
    h5_path = next(p for p in candidates if p and __import__("os").path.exists(p))

    with H5EventReader(h5_path) as reader:
        print(f"file: {h5_path}")
        print(f"events: {reader.num_events}")
        for i in range(min(n_events, reader.num_events)):
            recos, dt, pred = reader.reconstruct_event(i)
            stats = compare_dt_params(recos, dt)
            print(
                f"event {i}: ndet={len(dt)} "
                f"sig_corr={stats['signal_correlation']:.4f} "
                f"t_plane_corr={stats['plane_time_correlation']:.4f} "
                f"t_wf_corr={stats['waveform_time_correlation']:.4f} "
                f"sig_med_err={stats['signal_median_relative_error']:.3f}"
            )


if __name__ == "__main__":
    _demo()
