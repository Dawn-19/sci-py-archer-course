import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
import matplotlib.pyplot as plt

data_file = Path(__file__).resolve().parent / "dn_dM.txt"
data = np.loadtxt(data_file, skiprows=1)

# Number of mass samples for each redshift
n_mass = np.sum(np.isclose(data[:, 0], data[0, 0]))
z_grid = data[::n_mass, 0]
log10_1pz_grid = np.log10(1.0 + z_grid)
log10M_grid = np.log10(data[:n_mass, 1])
dn_dM_grid = data[:, 2].reshape(len(z_grid), n_mass)

dn_dM_interp = RegularGridInterpolator(
    (log10_1pz_grid, log10M_grid),
    dn_dM_grid,
    method="linear",
    bounds_error=False,
    fill_value=None,
)


def mf(M, z):
    z_arr, M_arr = np.broadcast_arrays(np.asarray(z, dtype=float), np.asarray(M, dtype=float))
    if np.any(M_arr <= 0.0):
        raise ValueError("M must be > 0 for log10(M).")
    if np.any(z_arr <= -1.0):
        raise ValueError("z must be > -1 for log10(1+z).")
    points = np.column_stack((np.log10(1.0 + z_arr.ravel()), np.log10(M_arr.ravel())))
    values = dn_dM_interp(points).reshape(z_arr.shape)
    if values.shape == ():
        return float(values)
    return values


def check_interpolator_on_grid():
    M_grid = 10.0 ** log10M_grid
    zz, mm = np.meshgrid(z_grid, M_grid, indexing="ij")
    pred = mf(mm, zz)

    abs_err = np.abs(pred - dn_dM_grid)
    max_abs_err = np.max(abs_err)

    mask = dn_dM_grid != 0.0
    rel_err = np.zeros_like(dn_dM_grid)
    rel_err[mask] = abs_err[mask] / np.abs(dn_dM_grid[mask])
    max_rel_err = np.max(rel_err)

    print(f"grid self-check: max_abs_err = {max_abs_err:.3e}")
    print(f"grid self-check: max_rel_err = {max_rel_err:.3e}")


def plot_interpolation_vs_raw():
    M_grid = 10.0 ** log10M_grid
    z_samples = [4.0, 8.0, 15.0, 30.0, 40.0]
    M_dense = np.logspace(log10M_grid[0], log10M_grid[-1], 500)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: curves at selected redshifts
    for z0 in z_samples:
        iz = int(np.argmin(np.abs(z_grid - z0)))
        z_use = z_grid[iz]
        raw = dn_dM_grid[iz, :]
        fit = mf(M_dense, z_use)

        axes[0].loglog(M_dense, fit, lw=1.8, label=f"interp z={z_use:.1f}")
        axes[0].loglog(
            M_grid[::40],
            raw[::40],
            "o",
            ms=3,
            alpha=0.85,
            label=f"raw z={z_use:.1f}",
        )

    axes[0].set_xlabel("M [Msun]")
    axes[0].set_ylabel("dn/dM [1 / (Msun Mpc^3)]")
    axes[0].set_title("Interpolation vs raw (selected z slices)")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.3, which="both")

    # Right panel: parity on a subset of raw grid points
    zz, mm = np.meshgrid(z_grid[::20], M_grid[::20], indexing="ij")
    raw_sub = dn_dM_grid[::20, ::20]
    fit_sub = mf(mm, zz)

    x = raw_sub.ravel()
    y = fit_sub.ravel()
    positive = (x > 0) & (y > 0)
    x = x[positive]
    y = y[positive]

    axes[1].loglog(x, y, ".", ms=3, alpha=0.7)
    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    axes[1].loglog([lo, hi], [lo, hi], "k--", lw=1.2)
    axes[1].set_xlabel("raw dn/dM")
    axes[1].set_ylabel("interpolated dn/dM")
    axes[1].set_title("Parity plot (subset of grid points)")
    axes[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = data_file.parent / "interpolation_vs_raw.png"
    fig.savefig(out, dpi=200)
    print(f"saved figure: {out}")


if __name__ == "__main__":
    z = 4.13
    M = 1.2e5
    aa = mf(M, z)
    print(f"dn/dM(z={z}, M={M:.3e}) = {aa:.6e}")
    check_interpolator_on_grid()
    plot_interpolation_vs_raw()
