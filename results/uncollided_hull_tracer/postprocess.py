#!/usr/bin/env python3
"""Reduces the Kobayashi point-source runs to tables and plots.

usage: postprocess.py

Reads runs/cases.json and the files of each case, written by run_study.py, and writes
plots/*.png, data/*.json and the tables of SUMMARY.md to data/tables.md.

The reference is the exact uncollided flux of the point source,

    phi(x) = S exp(-tau(x)) / (4 pi |x|^2),

with the optical depth tau computed from the benchmark geometry, not from the mesh: the straight
path from the origin to x is clipped against the four boxes of the duct, and
tau = sigma_shield |x| - (sigma_shield - sigma_duct) * (length of the path in the duct).

The numerical flux of a tetrahedron is linear, given by its four nodal values, so it is
evaluated exactly at the points of a quadrature of degree 4 on each tetrahedron. The source
region [0, 10]^3 is left out of the error norms: the flux is singular at the source and its L2
norm over that region is infinite.
"""

import csv
import json
import os
import re

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
PLOTS = os.path.join(HERE, "plots")
DATA = os.path.join(HERE, "data")

SOURCE_STRENGTH = 1000.0
SIGMA_SHIELD = 0.1
SIGMA_DUCT = 1.0e-4
# The four boxes of the dog-leg duct, (lower corner, upper corner), in cm.
DUCT = [
    ((0.0, 10.0, 0.0), (10.0, 50.0, 10.0)),
    ((0.0, 50.0, 0.0), (30.0, 60.0, 10.0)),
    ((30.0, 50.0, 0.0), (40.0, 60.0, 30.0)),
    ((30.0, 50.0, 30.0), (40.0, 100.0, 40.0)),
]
SOURCE_REGION = ((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))

# Quadrature of degree 4 on the unit tetrahedron (Xiao and Gimbutas 2010), as in Table A.2 of
# Woodsford, Ragusa and Morel (2026): x, y, z, weight. The weights sum to 1/6.
TET_RULE = np.array([
    [9.720464e-02, 1.066042e-01, 6.843904e-01, 1.774467e-02],
    [2.956950e-02, 3.292330e-01, 3.179036e-01, 1.837237e-02],
    [4.327102e-01, 1.038441e-01, 3.538232e-01, 2.582935e-02],
    [2.402767e-01, 3.044484e-01, 1.268017e-01, 3.223514e-02],
    [1.294114e-01, 5.380072e-01, 3.301904e-01, 1.269379e-02],
    [1.215415e-01, 8.991260e-03, 3.064940e-01, 1.323778e-02],
    [4.507659e-01, 4.329535e-01, 5.945662e-02, 1.157833e-02],
    [4.192663e-01, 5.334124e-02, 4.778144e-02, 9.988864e-03],
    [6.722329e-02, 7.412289e-01, 3.518393e-02, 9.232210e-03],
    [7.525085e-01, 8.140492e-02, 6.809937e-02, 9.212228e-03],
    [4.049051e-02, 1.746941e-01, 1.356070e-02, 6.541848e-03],
])

STRATEGY_LABELS = {
    "A": "A: sweep + cell-walk near source",
    "B": "B: sweep + hull near source",
    "C": "C: every cell hull-traced",
}
COLORS = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}


def length_in_box(points, lower, upper):
    """Length of the straight path from the origin to each point that lies in a box."""
    enter = np.zeros(len(points))
    leave = np.ones(len(points))
    for axis in range(3):
        x = points[:, axis]
        with np.errstate(divide="ignore", invalid="ignore"):
            t0 = np.where(x != 0.0, lower[axis] / x, np.where(lower[axis] <= 0.0, -np.inf, np.inf))
            t1 = np.where(x != 0.0, upper[axis] / x, np.where(upper[axis] >= 0.0, np.inf, -np.inf))
        enter = np.maximum(enter, np.minimum(t0, t1))
        leave = np.minimum(leave, np.maximum(t0, t1))
    return np.maximum(0.0, leave - enter) * np.linalg.norm(points, axis=1)


def exact_flux(points):
    radius = np.linalg.norm(points, axis=1)
    in_duct = sum(length_in_box(points, lower, upper) for lower, upper in DUCT)
    optical_depth = SIGMA_SHIELD * radius - (SIGMA_SHIELD - SIGMA_DUCT) * in_duct
    return SOURCE_STRENGTH * np.exp(-optical_depth) / (4.0 * np.pi * radius**2)


def inside(points, box):
    lower, upper = np.array(box[0]), np.array(box[1])
    return np.all((points >= lower) & (points <= upper), axis=1)


def read_flux(prefix):
    """Cell ids, nodes [cell, 4, 3] and nodal fluxes [cell, 4] of a tetrahedral run."""
    with h5py.File(os.path.join(RUNS, prefix + ".h5"), "r") as file:
        counts = file["cell node counts"][()]
        if not np.all(counts == 4):
            return None
        nodes = np.stack([file["nodes x"][()], file["nodes y"][()], file["nodes z"][()]], axis=1)
        return file["cell ids"][()], nodes.reshape(-1, 4, 3), file["0,0"][()].reshape(-1, 4)


def error_norms(prefix):
    """Error norms of a tetrahedral run against the exact flux."""
    flux = read_flux(prefix)
    if flux is None:
        return None
    cell_ids, nodes, phi = flux
    edges = nodes[:, 1:, :] - nodes[:, :1, :]
    volume6 = np.abs(np.linalg.det(edges))
    centroids = nodes.mean(axis=1)

    squared_error = np.zeros(len(nodes))
    squared_norm = np.zeros(len(nodes))
    for x, y, z, weight in TET_RULE:
        barycentric = np.array([1.0 - x - y - z, x, y, z])
        points = np.einsum("i,cid->cd", barycentric, nodes)
        difference = phi @ barycentric - exact_flux(points)
        squared_error += weight * volume6 * difference**2
        squared_norm += weight * volume6 * exact_flux(points) ** 2

    outside_source = ~inside(centroids, SOURCE_REGION)
    in_duct = np.zeros(len(nodes), dtype=bool)
    for box in DUCT:
        in_duct |= inside(centroids, box)
    regions = {
        "outside source region": outside_source,
        "duct": in_duct,
        "shield": outside_source & ~in_duct,
    }
    norms = {name: float(np.sqrt(squared_error[mask].sum() / squared_norm[mask].sum()))
             for name, mask in regions.items()}
    cell_errors = np.sqrt(squared_error / squared_norm)
    return {
        "num_cells": int(len(nodes)),
        "relative_l2": norms,
        "cell_errors": cell_errors[outside_source],
        "max_cell_error": float(cell_errors[outside_source].max()),
        "median_cell_error": float(np.median(cell_errors[outside_source])),
        "cell_ids": cell_ids,
        "centroids": centroids,
    }


def read_diagnostics(prefix):
    with h5py.File(os.path.join(RUNS, prefix + "_diagnostics.h5"), "r") as file:
        return {name: file[name][()] for name in file.keys()}


def read_lineout(prefix):
    with open(os.path.join(RUNS, prefix + "_lineout.csv")) as file:
        rows = list(csv.DictReader(file))
    s = np.array([float(row["s"]) for row in rows])
    points = np.array([[float(row[k]) for k in "xyz"] for row in rows])
    return s, points, np.array([float(row["phi"]) for row in rows])


def read_log(prefix):
    """Phase times and conservation report of a run, from its log."""
    text = open(os.path.join(RUNS, prefix + ".log")).read()
    result = {}
    phases = ["Material hull", "Cell ordering", "Ray tracing", "Conservation correction",
              "Bulk sweep", "Image sources", "Balance and moments", "Output"]
    for phase in phases:
        match = re.search(rf"{phase}\s+= ([0-9.eE+-]+)", text)
        result[phase] = float(match.group(1)) if match else float("nan")
    for key, pattern in [
        ("alpha_min", r"Smallest scale factor alpha\s+= ([0-9.eE+-]+)"),
        ("alpha_max", r"Largest scale factor alpha\s+= ([0-9.eE+-]+)"),
        ("max_cell_residual", r"Largest cell balance residual\s+= ([0-9.eE+-]+)"),
        ("conservation_error", r"Conservation error\s+= ([0-9.eE+-]+)"),
        ("hull_triangles", r"Material hull: (\d+) triangles"),
        ("traced_cells", r"Conservation correction of (\d+) cells"),
        ("edge_paths", r"Cell-walk ray tracing: (\d+) paths"),
        ("retraced_paths", r"Material-hull ray tracing: (\d+) paths"),
    ]:
        match = re.search(pattern, text)
        result[key] = float(match.group(1)) if match else None
    match = re.search(r"Cell-group pairs by \|alpha - 1\|\s+: (.*)", text)
    result["alpha_histogram"] = match.group(1).strip() if match else None

    # Program time when the solver starts and ends, and at the end of the run. What precedes the
    # solver is the set-up of the mesh and of the discretization; what follows it is the sampling
    # of the line-out by this study's input.
    def seconds(pattern):
        match = re.search(rf"(\d+):(\d+):([0-9.]+) {pattern}", text)
        return 3600 * int(match.group(1)) + 60 * int(match.group(2)) + float(match.group(3))

    start = seconds("Starting solver execution")
    finish = seconds("Finished solver execution")
    match = re.search(r"Elapsed execution time: (\d+):(\d+):([0-9.]+)", text)
    elapsed = 3600 * int(match.group(1)) + 60 * int(match.group(2)) + float(match.group(3))
    result["setup_time"] = start
    result["solver_time"] = finish - start
    result["lineout_time"] = elapsed - finish
    return result


def scientific(value):
    """A number in scientific notation, or n/a when it was not computed."""
    return "n/a" if value is None or np.isnan(value) else f"{value:.3e}"


def main():
    os.makedirs(PLOTS, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    cases = json.load(open(os.path.join(RUNS, "cases.json")))
    summary = []
    cell_errors = {}
    lineouts = {}
    alphas = {}
    for case in cases:
        prefix = case["prefix"]
        entry = dict(case)
        entry.update(read_log(prefix))
        norms = error_norms(prefix)
        s, points, phi = read_lineout(prefix)
        exact = exact_flux(points)
        entry["lineout_max_relative_error"] = float(np.max(np.abs(phi - exact) / exact))
        lineouts[prefix] = (s, phi, exact)
        diagnostics = read_diagnostics(prefix)
        entry["max_bulk_residual"] = (
            float(diagnostics["bulk balance residual 0"].max())
            if len(diagnostics["bulk balance residual 0"]) else None
        )
        if norms is not None:
            entry["relative_l2"] = norms["relative_l2"]
            entry["max_cell_error"] = norms["max_cell_error"]
            entry["median_cell_error"] = norms["median_cell_error"]
            entry["num_cells"] = norms["num_cells"]
            cell_errors[prefix] = norms["cell_errors"]
            # Scale factor of each traced cell against its distance from the source.
            index = {cell: i for i, cell in enumerate(norms["cell_ids"])}
            rows = np.array([index[cell] for cell in diagnostics["cell ids 0"]], dtype=int)
            alphas[prefix] = (np.linalg.norm(norms["centroids"][rows], axis=1),
                              diagnostics["alpha 0"])
        far = np.abs(diagnostics["alpha 0"] - 1.0)
        entry["alpha_fraction_above_1_percent"] = float(np.mean(far > 0.01))
        entry["alpha_fraction_above_10_percent"] = float(np.mean(far > 0.1))
        summary.append(entry)
    json.dump(summary, open(os.path.join(DATA, "summary.json"), "w"), indent=1)

    serial = [e for e in summary if e["ranks"] == 1]
    order = ["coarse", "fine", "xfine", "polyhedra"]
    meshes = sorted({e["mesh"] for e in summary}, key=order.index)

    # Line-outs along the centerline of the duct.
    for mesh in meshes:
        figure, (top, bottom) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                             gridspec_kw={"height_ratios": [2, 1]})
        entries = [e for e in serial if e["mesh"] == mesh]
        s, _, exact = lineouts[entries[0]["prefix"]]
        top.semilogy(s, exact, "k-", linewidth=1.2, label="exact")
        for entry in entries:
            s, phi, exact = lineouts[entry["prefix"]]
            color = COLORS[entry["strategy"]]
            top.semilogy(s, np.abs(phi), "--", color=color, linewidth=1.0,
                         label=STRATEGY_LABELS[entry["strategy"]])
            bottom.semilogy(s, np.abs(phi - exact) / exact, color=color, linewidth=1.0)
        top.set_ylabel("uncollided scalar flux")
        top.set_title(f"Centerline of the duct, {mesh} mesh "
                      f"({entries[0].get('num_cells', 'n/a')} cells)")
        top.legend()
        top.grid(True, which="both", alpha=0.3)
        bottom.set_xlabel("path length along the centerline (cm)")
        bottom.set_ylabel("relative error")
        bottom.grid(True, which="both", alpha=0.3)
        figure.tight_layout()
        figure.savefig(os.path.join(PLOTS, f"lineout_{mesh}.png"), dpi=150)
        plt.close(figure)

    # Histograms of the cell-wise errors.
    for mesh in meshes:
        entries = [e for e in serial if e["mesh"] == mesh and e["prefix"] in cell_errors]
        if not entries:
            continue
        figure, axes = plt.subplots(len(entries), 1, figsize=(8, 2.6 * len(entries)), sharex=True)
        bins = np.logspace(-6, 1, 71)
        for axis, entry in zip(np.atleast_1d(axes), entries):
            axis.hist(np.clip(cell_errors[entry["prefix"]], bins[0], bins[-1]), bins=bins,
                      color=COLORS[entry["strategy"]])
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_ylabel("cells")
            axis.set_title(STRATEGY_LABELS[entry["strategy"]], fontsize=10)
            axis.grid(True, which="both", alpha=0.3)
        np.atleast_1d(axes)[-1].set_xlabel("relative L2 error of the cell")
        figure.suptitle(f"Cell-wise errors outside the source region, {mesh} mesh")
        figure.tight_layout()
        figure.savefig(os.path.join(PLOTS, f"cell_errors_{mesh}.png"), dpi=150)
        plt.close(figure)

    # Scale factor against the distance from the source, strategy C.
    for mesh in meshes:
        entries = [e for e in serial if e["mesh"] == mesh and e["strategy"] == "C"
                   and e["prefix"] in alphas]
        if not entries:
            continue
        radius, alpha = alphas[entries[0]["prefix"]]
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.semilogy(radius, np.maximum(np.abs(alpha - 1.0), 1.0e-16), ".", markersize=1.5,
                      color=COLORS["C"], alpha=0.5)
        axis.set_xlabel("distance of the cell centroid from the source (cm)")
        axis.set_ylabel("|alpha - 1|")
        axis.set_title(f"Scale factor of the conservation correction, every cell traced, "
                       f"{mesh} mesh")
        axis.grid(True, which="both", alpha=0.3)
        figure.tight_layout()
        figure.savefig(os.path.join(PLOTS, f"alpha_{mesh}.png"), dpi=150)
        plt.close(figure)

    # Wall time by phase.
    phases = ["Material hull", "Cell ordering", "Ray tracing", "Conservation correction",
              "Bulk sweep", "Balance and moments", "Output"]
    for mesh in meshes:
        entries = [e for e in summary if e["mesh"] == mesh]
        labels = [f"{e['strategy']}, {e['ranks']} rank{'s' if e['ranks'] > 1 else ''}"
                  for e in entries]
        figure, axis = plt.subplots(figsize=(8, 4.5))
        bottom = np.zeros(len(entries))
        for phase in phases:
            values = np.array([e[phase] for e in entries])
            axis.bar(labels, values, bottom=bottom, label=phase)
            bottom += values
        axis.set_ylabel("wall time of rank 0 (s)")
        axis.set_title(f"Uncollided generation time by phase, {mesh} mesh")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(os.path.join(PLOTS, f"timing_{mesh}.png"), dpi=150)
        plt.close(figure)

    # Tables.
    lines = []
    for mesh in meshes:
        entries = [e for e in summary if e["mesh"] == mesh]
        lines.append(f"### {mesh} mesh ({entries[0].get('num_cells', 'n/a')} cells)\n")
        lines.append("| Strategy | Ranks | L2 error outside source region | L2 error, duct | "
                     "L2 error, shield | Median cell error | Max line-out error |")
        lines.append("|---|---|---|---|---|---|---|")
        for e in entries:
            l2 = e.get("relative_l2", {})
            lines.append(
                f"| {e['strategy']} | {e['ranks']} "
                f"| {scientific(l2.get('outside source region'))} "
                f"| {scientific(l2.get('duct'))} | {scientific(l2.get('shield'))} "
                f"| {scientific(e.get('median_cell_error'))} "
                f"| {e['lineout_max_relative_error']:.3e} |")
        lines.append("")
        lines.append("| Strategy | Ranks | Traced cells | alpha min | alpha max | "
                     "Cells with abs(alpha - 1) > 1% | > 10% | Max cell balance residual, traced | "
                     "swept | Global balance |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for e in entries:
            bulk = e["max_bulk_residual"]
            lines.append(
                f"| {e['strategy']} | {e['ranks']} | {int(e['traced_cells'])} "
                f"| {e['alpha_min']:.4f} | {e['alpha_max']:.4f} "
                f"| {100 * e['alpha_fraction_above_1_percent']:.3f}% "
                f"| {100 * e['alpha_fraction_above_10_percent']:.3f}% "
                f"| {e['max_cell_residual']:.1e} | {'n/a' if bulk is None else f'{bulk:.1e}'} "
                f"| {e['conservation_error']:.1e} |")
        lines.append("")
        lines.append("| Strategy | Ranks | Solver (s) | Hull | Ordering | Ray tracing | "
                     "Correction | Bulk sweep | Balance and moments | Output | Set-up (s) | "
                     "Line-out sampling (s) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for e in entries:
            lines.append(
                f"| {e['strategy']} | {e['ranks']} | {e['solver_time']:.1f} "
                f"| {e['Material hull']:.3f} | {e['Cell ordering']:.2f} | {e['Ray tracing']:.2f} "
                f"| {e['Conservation correction']:.2f} | {e['Bulk sweep']:.2f} "
                f"| {e['Balance and moments']:.2f} | {e['Output']:.2f} "
                f"| {e['setup_time']:.1f} | {e['lineout_time']:.1f} |")
        lines.append("")
    open(os.path.join(DATA, "tables.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
