#!/usr/bin/env python3
"""Compares the searches of the material hull on the Kobayashi point-source problem.

usage: run_search_study.py [--exe PATH] [--meshes fine xfine] [--ranks 1] [--threads 1]
                           [--repeat 3] [--configs NAME ...]

Every case traces every cell of the mesh against the material hull (strategy C of
../uncollided_hull_tracer) with one configuration of the tracer. A configuration is run
--repeat times with the statistics of the tracer, which give the counts of the work and the
time spent searching the hull, and --repeat times without them, which gives the wall time of
the ray-tracing phase free of the two clock readings per path that the statistics add. The
fastest run of each kind is kept. The counts must be the same in every repeat, and the flux
moments of every configuration must be identical, bit for bit, to those of the first one.

Outputs: runs/ (logs and solver files, not tracked) and data/search_study.csv.
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
INPUT = os.path.join(REPO, "results", "uncollided_hull_tracer", "kobayashi_point_source.py")

# Name of each configuration and the options of the tracer it sets.
CONFIGS = {
    "segment": {"hull_tracer_search": "segment"},
    "segment_leaf1": {"hull_tracer_search": "segment", "hull_tracer_leaf_size": 1},
    "segment_leaf2": {"hull_tracer_search": "segment", "hull_tracer_leaf_size": 2},
    "segment_leaf8": {"hull_tracer_search": "segment", "hull_tracer_leaf_size": 8},
    "segment_leaf16": {"hull_tracer_search": "segment", "hull_tracer_leaf_size": 16},
    "cell_shaft": {"hull_tracer_search": "cell_shaft"},
    "cell_shaft_leaf1": {"hull_tracer_search": "cell_shaft", "hull_tracer_leaf_size": 1},
}
for bins in (16, 32, 64, 128, 256, 512, 1024):
    CONFIGS[f"source_directions_{bins}"] = {"hull_tracer_search": "source_directions",
                                            "hull_tracer_direction_bins": bins}

COUNTS = ["Paths", "Segments", "Box tests", "Triangle tests", "Crossings", "Candidate sets",
          "Candidate box tests", "Candidate triangles"]
TIMES = ["Search time (s)", "Collection time (s)"]
PHASES = ["Material hull", "Ray tracing", "Conservation correction"]


def value_of(log, label):
    """Value printed after 'label =' in the log of the solver, or None."""
    match = re.search(r"^\[0\]\s+" + re.escape(label) + r"\s+= (\S+)", log, re.MULTILINE)
    return None if match is None else match.group(1)


def datasets(file_name):
    """Datasets and attributes of an HDF5 file as bytes."""
    items = {}
    with h5py.File(file_name, "r") as file:
        file.visititems(lambda name, obj: items.update({name: np.asarray(obj[()]).tobytes()})
                        if isinstance(obj, h5py.Dataset) else None)
        for name, value in file.attrs.items():
            items["@" + name] = np.asarray(value).tobytes()
    return items


def run(args, mesh, config, statistics, prefix):
    """Runs one case and returns its log."""
    options = dict(CONFIGS[config], hull_tracer_statistics=statistics, lineout=False)
    command = ["mpiexec", "-n", str(args.ranks), args.exe, "-i", INPUT, "--suppress-color",
               "--py", f'mesh_name="{mesh}"', "--py", 'strategy="C"',
               "--py", f'prefix="{prefix}"',
               "--py", 'mesh_dir="' + os.path.join(REPO, "test", "assets", "mesh",
                                                   "kobayashi") + '"']
    for name, value in options.items():
        command += ["--py", f"{name}={value!r}".replace("'", '"')]
    env = dict(os.environ, OPENSN_NUM_THREADS=str(args.threads))
    result = subprocess.run(command, cwd=os.path.join(HERE, "runs"), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(os.path.join(HERE, "runs", prefix + ".log"), "w") as file:
        file.write(result.stdout)
    if result.returncode != 0 or "KobayashiPointSourceDone" not in result.stdout:
        raise RuntimeError(f"{prefix} failed; see runs/{prefix}.log")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--exe", default=os.path.join(REPO, "build", "python", "opensn"))
    parser.add_argument("--meshes", nargs="+", default=["fine", "xfine"])
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    args = parser.parse_args()
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)

    rows = []
    for mesh in args.meshes:
        reference = None
        for config in args.configs:
            prefix = f"{mesh}_{config}_np{args.ranks}_nt{args.threads}"
            row = {"mesh": mesh, "config": config, "ranks": args.ranks, "threads": args.threads}
            counts = None
            for statistics in (True, False):
                best = None
                for _ in range(args.repeat):
                    log = run(args, mesh, config, statistics, prefix)
                    phases = {phase: float(value_of(log, phase)) for phase in PHASES}
                    if statistics:
                        found = {name: int(value_of(log, name)) for name in COUNTS}
                        if counts is not None and found != counts:
                            raise RuntimeError(f"{prefix}: the counts changed between repeats")
                        counts = found
                    if best is None or phases["Ray tracing"] < best[0]["Ray tracing"]:
                        best = (phases, log)
                phases, log = best
                if statistics:
                    row.update(counts)
                    row.update({name: float(value_of(log, name)) for name in TIMES})
                    row["Ray tracing with statistics (s)"] = phases["Ray tracing"]
                else:
                    row.update({phase + " (s)": phases[phase] for phase in PHASES})
            flux = datasets(os.path.join(HERE, "runs", prefix + ".h5"))
            if reference is None:
                reference = flux
            row["identical to first"] = flux == reference
            if not row["identical to first"]:
                raise RuntimeError(f"{prefix}: the flux moments differ from the first case")
            rows.append(row)
            print(", ".join(f"{name}={value}" for name, value in row.items()))
            sys.stdout.flush()

    name = f"search_study_np{args.ranks}_nt{args.threads}.csv"
    with open(os.path.join(HERE, "data", name), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
