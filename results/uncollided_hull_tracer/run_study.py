#!/usr/bin/env python3
"""Runs the Kobayashi point-source cases of the uncollided ray-tracer comparison.

usage: run_study.py [--exe PATH] [--meshes fine xfine] [--ranks 1 4 16] [--threads 1]
                    [--strategies A B C] [--repeat N]

Each case runs kobayashi_point_source.py with mpiexec in runs/, keeps its log and output files
there, and is recorded in runs/cases.json with its wall time. Strategies A and B need one MPI
rank and run on one; strategy C runs on every rank count given. With --repeat, each case runs
N times and the shortest wall time and its log are kept.

The material-hull ray tracer searches the hull for every path (hull_tracer_search="segment"),
as it did when the results of SUMMARY.md were taken; ../hull_tracer_search compares the
searches.

runs/ is not tracked: it holds the HDF5 outputs, which postprocess.py reduces to the tables and
plots that are.
"""

import argparse
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", default=os.path.join(REPO, "build", "python", "opensn"))
    parser.add_argument("--meshes", nargs="+", default=["fine", "xfine"])
    parser.add_argument("--strategies", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    runs = os.path.join(HERE, "runs")
    os.makedirs(runs, exist_ok=True)
    cases_file = os.path.join(runs, "cases.json")
    cases = {}
    if os.path.exists(cases_file):
        cases = {case["prefix"]: case for case in json.load(open(cases_file))}

    mesh_dir = os.path.join(REPO, "test", "assets", "mesh", "kobayashi")
    env = dict(os.environ, OPENSN_NUM_THREADS=str(args.threads))
    for mesh in args.meshes:
        for strategy in args.strategies:
            for ranks in args.ranks if strategy == "C" else [1]:
                prefix = f"kobayashi_{mesh}_{strategy}_np{ranks}_nt{args.threads}"
                command = [
                    "mpiexec", "-n", str(ranks), args.exe,
                    "-i", os.path.join(HERE, "kobayashi_point_source.py"),
                    "--suppress-color",
                    "--py", f'mesh_name="{mesh}"',
                    "--py", f'strategy="{strategy}"',
                    "--py", f'prefix="{prefix}"',
                    "--py", f'mesh_dir="{mesh_dir}"',
                    "--py", 'hull_tracer_search="segment"',
                ]
                best = None
                for _ in range(args.repeat):
                    start = time.time()
                    result = subprocess.run(command, cwd=runs, env=env, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True)
                    wall = time.time() - start
                    if result.returncode != 0 or "KobayashiPointSourceDone" not in result.stdout:
                        open(os.path.join(runs, prefix + ".log"), "w").write(result.stdout)
                        raise RuntimeError(f"{prefix} failed; see runs/{prefix}.log")
                    if best is None or wall < best:
                        best = wall
                        open(os.path.join(runs, prefix + ".log"), "w").write(result.stdout)
                cases[prefix] = {"prefix": prefix, "mesh": mesh, "strategy": strategy,
                                 "ranks": ranks, "threads": args.threads, "wall_time": best}
                json.dump(list(cases.values()), open(cases_file, "w"), indent=1)
                print(f"{prefix}: {best:.1f} s", flush=True)


if __name__ == "__main__":
    main()
