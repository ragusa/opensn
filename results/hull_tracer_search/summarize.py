#!/usr/bin/env python3
"""Prints the tables of SUMMARY.md from data/search_study_*.csv.

usage: summarize.py data/search_study_np1_nt1.csv
"""

import csv
import sys


def main():
    rows = list(csv.DictReader(open(sys.argv[1])))
    for mesh in dict.fromkeys(row["mesh"] for row in rows):
        cases = [row for row in rows if row["mesh"] == mesh]
        paths = float(cases[0]["Paths"])
        base = float(cases[0]["Ray tracing (s)"])
        base_search = float(cases[0]["Search time (s)"])
        print(f"\n### {mesh} mesh: {int(paths):,} paths, "
              f"{float(cases[0]['Crossings']) / paths:.2f} crossings per path\n")
        print("| Configuration | Bound tests per path | Triangle tests per path | "
              "Candidate triangles | Search (s) | Building (s) | Ray-tracing phase (s) | "
              "Search speed-up | Phase speed-up |")
        print("|---|---|---|---|---|---|---|---|---|")
        for row in cases:
            search = float(row["Search time (s)"])
            phase = float(row["Ray tracing (s)"])
            print(f"| `{row['config']}` | {float(row['Box tests']) / paths:.1f} | "
                  f"{float(row['Triangle tests']) / paths:.1f} | "
                  f"{int(row['Candidate triangles']):,} | {search:.2f} | "
                  f"{float(row['Collection time (s)']):.3f} | {phase:.2f} | "
                  f"{base_search / search:.2f} | {base / phase:.2f} |")


if __name__ == "__main__":
    main()
