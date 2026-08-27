#!/usr/bin/env python3
"""Generate report.md's comparison tables from measured data.

Why this exists: every table in report.md used to be hand-copied out of raw
benchmark logs, and hand-copying kept introducing errors that survived review —
a 22.15 rounded to "22.2", a TTFT/TPOT column order printed backwards, an empty
C2 row, and two per-stream columns silently computed by different formulas
(`e2e / concurrency` for one engine, a measured median for the other), which
made a slower engine look faster at C2.

So: measurements go in `bench/measurements.json`, tables come out of here, and
nothing is retyped. Derived values (ratios, winners) are computed, never typed.

    python bench/make_report.py                # print all tables
    python bench/make_report.py --table vs     # just the vLLM-vs-Ollama table
    python bench/make_report.py --check        # verify internal consistency

The prose in report.md stays hand-written — the analysis is the valuable part
and is not derivable from the numbers. Only the tables are generated.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "measurements.json")


def load():
    if not os.path.exists(DATA):
        sys.exit(f"missing {DATA}\nWrite measurements there (see the docstring).")
    with open(DATA) as f:
        return json.load(f)


def fmt(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def bold(s, on):
    return f"**{s}**" if on else s


# --------------------------------------------------------------- tables ----
def table_vs(d):
    """vLLM vs Ollama at matched concurrency.

    Both per-stream columns are TPOT (ms per output token), because that is a
    measured per-request quantity on both engines. The previous version of this
    table divided vLLM's aggregate by the concurrency and compared it against
    Ollama's measured median — two different quantities in adjacent columns.
    """
    v, o = d["vllm_cohort"], d["ollama_cohort"]
    out = [
        "| cohort | vLLM e2e tok/s | Ollama e2e tok/s | winner | vLLM TPOT | Ollama TPOT |",
        "|---|---|---|---|---|---|",
    ]
    for c in d["concurrency"]:
        k = str(c)
        ve, oe = v[k]["e2e"], o[k]["e2e"]
        ratio = ve / oe
        win = f"**vLLM {ratio:.2f}x**" if ratio > 1 else f"Ollama {1/ratio:.2f}x"
        out.append(
            f"| C{c} | {bold(fmt(ve), ratio > 1)} | {bold(fmt(oe), ratio < 1)} | {win} "
            f"| {fmt(v[k]['tpot'], 2)} ms | {fmt(o[k]['tpot'], 2)} ms |"
        )
    return "\n".join(out)


def table_matrix(d):
    """The 8-config factorial."""
    out = [
        "| config | cudagraph | `SPEC` | `PREFIX_CACHE` | "
        + " | ".join(f"C{c}" for c in d["matrix_concurrency"])
        + " | sat C64 |",
        "|---|---|---|---" + "|---" * (len(d["matrix_concurrency"]) + 1) + "|",
    ]
    best = max(m["cohort"].get("8", 0) for m in d["matrix"].values())
    for label, m in d["matrix"].items():
        cells = [fmt(m["cohort"].get(str(c))) for c in d["matrix_concurrency"]]
        sat = fmt(m.get("saturation"))
        if m.get("crashed"):
            sat += " ⚠"
        star = m["cohort"].get("8", 0) == best
        out.append(
            f"| {bold(label, star)} | {m['cudagraph']} | {m['spec']} | {m['prefix_cache']} | "
            + " | ".join(cells) + f" | {sat} |"
        )
    return "\n".join(out)


def table_cudagraph(d):
    """CUDA graphs on vs off, like for like."""
    on, off = d["matrix"]["A"], d["matrix"]["E"]
    out = ["| workload | graphs on | graphs off | speedup |", "|---|---|---|---|"]
    for c in d["matrix_concurrency"]:
        a, e = on["cohort"].get(str(c)), off["cohort"].get(str(c))
        if a and e:
            out.append(f"| cohort C{c} | {fmt(a)} | {fmt(e)} | **{a/e:.2f}x** |")
    a, e = on["saturation"], off["saturation"]
    out.append(f"| saturation C64 | {fmt(a)} | {fmt(e)} | **{a/e:.2f}x** |")
    return "\n".join(out)


TABLES = {"vs": table_vs, "matrix": table_matrix, "cudagraph": table_cudagraph}


# ---------------------------------------------------------------- checks ----
def check(d):
    """Catch the mistakes that actually happened in this project."""
    problems = []

    # 1. C1 must agree between aggregate and per-request measures: with one
    #    stream the two describe the same thing.
    for eng in ("vllm_cohort", "ollama_cohort"):
        r = d[eng].get("1")
        if r and r.get("tpot"):
            implied = 1000.0 / r["tpot"]
            if abs(implied - r["e2e"]) / r["e2e"] > 0.25:
                problems.append(
                    f"{eng} C1: e2e {r['e2e']:.1f} tok/s vs TPOT {r['tpot']:.2f} ms "
                    f"(implies {implied:.1f} tok/s) — inconsistent")

    # 2. Aggregate throughput must not fall as concurrency rises.
    for eng in ("vllm_cohort", "ollama_cohort"):
        prev = None
        for c in d["concurrency"]:
            e = d[eng][str(c)]["e2e"]
            if prev is not None and e < prev * 0.9:
                problems.append(f"{eng}: e2e drops {prev:.1f} -> {e:.1f} at C{c}")
            prev = e

    # 3. Both engines must have done comparable work.
    for c in d["concurrency"]:
        tv = d["vllm_cohort"][str(c)].get("tokens")
        to = d["ollama_cohort"][str(c)].get("tokens")
        if tv and to and abs(tv - to) / max(tv, to) > 0.10:
            problems.append(f"C{c}: token counts differ >10% ({tv} vs {to})")

    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", choices=sorted(TABLES) + ["all"], default="all")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    d = load()

    if a.check:
        problems = check(d)
        print("\n".join(f"  PROBLEM: {p}" for p in problems) if problems
              else "  consistency checks pass")
        sys.exit(1 if problems else 0)

    names = sorted(TABLES) if a.table == "all" else [a.table]
    for n in names:
        print(f"<!-- generated by bench/make_report.py --table {n} -->")
        print(TABLES[n](d))
        print()


if __name__ == "__main__":
    main()
