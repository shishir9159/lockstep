"""Method matrix for one run: tables in matrix.md (also printed), charts, report.html.

    uv run report.py results/<run>

Reads what the run produced: steps.csv, train.csv, wire.csv, bench.json,
train_step.json, curves/ and run.log (rig output). Missing pieces are skipped.
"""

import base64
import csv
import html
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                          # headless
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",       # validated categorical order
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BASE = "fp32 / dense"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 9,
    "font.family": "sans-serif", "lines.linewidth": 2, "lines.solid_capstyle": "round",
})


# ------------------------------------------------------------------- inputs
def sections(log):
    """Split run.log on the '== name  [...]' step markers printed by run.sh."""
    parts = re.split(r"^== (\S+).*$", log, flags=re.M)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def block(text, start, end):
    """Lines of rig output from the '[start]' header to the next '[end]' header."""
    m = re.search(rf"^\[{start}\].*?(?=^\[{end}\]|\Z)", text, flags=re.M | re.S)
    return m.group(0).splitlines() if m else []


def parse_splitk(rig):
    rows, S = {}, None
    for line in block(rig, 5, 6) or rig.splitlines():
        m = re.match(r"\s+(\d+)\s+\d+\s+(?:safe|OVERFL)\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+int32x2",
                     line)
        if m:
            S = int(m.group(1))
            rows[S] = {"int32 x2": float(m.group(2))}
        elif S and (m := re.match(r"\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+packed", line)):
            rows[S]["packed int16"] = float(m.group(1))
        elif S and (m := re.match(r"\s+[\d.]+\s+-\s+([\d.]+)\s+atomics", line)):
            rows[S]["atomics"] = float(m.group(1))
    return rows


def parse_fair(rigq):
    out = {}
    for line in block(rigq, 13, 14):
        if m := re.match(r"\s+(\d+)\s{2}(\S.*?\S)\s+\d+\.\d+\s+(\d+\.\d+)%$", line):
            out.setdefault(m.group(2), {})[int(m.group(1))] = float(m.group(3))
    return out


def parse_ef(rigq):
    out = {}
    pat = (r"\s{4}(\S.*?\S)\s+\d\.\d{3}\s+[\d.]+%"
           r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+%\s+[\d.]+%$")
    for line in block(rigq, 15, 16):
        if m := re.match(pat, line):
            out[m.group(1)] = [float(m.group(i)) for i in (2, 3, 4)]
    return out


def parse_predict(rigq):
    out = {}
    for line in block(rigq, 16, 17):
        if m := re.match(r"\s{4}(\S.*?\S)\s+[\d.]+%\s+[\d.]+%\s+[\d.]+%\s+([\d.]+)%$", line):
            out[m.group(1)] = float(m.group(2))
    return out


def read_csv(path):
    return list(csv.DictReader(open(path))) if path.exists() else []


# ------------------------------------------------------------------- tables
def table(title, head, rows, note=""):
    """A markdown table padded to line up in a terminal."""
    rows = [[str(c) for c in r] for r in rows]
    w = [max(len(str(h)), *(len(r[i]) for r in rows)) for i, h in enumerate(head)]
    line = lambda cells: "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cells)) + " |"
    out = [f"## {title}", "", line(head), "|" + "|".join("-" * (x + 2) for x in w) + "|"]
    out += [line(r) for r in rows]
    return "\n".join(out + ([f"\n{note}"] if note else [])) + "\n"


def matrix(d):
    t = []
    if steps := d["steps"]:
        total = sum(float(s["seconds"]) for s in steps)
        t.append(table("Steps", ["step", "GPUs busy", "GPUs idle", "seconds", "status"],
                       [[s["step"], s["gpus_busy"], s["gpus_idle"], s["seconds"], s["status"]]
                        for s in steps], f"Total: {total / 60:.1f} min."))
    if tr := d["train"]:
        tr = sorted(tr, key=lambda r: r["method"] != BASE)
        b = tr[0]
        r0 = tr[0]
        t.append(table(
            f"Training: GPT-2 '{r0['preset']}' on FineWeb, {r0['gpus']} GPU(s), "
            f"{int(r0['tokens']) / 1e6:.1f}M tokens, {r0['compute']} compute",
            ["wire / gemm", "val loss", "Δ vs base", "time s", "× base time", "tok/s",
             "comm ms/step", "wire B/el", "peak GB", "gemm path"],
            [[r["method"], r["val_loss"], f"{float(r['val_loss']) - float(b['val_loss']):+.4f}",
              r["seconds"], f"{float(r['seconds']) / float(b['seconds']):.2f}", r["tok_per_s"],
              r["comm_ms_per_step"], r["wire_B_per_el"], r["peak_GB"], r["gemm_path"]]
             for r in tr],
            f"Baseline: {b['method']}. Same model, data order, seed and schedule in every row."))
    if wr := d["wire"]:
        b = wr[0]
        t.append(table(f"Gradient wire microbench: GPT-2-sized gradient, {b['gpus']} GPUs",
                       ["wire", "B/el", "ms", "× fp32 time", "grad GB/s", "rel err",
                        "order-independent", "time s"],
                       [[r["method"], r["B_per_el"], r["ms"],
                         f"{float(r['ms']) / float(b['ms']):.2f}", r["grad_GBps"], r["rel_err"],
                         r["order_independent"], r["seconds"]] for r in wr]))
    secs = d["secs"]
    if m := d["bench"].get("mxfp4"):
        rows = [[r["name"], f"{r['ms']:.3f}", f"{r['tflops']:.0f}",
                 f"{r['tflops'] / m['rows'][0]['tflops']:.2f}"] for r in m["rows"]]
        cuda = re.search(r"(\d+)x(\d+)x(\d+)\s+([\d.]+) ms\s+([\d.]+) TFLOP/s",
                         secs.get("cuda-kernel", ""))
        if cuda:
            rows.append(["CUDA mma.sync reference", cuda.group(4), f"{float(cuda.group(5)):.0f}",
                         f"{float(cuda.group(5)) / m['rows'][0]['tflops']:.2f}"])
        ops = re.findall(r"^\s+(\d+)\s+([A-Z][\w.]+)$", secs.get("sass", ""), flags=re.M)
        sass = ", ".join(f"{n} {o}" for n, o in ops) or "-"
        t.append(table("MXFP4 GEMM on FP8 tensor cores, 4096^3",
                       ["kernel", "ms", "TFLOP/s", "× bf16"], rows,
                       f"Exactness: {m['mismatches']} mismatches at K={m['exact_k']}, max rel err "
                       f"{m['rel_err']:.1e}. SASS: {sass}."))
    if du := d["bench"].get("dual"):
        b = du["rows"][0]
        t.append(table("Packing: two GEMMs, separate vs one packed accumulator",
                       ["variant", "ms", "TFLOP/s", "× first"],
                       [[r["name"], f"{r['ms']:.3f}", f"{r['tflops']:.0f}",
                         f"{b['ms'] / r['ms']:.2f}"] for r in du["rows"]]))
        t.append(table("Packing: does one accumulator hold both results exactly?",
                       ["K", "slot s", "bits needed", "bf16 exact", "fp8 exact"],
                       [[n["k"], n["s"], n["bits"], f"{100 * n['bf16_exact']:.1f}%",
                         "-" if n["fp8_exact"] is None else f"{100 * n['fp8_exact']:.1f}%"]
                        for n in du["numerics"]]))
    if ts := d["train_step"].get("rows"):
        t.append(table("Packing: one Linear layer, fwd / dgrad / wgrad",
                       ["pass", "variant", "ms", "TFLOP/s", "speedup"],
                       [[r["pass"], r["name"], f"{r['ms']:.3f}", f"{r['tflops']:.0f}",
                         f"{r['speedup']:.2f}"] for r in ts],
                       "Speedup is against each pass's first row."))
    if sk := {S: v for S, v in parse_splitk(secs.get("rig", "")).items() if len(v) == 3}:
        pair = next((ln.strip() for ln in secs.get("rig_q", "").splitlines()
                     if ln.strip().startswith("reduce at S=")), "")
        t.append(table("Packing: split-K partials as int32 pairs, packed int16, or atomics",
                       ["S", "int32 x2 ms", "packed ms", "atomics ms", "packed × int32"],
                       [[S, v["int32 x2"], v["packed int16"], v["atomics"],
                         f"{v['int32 x2'] / v['packed int16']:.2f}"]
                        for S, v in sorted(sk.items())], pair))
    if fair := parse_fair(secs.get("rig_q", "")):
        ps = sorted({p for v in fair.values() for p in v})
        t.append(table("Simulated wire [13]: relative error by GPU count", ["wire", *map(str, ps)],
                       [[k, *(f"{v[p]}%" for p in ps)] for k, v in fair.items()]))
    if ef := parse_ef(secs.get("rig_q", "")):
        t.append(table("Simulated error feedback [15]: drift after 8 / 32 / 128 steps",
                       ["wire", "8", "32", "128"], [[k, *v] for k, v in ef.items()]))
    return t


# ------------------------------------------------------------------- charts
def hbar(ax, labels, values, fmt, ref=()):
    """Single-series horizontal bars from a zero baseline, value at each tip."""
    y = range(len(labels))[::-1]
    ax.barh(list(y), values, height=0.55, color=SERIES[0])
    top = max(values + [r for r, _ in ref]) if values else 1
    for yi, v in zip(y, values, strict=True):
        ax.text(v + top * 0.01, yi, fmt(v), va="center", fontsize=9, color=INK2, zorder=4,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
    for r, name in ref:
        ax.axvline(r, color=MUTED, linewidth=1, zorder=2)
        ax.text(r, -0.85, f" {name}", fontsize=8, color=MUTED, va="center")
    ax.set_yticks(list(y), labels)
    ax.set_ylim(-1.1 if ref else -0.6, len(labels) - 0.4)
    ax.set_xlim(0, top * 1.18)
    ax.grid(axis="y", visible=False)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return path


def ema(xs, a=0.9):
    out, m = [], xs[0]
    for x in xs:
        m = a * m + (1 - a) * x
        out.append(m)
    return out


def charts(run, d):
    plots = run / "plots"
    plots.mkdir(exist_ok=True)
    made, secs = [], d["secs"]
    curves = {p.stem: read_csv(p) for p in sorted((run / "curves").glob("*.csv"))}
    if "fp32-dense" in curves and len(curves) > 1:
        base = ema([float(r["loss"]) for r in curves["fp32-dense"]])
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, (name, rows) in enumerate(c for c in curves.items() if c[0] != "fp32-dense"):
            loss = ema([float(r["loss"]) for r in rows])
            ax.plot([int(r["tokens"]) / 1e6 for r in rows],
                    [a - b for a, b in zip(loss, base, strict=False)],
                    color=SERIES[i % len(SERIES)], label=name.replace("-", " / "))
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xlabel("tokens (M)")
        ax.set_ylabel("train loss minus fp32 / dense (smoothed)")
        ax.set_title("Training: loss difference from the baseline, same batches")
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3)
        made.append(("Loss vs baseline", save(fig, plots / "loss_delta.png")))
    if tr := d["train"]:
        tr = sorted(tr, key=lambda r: r["method"] != BASE)
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(tr) + 1.4))
        hbar(ax, [f"{r['method']}  (val {float(r['val_loss']):.3f})" for r in tr],
             [float(r["seconds"]) for r in tr], lambda v: f"{v:.0f} s")
        ax.set_title("Training: time to finish the same token budget")
        ax.set_xlabel("seconds")
        made.append(("Time to finish", save(fig, plots / "train_time.png")))
    if wr := d["wire"]:
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(wr) + 1.4))
        hbar(ax, [f"{r['method']}  (err {r['rel_err']})" for r in wr], [float(r["ms"]) for r in wr],
             lambda v: f"{v:.1f} ms")
        ax.set_title("Gradient wire: one all-reduce of a GPT-2-sized gradient")
        ax.set_xlabel("ms")
        made.append(("Wire microbench", save(fig, plots / "wire.png")))
    if m := d["bench"].get("mxfp4"):
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(m["rows"]) + 1.4))
        hbar(ax, [r["name"] for r in m["rows"]], [r["tflops"] for r in m["rows"]],
             lambda v: f"{v:.0f}", ref=[(989, "bf16 peak"), (1979, "fp8 peak")])
        ax.set_title("MXFP4 on H100 FP8 tensor cores")
        ax.set_xlabel("TFLOP/s, 4096^3")
        made.append(("MXFP4 GEMM", save(fig, plots / "mxfp4.png")))
    if du := d["bench"].get("dual"):
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(du["rows"]) + 1.4))
        hbar(ax, [r["name"] for r in du["rows"]], [r["tflops"] for r in du["rows"]],
             lambda v: f"{v:.0f}")
        ax.set_title("Packing: two GEMMs, separate vs one packed accumulator")
        ax.set_xlabel("TFLOP/s (both GEMMs)")
        made.append(("Packed accumulator", save(fig, plots / "dual.png")))
    if (sk := parse_splitk(secs.get("rig", ""))) and all(len(v) == 3 for v in sk.values()):
        S, names = sorted(sk), ["int32 x2", "packed int16", "atomics"]
        fig, ax = plt.subplots(figsize=(8, 3.6))
        for i, n in enumerate(names):
            ax.bar([j + (i - 1) * 0.28 for j in range(len(S))], [sk[s][n] for s in S],
                   width=0.26, color=SERIES[i], label=n)
        ax.set_xticks(range(len(S)), [f"S={s}" for s in S])
        ax.set_ylabel("total ms (GEMM + reduce)")
        ax.set_title("Packing: split-K partials on this GPU")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left")
        made.append(("Split-K transport", save(fig, plots / "splitk.png")))
    return made


# ------------------------------------------------------------------- main
def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")    # Windows consoles: Δ, ×
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    load = lambda n: json.loads((run / n).read_text()) if (run / n).exists() else {}
    log = (run / "run.log").read_text(errors="replace") if (run / "run.log").exists() else ""
    d = {"steps": read_csv(run / "steps.csv"), "train": read_csv(run / "train.csv"),
         "wire": read_csv(run / "wire.csv"), "bench": load("bench.json"),
         "train_step": load("train_step.json"), "secs": sections(log)}
    md = f"# {run.name}\n\n" + "\n".join(matrix(d))
    (run / "matrix.md").write_text(md, encoding="utf-8")
    made = charts(run, d)
    figs = "".join(
        f"<figure><img alt='{html.escape(t)}' src='data:image/png;base64,"
        f"{base64.b64encode(p.read_bytes()).decode()}'><figcaption>{html.escape(t)}</figcaption>"
        f"</figure>" for t, p in made)
    raw = "".join(f"<details><summary>{html.escape(k)}</summary>"
                  f"<pre>{html.escape(v.strip())}</pre></details>" for k, v in d["secs"].items())
    (run / "report.html").write_text(f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>{run.name}</title><style>
body{{margin:0 auto;max-width:1000px;padding:24px 16px;background:#f9f9f7;color:{INK};
font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
pre{{background:{SURFACE};padding:12px;overflow-x:auto;border:1px solid {GRID};
border-radius:6px;font-size:12px}}figure{{margin:16px 0}}
img{{max-width:100%;border:1px solid {GRID};border-radius:6px}}
figcaption{{color:{INK2};font-size:12px}}summary{{cursor:pointer;color:{INK2};padding:4px 0}}
</style></head><body><pre>{html.escape(md)}</pre>{figs}<h2>Raw output</h2>{raw}</body></html>""",
                                      encoding="utf-8")
    print(md)
    print(f"charts: {len(made)} in {run / 'plots'}\nmatrix: {run / 'matrix.md'}\n"
          f"report: {run / 'report.html'}")


if __name__ == "__main__":
    main()
