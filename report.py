"""Turn one H100 run into charts, a single-file HTML report and a terminal summary.

    uv run report.py results/<run>      # reads run.log, bench.json, train.json

Writes <run>/plots/*.png and <run>/report.html. Anything missing is skipped.
"""

import base64
import html
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                          # headless: no display over ssh
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # validated categorical order

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


# ------------------------------------------------------------------- parsing
def sections(log):
    """Split run.log on the '== name' markers that `just h100-all` prints."""
    parts = re.split(r"^== (\S+)$", log, flags=re.M)
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
    ax.set_xlim(0, top * 1.15)
    ax.grid(axis="y", visible=False)


def lines(ax, xs, series, logy=True, logx=False):
    """Up to four labelled lines: legend plus a direct label at each line's end."""
    for i, (name, ys) in enumerate(series.items()):
        c = SERIES[i]
        ax.plot(xs, ys, color=c, marker="o", markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=name)
        ax.annotate(name, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.minorticks_off()
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=len(series))
    ax.margins(x=0.05)
    ax.set_xlim(right=xs[-1] * (1.9 if logx else 1.35))


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return path


def charts(run, bench, train, secs):
    plots = run / "plots"
    plots.mkdir(exist_ok=True)
    made = []
    rig, rigq = secs.get("rig", ""), secs.get("rig_q", "")
    if bench.get("mxfp4"):
        rows = bench["mxfp4"]["rows"]
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.4))
        hbar(ax, [r["name"] for r in rows], [r["tflops"] for r in rows], lambda v: f"{v:.0f}",
             ref=[(989, "bf16 peak"), (1979, "fp8 peak")])
        ax.set_title("MXFP4 on H100 FP8 tensor cores")
        ax.set_xlabel("TFLOP/s, 4096^3")
        made.append(("MXFP4 GEMM throughput", save(fig, plots / "mxfp4_tflops.png")))
    if bench.get("dual"):
        rows = bench["dual"]["rows"]
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.4))
        hbar(ax, [r["name"] for r in rows], [r["tflops"] for r in rows], lambda v: f"{v:.0f}")
        ax.set_title("Two GEMMs: separate vs one packed accumulator")
        ax.set_xlabel("TFLOP/s (both GEMMs)")
        made.append(("Packed dual GEMM throughput", save(fig, plots / "dual_tflops.png")))
    if train.get("rows"):
        passes = list(dict.fromkeys(r["pass"] for r in train["rows"]))
        ratios = [sum(r["pass"] == p for r in train["rows"]) for p in passes]
        fig, axes = plt.subplots(len(passes), 1, figsize=(8, 1.2 + 0.45 * len(train["rows"])),
                                 gridspec_kw={"height_ratios": ratios})
        for ax, p in zip(axes, passes, strict=True):
            rows = [r for r in train["rows"] if r["pass"] == p]
            hbar(ax, [r["name"] for r in rows], [r["speedup"] for r in rows], lambda v: f"{v:.2f}x",
                 ref=[(1.0, "baseline")])
            ax.set_title(p, fontsize=10)
        axes[-1].set_xlabel("speedup over the first row of each pass")
        made.append(("One Linear layer, fwd / dgrad / wgrad",
                     save(fig, plots / "train_speedup.png")))
    if (sk := parse_splitk(rig)) and all(len(v) == 3 for v in sk.values()):
        S = sorted(sk)
        names = ["int32 x2", "packed int16", "atomics"]
        fig, ax = plt.subplots(figsize=(8, 3.6))
        w = 0.26
        for i, n in enumerate(names):
            xs = [j + (i - 1) * (w + 0.02) for j in range(len(S))]
            ax.bar(xs, [sk[s][n] for s in S], width=w, color=SERIES[i], label=n)
        ax.set_xticks(range(len(S)), [f"S={s}" for s in S])
        ax.set_ylabel("total ms (GEMM + reduce)")
        ax.set_title("[5] split-K on this GPU: partials as int32 pairs, packed int16, or atomics")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left")
        made.append(("Split-K transport", save(fig, plots / "splitk.png")))
    if fair := parse_fair(rigq):
        keep = ["bf16, ring", "fp16, ring", "int16, ring, oracle grid", "int16, direct"]
        if all(k in fair for k in keep):
            P = sorted(fair[keep[0]])
            fig, ax = plt.subplots(figsize=(8, 4))
            lines(ax, P, {k: [fair[k][p] for p in P] for k in keep}, logx=True)
            ax.set_xlabel("GPUs (P)")
            ax.set_ylabel("relative error of the reduced gradient, %")
            ax.set_title("[13] All-reduce at 2 bytes per element: ring vs direct")
            made.append(("Ring vs direct reduce-scatter", save(fig, plots / "ring_vs_direct.png")))
    if ef := parse_ef(rigq):
        keep = ["bf16 ring", "int8 ring, nearest", "int8 ring, EF", "int8 direct, EF"]
        if all(k in ef for k in keep):
            fig, ax = plt.subplots(figsize=(8, 4))
            lines(ax, [8, 32, 128], {k: ef[k] for k in keep}, logx=True)
            ax.set_xlabel("training steps")
            ax.set_ylabel("drift (steps of gradient lost)")
            ax.set_title("[15] Error feedback: accumulated error over steps")
            made.append(("Error feedback", save(fig, plots / "drift.png")))
    if pr := parse_predict(rigq):
        names = list(pr)[::-1]
        fig, ax = plt.subplots(figsize=(8, 0.4 * len(names) + 1.4))
        ax.scatter([pr[n] for n in names], range(len(names)), s=48, color=SERIES[0],
                   edgecolors=SURFACE, linewidths=1.5, zorder=3)
        for i, n in enumerate(names):
            ax.annotate(f"{pr[n]:.4f}%", (pr[n], i), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK2)
        ax.set_yticks(range(len(names)), names)
        ax.set_xscale("log")
        ax.set_xlim(right=max(pr.values()) * 8)
        ax.set_xlabel("accumulated error after 128 steps, % (log)")
        ax.set_title("[16] Predicted vs exact grids")
        made.append(("Grid prediction", save(fig, plots / "predict.png")))
    return made


# ------------------------------------------------------------------- summary
def summary(bench, train, secs):
    out = []
    env = secs.get("env", "").strip().splitlines()
    if env:
        out.append(f"GPU: {env[0].strip()}")
    for line in secs.get("cuda", "").splitlines():
        if "rel err" in line or "TFLOP/s" in line or "expansion" in line:
            out.append(f"CUDA kernel: {line.strip()}")
    ops = re.findall(r"^\s+(\d+)\s+([A-Z][\w.]+)$", secs.get("sass", ""), flags=re.M)
    if ops:
        out.append("SASS: " + ", ".join(f"{n} x {op}" for n, op in ops))
    if m := bench.get("mxfp4"):
        out.append(f"Triton MXFP4: {m['mismatches']} mismatches at K={m['exact_k']}, "
                   f"max rel err {m['rel_err']:.1e}")
        out += [f"  {r['name']}: {r['tflops']:.0f} TFLOP/s" for r in m["rows"]]
    if d := bench.get("dual"):
        out += [f"  dual {r['name']}: {r['tflops']:.0f} TFLOP/s" for r in d["rows"]]
    for r in train.get("rows", []):
        out.append(f"  {r['pass']} {r['name']}: {r['ms']:.3f} ms, {r['speedup']:.2f}x")
    for S, v in sorted(parse_splitk(secs.get("rig", "")).items()):
        if len(v) == 3:
            ratio = v["int32 x2"] / v["packed int16"]
            out.append(f"split-K S={S}: packed {ratio:.2f}x vs int32 pairs")
    fair = parse_fair(secs.get("rig_q", ""))
    if "int16, direct" in fair and "fp16, direct" in fair:
        out.append(f"[13] P=128: int16 direct {fair['int16, direct'][128]}%, "
                   f"fp16 direct {fair['fp16, direct'][128]}%")
    return out


def main():
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    load = lambda n: json.loads((run / n).read_text()) if (run / n).exists() else {}
    bench, train = load("bench.json"), load("train.json")
    log = (run / "run.log").read_text(errors="replace") if (run / "run.log").exists() else ""
    secs = sections(log)
    made = charts(run, bench, train, secs)
    lines_ = summary(bench, train, secs)
    failed = (run / "failed.txt").read_text().split() if (run / "failed.txt").exists() else []

    figs = "".join(
        f"<figure><img alt='{html.escape(t)}' src='data:image/png;base64,"
        f"{base64.b64encode(p.read_bytes()).decode()}'><figcaption>{html.escape(t)}</figcaption></figure>"
        for t, p in made)
    body = "\n".join(html.escape(x) for x in lines_)
    fail = f"<p class=bad>Failed steps: {html.escape(' '.join(failed))}</p>" if failed else ""
    tables = "".join(f"<details><summary>{html.escape(k)}</summary>"
                     f"<pre>{html.escape(v.strip())}</pre></details>" for k, v in secs.items())
    (run / "report.html").write_text(f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>H100 run {run.name}</title>
<style>
body{{margin:0 auto;max-width:960px;padding:24px 16px;background:#f9f9f7;color:{INK};
font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}h1{{font-size:20px;margin:0 0 12px}}
pre{{background:{SURFACE};padding:12px;overflow-x:auto;border:1px solid {GRID};border-radius:6px;
font-size:12px}}figure{{margin:16px 0}}
img{{max-width:100%;border:1px solid {GRID};border-radius:6px}}
figcaption{{color:{INK2};font-size:12px}}
.bad{{color:#d03b3b;font-weight:600}}summary{{cursor:pointer;color:{INK2};padding:4px 0}}
</style></head><body><h1>H100 run {html.escape(run.name)}</h1>{fail}<pre>{body}</pre>{figs}
<h2 style="font-size:15px">Raw output</h2>{tables}</body></html>""", encoding="utf-8")

    print("\n".join(lines_))
    if failed:
        print("failed steps:", " ".join(failed))
    print(f"charts: {len(made)} in {run / 'plots'}\nreport: {run / 'report.html'}")


if __name__ == "__main__":
    main()
