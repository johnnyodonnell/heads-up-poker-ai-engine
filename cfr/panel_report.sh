#!/usr/bin/env bash
# Summarize cfr/panel.sh (round 2): scored live - table per opponent and
# pooled, the paired cross-check, live turn - live river on the same deals,
# and the probe bots. bb/100 with 95% intervals.
#
#   cfr/box.sh ./panel_report.sh
set -uo pipefail
cd "$(dirname "$0")"
source ./box_env.sh
python3 - "$HU_HOME/evals/round2" <<'PY'
import glob, math, os, re, sys

S = sys.argv[1]

def load(p):
    return [float(x) for x in open(p)]

def series(tag, opp, ext):
    """A match's per-deal values, its chunks (--from) joined in deal order."""
    parts = []
    for p in glob.glob(f"{S}/{tag}--vs-{opp}--100bb-*-2.{ext}"):
        m = re.search(r"--100bb-(?:(\d+)\+)?(\d+)-2\.", os.path.basename(p))
        parts.append((int(m.group(1) or 0), int(m.group(2)), p))
    parts.sort()
    out, at = [], 0
    for start, n, p in parts:
        if start != at:
            break  # a gap: use the prefix
        out += load(p)
        at = start + n
    return out

def stat(x):
    n = len(x)
    if n < 2:
        return None
    m = sum(x) / n
    v = sum((y - m) ** 2 for y in x) / (n - 1)
    return m / 2 * 100, 1.96 * math.sqrt(v / n) / 2 * 100, n

def show(label, x):
    s = stat(x)
    print(f"  {label:44s} " + ("n/a" if s is None else f"{s[0]:+8.2f} ± {s[1]:5.2f}   ({s[2]} deals)"))

def paired(a, b):
    n = min(len(a), len(b))
    return [x - y for x, y in zip(a[:n], b[:n])]

opps = ["run_v1c", "run_v1a"]
for tag, name in [("river", "live river"), ("turn", "live turn")]:
    print(f"== {name} - table (scored, opponent's range)")
    pooled = []
    for o in opps:
        a = series(tag, o, "adv")
        pooled += a
        show(o, a)
    show("pooled", pooled)
    print(f"== {name} - table (paired matches, cross-check)")
    for o in opps:
        show(o, paired(series(tag, o, "txt"), series("table", o, "txt")))
print("== live turn - live river (scored, same deals)")
pooled = []
for o in opps:
    d = paired(series("turn", o, "adv"), series("river", o, "adv"))
    pooled += d
    show(o, d)
show("pooled", pooled)
print("== probe bots: table, and live minus table on the same deals")
for b in ["station", "maniac", "nit", "shover", "minbet", "overbet", "random"]:
    t = series("table", b, "txt")
    if not t:
        continue
    show(f"{b}: table", t)
    for tag in ["river", "turn"]:
        x = series(tag, b, "txt")
        if x:
            show(f"{b}: live {tag}", x)
            show(f"{b}: live {tag} - table", paired(x, t))
PY
