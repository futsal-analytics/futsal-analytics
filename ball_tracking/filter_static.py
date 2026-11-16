import csv, sys
from collections import defaultdict

inp, outp = sys.argv[1], sys.argv[2]
WINDOW, RADIUS, FRAC, MINCONF = 60.0, 15.0, 0.5, 0.25

rows = list(csv.DictReader(open(inp)))
times = sorted({float(r["t"]) for r in rows})
t0 = times[0]
win_of = lambda t: int((t - t0) // WINDOW)
frames_in = defaultdict(set)
for t in times:
    frames_in[win_of(t)].add(t)

def center(r):
    return (float(r["x1"]) + float(r["x2"])) / 2, (float(r["y1"]) + float(r["y2"])) / 2

by_win = defaultdict(list)
for r in rows:
    if r["conf"] != "" and float(r["conf"]) >= MINCONF:
        by_win[win_of(float(r["t"]))].append(r)

static = {}
for w, rs in by_win.items():
    clusters = []   # [cu, cv, set(시각)]
    for r in sorted(rs, key=lambda r: -float(r["conf"])):
        u, v = center(r)
        for c in clusters:
            if abs(u - c[0]) <= RADIUS and abs(v - c[1]) <= RADIUS:
                c[2].add(float(r["t"]))
                break
        else:
            clusters.append([u, v, {float(r["t"])}])
    static[w] = [(c[0], c[1], len(c[2]) / len(frames_in[w])) for c in clusters if len(c[2]) / len(frames_in[w]) >= FRAC]
    for cu, cv, fr in static[w]:
        print(f"창 {w} 고정 후보: 화면({cu:.0f},{cv:.0f})  전체 프레임의 {fr*100:.0f}%")
if not any(static.values()):
    print("고정 후보 없음")

kept = defaultdict(list)
dropped = 0
for r in rows:
    t = float(r["t"])
    if r["conf"] == "":
        kept[t]
        continue
    u, v = center(r)
    if any(abs(u - cu) <= RADIUS and abs(v - cv) <= RADIUS for cu, cv, _ in static.get(win_of(t), [])):
        dropped += 1
        kept[t]
    else:
        kept[t].append(r)

with open(outp, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t", "x1", "y1", "x2", "y2", "conf"])
    for t in sorted(kept):
        if kept[t]:
            for r in kept[t]:
                w.writerow([r["t"], r["x1"], r["y1"], r["x2"], r["y2"], r["conf"]])
        else:
            w.writerow([f"{t}", "", "", "", "", ""])
before = sum(1 for r in rows if r["conf"] != "")
fr_before = len({float(r["t"]) for r in rows if r["conf"] != ""})
fr_after = len({t for t in kept if kept[t]})
print(f"후보 {before}개 중 {dropped}개 제거 / 후보가 있는 프레임 {fr_before} → {fr_after}")
