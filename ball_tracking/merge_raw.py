import csv, sys
from collections import defaultdict

def load(p):
    d = defaultdict(list)
    for r in csv.DictReader(open(p)):
        d[r["t"]]
        if r["conf"] != "":
            d[r["t"]].append(tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2", "conf")))
    return d

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0

a = load(sys.argv[1])     # 1280
b = load(sys.argv[2])     # 1920
ts = sorted(set(a) | set(b), key=float)

has = lambda d, t: any(x[4] >= 0.25 for x in d.get(t, []))
only_a = sum(1 for t in ts if has(a, t) and not has(b, t))
only_b = sum(1 for t in ts if has(b, t) and not has(a, t))
both = sum(1 for t in ts if has(a, t) and has(b, t))
print(f"후보(>=0.25)가 있는 프레임: 1280만 {only_a}, 1920만 {only_b}, 둘 다 {both}")

merged, nboxes = {}, 0
for t in ts:
    kept = []
    for bx in sorted(a.get(t, []) + b.get(t, []), key=lambda x: -x[4]):
        if all(iou(bx, k) < 0.3 for k in kept):
            kept.append(bx)
    merged[t] = kept
    nboxes += len(kept)

with open(sys.argv[3], "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t", "x1", "y1", "x2", "y2", "conf"])
    for t in ts:
        if merged[t]:
            for x in merged[t]:
                w.writerow([t] + [round(v, 3) for v in x])
        else:
            w.writerow([t, "", "", "", "", ""])
print(f"합친 후보 박스 {nboxes}개 (1280 {sum(len(v) for v in a.values())}개 + 1920 {sum(len(v) for v in b.values())}개)")
print(f"합친 후보가 있는 프레임(>=0.25): {sum(1 for t in ts if any(x[4] >= 0.25 for x in merged[t]))}")
