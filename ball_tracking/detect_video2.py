import csv, sys, cv2
from ultralytics import YOLO

# 사용법: python detect_video2.py 영상 출력csv 시작오프셋(초) stride 해상도 모델파일(best.pt)
video, out_csv = sys.argv[1], sys.argv[2]
offset_s, stride = float(sys.argv[3]), int(sys.argv[4])
IMGSZ = int(sys.argv[5]) if len(sys.argv) > 5 else 1280
WEIGHTS = sys.argv[6] if len(sys.argv) > 6 else "best.pt"   # 모델 파일 경로 (안 적으면 현재 폴더의 best.pt)

cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

model = YOLO(WEIGHTS)
n_frames = n_det = 0
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t", "x1", "y1", "x2", "y2", "conf"])
    for i, r in enumerate(model.predict(source=video, stream=True, imgsz=IMGSZ,
                                        device=0, conf=0.1, vid_stride=stride, verbose=False)):
        t = round(offset_s + i * stride / fps, 4)
        n_frames += 1
        if len(r.boxes) == 0:
            w.writerow([t, "", "", "", "", ""])      # 검출 없음도 기록 (시간축 유지)
            continue
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            w.writerow([t, round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1), round(float(b.conf[0]), 3)])
            n_det += 1
        if n_frames % 300 == 0:
            f.flush()
print(f"frames={n_frames}, boxes={n_det}, fps={fps:.3f}, stride={stride}")

