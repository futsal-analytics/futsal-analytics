from ultralytics import YOLO

model = YOLO("yolo11m.pt")   # 처음 실행 시 자동 다운로드

model.track(
    source=r"C:\futsal\test_1min.mp4",
    classes=[0],            # 사람만
    conf=0.3,
    imgsz=1280,
    tracker="bytetrack.yaml",
    device=0,               # GPU 사용
    save=True               # 박스가 그려진 결과 영상 저장
)