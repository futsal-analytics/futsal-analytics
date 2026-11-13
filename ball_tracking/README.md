# ball_tracking: 공 검출·추적 파이프라인

두 카메라(cam1, cam2) 영상에서 공을 찾고, 프레임마다 공 위치 하나를 정해서 CSV로 저장합니다.
모든 결과는 프레임 번호가 아니라 **시간(초)** 기준입니다.

- 모델: YOLO26s를 우리 영상으로 학습한 공 전용 모델(`best.pt`). 모델 파일과 학습 코드는 GitHub에 없고 구글 드라이브로 따로 전달합니다.
- 필요한 것: Python 3.12, ultralytics, opencv-python, numpy, (GPU 권장)

## 순서

| 단계 | 파일 | 하는 일 |
|---|---|---|
| 1 | detect_video2.py | 영상에서 공 후보 박스 찾기. 1280, 1920 두 해상도로 각각 실행 (서로 다른 공을 잡음) |
| 2 | merge_raw.py | 두 해상도 결과 합치기 (겹치는 박스 IoU 0.3 이상은 하나로) |
| 3 | filter_static.py | 한자리에 오래 가만히 있는 공(예비공) 지우기. cam2에만 사용 |
| 4 | track_ball.py | 칼만 필터로 프레임마다 공 하나를 고르고, 잠깐 안 보이면 위치 예측 |
| 5 | to_court.py | 박스 아래 중앙점을 코트 좌표(m)로 변환 (`../calibration/` 사용) |
| - | render_dual.py | 두 카메라 추적 결과를 좌우로 붙인 확인용 영상 |

## 실행 예 (동기화된 원본 영상 기준)

```
# 1. 검출 (cam1은 60fps라 stride 2로 약 30fps, cam2는 30fps라 stride 1)
python detect_video2.py cam1.mp4 c1_raw_1280.csv 0 2 1280 best.pt
python detect_video2.py cam1.mp4 c1_raw_1920.csv 0 2 1920 best.pt
python detect_video2.py cam2.mp4 c2_raw_1280.csv 0 1 1280 best.pt
python detect_video2.py cam2.mp4 c2_raw_1920.csv 0 1 1920 best.pt

# 2. 합치기
python merge_raw.py c1_raw_1280.csv c1_raw_1920.csv c1_raw_merged.csv
python merge_raw.py c2_raw_1280.csv c2_raw_1920.csv c2_raw_merged.csv

# 3. cam2 예비공 지우기
python filter_static.py c2_raw_merged.csv c2_raw_merged_ns.csv

# 4. 추적
python track_ball.py c1_raw_merged.csv c1_track.csv
python track_ball.py c2_raw_merged_ns.csv c2_track.csv

# 5. 코트 좌표
python to_court.py 1 c1_track.csv ball_cam1.csv
python to_court.py 2 c2_track.csv ball_cam2.csv

# (확인용 영상) 오프셋 0, 21초~3158초 구간
python render_dual.py cam1.mp4 c1_track.csv cam2.mp4 c2_track.csv dual.mp4 0 21 3158
```

52분 영상 기준 소요 시간(RTX 5060): 검출 4번 약 2시간, 확인용 영상 약 36분.

## 출력 형식

- 추적 CSV (`track_ball.py`): `t, status, x1, y1, x2, y2, conf, track_id`
  - status: `detected`(모델이 찾음, 영상의 초록 박스) / `predicted`(잠깐 안 보여서 예측, 주황) / `missing`(없음)
- 코트 좌표 CSV (`to_court.py`): `t, status, track_id, u, v, conf, x_m, y_m`
  - u, v: 박스 아래 중앙 픽셀. x_m, y_m: 코트 좌표(m, 센터 원점)

## 주의

- 공중에 뜬 공은 바닥 기준으로 변환하기 때문에 코트 좌표가 틀어집니다. 코트 밖 좌표나 두 카메라 좌표 불일치를 이유로 공을 지우면 안 됩니다.
- 박스 중심이 아니라 **박스 아래 중앙**을 변환해야 두 카메라 좌표가 맞습니다 (바닥 공 717쌍 기준 두 카메라 간 거리 중앙값 2.20m → 0.32m).
- 각 카메라의 먼 코너 뒤쪽은 사각지대이고, 반대편 카메라가 거의 메우지 못합니다.
- 47:00~48:00 구간을 직접 눈으로 라벨링해 평가한 결과: 오류 없는 프레임 cam1 99.7%, cam2 91.0%. 1분 구간 결과라 전체 성능을 대표하지는 않습니다.
- 모델 학습에 0~35분 영상을 썼으므로, 이 구간 결과는 실제보다 좋아 보일 수 있습니다.
