# futsal-analytics

일반 휴대폰 2대로 풋살 경기를 찍어서 **선수 위치를 2D 지도에 표시하고 이동거리·속도·히트맵을 계산**하는 시스템입니다. (AX 캡스톤 디자인)

**서버 없이 휴대폰에서 모두 처리합니다.** 각 폰이 경기 후 자기 영상에서 선수를 검출하고, 결과(수 MB)만 한 폰으로 모아 병합·추적·기록 계산까지 합니다. 영상은 폰 밖으로 나가지 않아 서버 비용이 들지 않습니다. → [시스템 구조](docs/architecture.md)

![2D 지도 데모](docs/img/demo.png)

## 무엇이 들어 있나

| 경로 | 내용 | 상태 |
|---|---|---|
| `src/futsal/` | **분석 기준 구현**: FIFA 규격 코트, 카메라 모델, 기준점 보정, 두 카메라 병합, 지표, 시간 맞추기, 결과 포맷. 앱(Dart)으로 옮길 원본이자 정답 비교용 | ✅ 테스트 있음 |
| `src/futsal/sim/` | 배치·화각 시뮬레이터, 합성 경기·합성 영상 | ✅ |
| `src/futsal/trackview.py`, `calibtool.py` | 실제 영상 두 개 → 검출 → 좌표 변환 → 추적 → 결과 + 확인 영상 ([한 번에 돌리기](#한-번에-돌리기)). `calibtool.py`는 우리 기준점 탭 (예전 방식) | ✅ 10/8 실제 영상 1분으로 확인 |
| `src/futsal/camshift.py` | 촬영 중 카메라가 움직인 순간 자동 감지 → 시간 구간별 좌표 보정 | ✅ |
| `src/futsal/check.py` | 촬영 영상 점검: `python -m futsal.check cam1.mp4 cam2.mp4` (프레임 간격, 밝기, 미리보기, 시간 차이) | ✅ |
| `src/futsal/syncview.py` | 시간 맞추기 눈으로 확인: `python -m futsal.syncview cam1.mp4 cam2.mp4 --offset 124.29` (같은 순간 프레임 + 소리 겹쳐 그림) | ✅ |
| `src/futsal/ball/` | 공 검출: 라벨링용 프레임 뽑기, 조각 학습 데이터, 공개 데이터 합치기, 학습, 조각 검출 + 궤적 정리 | ✅ 도구 완료, 학습 진행 중 |
| `src/futsal/pipeline/` | PC에서 실제 영상 → 검출 → 병합 → 추적 (앱 결과와 비교할 정답 생성) | ✅ 합성 영상으로 검증, 실제 영상 검증 필요 |
| `detection/` | **탐지 (차재혁)**: YOLO 사람 검출 + ByteTrack 스크립트. 파이프라인이 이 파일의 모델·설정을 읽어 씀 (YOLO26x) | ✅ 파이프라인에 연결 |
| `calibration/` | **좌표 변환 (HOGEUN)**: 카메라마다 두 호모그래피(가까운 쪽·먼 쪽), 흰 선 보정, `court_map.py` | ✅ 파이프라인에 연결 |
| `server/` | **개발용 PC 도구** (운영 서버 아님): 영상 올려서 분석·확인 | ✅ 테스트 있음 |
| `web/viewer/` | 2D 지도 뷰어 (재생, 꼬리, 선수 기록, 히트맵). 앱 웹뷰·공유용 | ✅ |
| `app/` | Flutter 앱 (촬영 → 폰 안에서 검출 → 결과) | ⏳ 다음 단계 |
| `docs/` | 구조·촬영 가이드·시뮬레이션·로드맵 | |

## 한 번에 돌리기

세 폴더가 하나의 파이프라인으로 이어져 있습니다.

| 폴더 | 맡은 일 | 파이프라인에서 쓰는 방법 |
|---|---|---|
| `detection/` (차재혁) | 탐지 | `detect_person.py`를 진짜 YOLO 대신 가짜 `ultralytics`로 실행해서 모델·설정만 읽고(YOLO26x, conf 0.3, imgsz 1280, ByteTrack), 그 설정을 우리 영상에 그대로 돌립니다. 그 파일을 고치면 파이프라인도 바뀝니다. 파일 안의 YOLO 말고 다른 코드(파일 열기, print 등)는 그대로 실행됩니다 (`src/futsal/pipeline/yolo_track.py`) |
| `calibration/` (HOGEUN) | 좌표 변환 | `court_map.py`의 `load`, `pixel_to_court`를 그대로 불러 씁니다 (카메라마다 가까운 쪽·먼 쪽 호모그래피 두 개). 우리 좌표(x 0~40, y 0~20 m)로는 x + 20, y + 10만 옮깁니다 (`src/futsal/nearfar.py`) |
| `src/futsal/` | 추적 | ReID, 시간 맞추기, 추적 v2, 이동거리·속도 |

두 폴더는 모델 이름 한 줄(`yolo11m.pt` → `yolo26x.pt`) 말고 바꾸지 않았습니다. 돌릴 때도 두 폴더에 아무것도 쓰지 않습니다.

**영상 두 개 → 추적 결과** (저장소 맨 위 폴더에서):

```powershell
python -m futsal.trackview --video C:\Users\user\dev\futsal-videos\20261008_170919.mp4 C:\Users\user\dev\futsal-videos\20261008_171122.mp4 `
    --offset 0 124.27 --start 600 --duration 180 --out trackview
```

- 순서: 검출(모든 프레임) → ReID(초당 10장) → 좌표 변환 → 시간 맞추기(`--offset`, 10분 이상이면 선수 위치로 한 번 더) → 추적 v2
- 결과: `tracks.json`(선수별 위치), `summary.json`(선수별 이동거리·속도, 쓴 검출 설정과 보정 파일), `events.csv`, `trackview.mp4`(확인 영상, `--no-video`면 안 만듦). 확인 영상에는 사람이 찍혀 있으니 저장소에 넣지 않습니다.
- 10/8 영상은 기록된 카메라 움직임(cam1 2:40, cam2 46:13)을 자동으로 보정합니다. 10/8 영상인지는 파일 이름(`20261008_170919.mp4`, `20261008_171122.mp4`)이나 프레임 수로 알아봅니다. 보정 값은 HOGEUN 보정을 만든 프레임(파일의 `t_sec`, 지금 60초) 기준으로 계산합니다. 다른 영상은 카메라가 안 움직였다고 보고 계산하고, 그렇다고 화면에 알려 줍니다. `--auto-moves`(자동 감지)는 10/8 영상에서 틀린 움직임을 찾으니 쓰지 않습니다.
- 카메라가 켜진 직후(cam1 약 20초, cam2 약 30초)와 부딪힌 직후(cam1 2:39~2:52, cam2 46:13~46:16)는 흔들리는 중이라 이 보정이 맞지 않습니다. 먼 쪽 위치가 몇 m 틀릴 수 있습니다. 53분 결과에는 cam2 처음 30초와 cam1 2:39~2:52가 들어 있습니다.
- 시작 전에 확인합니다: 10/8 영상을 cam2, cam1 순서로 넣었거나 `--offset`·`--drift` 개수가 영상 수와 다르면 바로 멈춥니다. 다른 경기 영상이면 HOGEUN 보정이 그 영상으로 만든 것이 아니라고 알려 줍니다. 그때는 그 경기 영상으로 `calibration/`에서 다시 찍어야 합니다.
- `detection/`, `calibration/` 폴더를 저장소에서 바로 읽기 때문에 `pip install -e`(저장소 안에서 설치)로 설치해야 합니다.
- 처리 시간: 1분 구간에 약 9분 (RTX 2070 SUPER, 다른 작업과 같이 돌린 값. 검출 cam1 5.5분 + cam2 3분, 확인 영상 1분). ByteTrack이 모든 프레임을 봐야 해서, 초당 10장만 검출하던 예전보다 cam1(60fps)은 6배, cam2(30fps)는 3배 검출이 많습니다.

**53분 전체 (저장된 검출 결과로):**

```powershell
python experiments/full_match.py --out experiments/full_match_team
```

- 좌표 변환은 `calibration/`입니다. 검출은 `detection/`이 아니라 이미 만들어 둔 `experiments/cache_full26`(우리 `experiments/build_cache.py`: YOLO26x, conf 0.1, 초당 10장)입니다. 53분 전체를 `detection/`로 다시 검출하지는 않았습니다.
- 추적만 약 9~10분 걸립니다.

**우리가 검증한 결과와 다른 점** (10/8 영상 10:00부터 1분. 숫자는 앞이 `detection/`, 뒤가 우리 검출. 박스는 같은 프레임에 우리 검출(YOLO26x, conf 0.1)을 다시 돌려 비교):

- **conf 0.3** (우리는 0.1): 박스가 프레임당 cam1 10.7개 대 12.2개, cam2 12.6개 대 14.6개로 약 13% 적습니다. 빠진 것은 대부분 신뢰도 0.1~0.3 박스이고, 대부분 코트 밖입니다. 코트 안(발이 코트에서 1 m 이내) 박스만 세면 cam1 7.9개 대 8.2개, cam2 8.5개 대 8.8개로 약 4% 적습니다 (코트 안 숫자는 저장된 `cache_full26` 박스로 셈). 신뢰도 0.3 이상인 우리 박스 중 ByteTrack이 내지 않은 것(아직 확정 안 된 트랙)은 1.0%(cam1), 1.5%(cam2)입니다.
- **ByteTrack 박스**: 칼만 필터로 다듬은 위치지만 차이는 작습니다. detection/ 박스는 99.9% 이상 우리 박스와 겹치고(IoU 0.5 이상), 발 위치 차이는 중앙값 0.2 px(90%가 1.1 px 이내), 코트 위로 중앙값 0.01 m(90%가 0.08 m 이내)입니다.
- **추적 결과** (같은 1분, 같은 `calibration/` 좌표. 우리 쪽은 다시 돌린 박스가 아니라 저장된 `cache_full26` 박스라서 cam1 프레임이 한 장(16.7 ms) 어긋납니다): `detection/` 박스는 ID 11개, 절반 넘게 보인 ID 10개, 한가운데서 생기거나 끊긴 ID 1건, 두 카메라 위치 차(정렬 후) 0.148 m, 절반 넘게 보인 10명의 이동거리 합 751 m. 우리 박스는 ID 12개, 10개, 4건, 0.148 m, 780 m. 정답 위치가 없어서 어느 쪽이 맞는지는 모릅니다. 1분 한 구간만 봤습니다.
- 같은 검출 결과로 다시 돌리면 추적 결과도 같습니다. 처음 돌릴 때도 저장된 검출 결과(ReID 특징은 float16)를 다시 읽어서 추적하기 때문입니다. 예전에는 처음 돌릴 때만 float32 특징을 써서, 20:00부터 3분 구간에서 처음 결과(절반 넘게 보인 ID 10개, 사건 3건)와 다시 돌린 결과(9개, 4건)가 달랐습니다. 반올림 정도의 차이로도 결과가 바뀌니, 짧은 구간 하나의 비교는 조심해서 봐야 합니다.
- 추적 v2의 설정과 [추적 v2](docs/tracking-v2.md)의 숫자는 우리 검출(conf 0.1)과 우리 탭 보정으로 맞추고 잰 값입니다.

**보정 비교** (53분, 같은 검출 `cache_full26`, 같은 추적):

| | 우리 탭 보정 (`calib/`) | 우리 탭 보정 + 새로 잰 카메라 움직임 | `calibration/` (HOGEUN) |
|---|---|---|---|
| ID 수 | 32 | 29 | 26 |
| 경기 절반 넘게 보인 ID / 80% 넘게 보인 ID | 6 / 5 | 7 / 6 | 7 / 4 |
| 한가운데서 생기거나 끊긴 ID | 33번 (분당 0.62) | 27번 (분당 0.51) | 21번 (분당 0.40) |
| 두 카메라 위치 차 (정렬 전 → 후) | 0.647 → 0.207 m | 0.664 → 0.209 m | 0.684 → 0.210 m |

- 라벨 없는 지표 비교입니다. 정답 위치가 없어서 어느 보정이 더 정확한지는 아직 모릅니다. 첫 열은 기존 `experiments/full_match26` 결과와 숫자가 같습니다.
- ID 수와 사건 수는 따로 보는 지표가 아닙니다. 6분 창이 이어지는 10곳에서 새 ID가 첫 열 20개, HOGEUN 쪽 14개 생기고(12 + 20 = 32, 12 + 14 = 26), 사건 33번 중 27번, 21번 중 17번이 경기 중간에 새로 생긴 ID의 것입니다.
- `summary.json`의 최고 속도는 아직 믿기 어렵습니다. 두 보정 모두 10 m/s가 넘는 값은 전부 6분 창 이음매 하나(17:35, 1055초 근처)에서 나옵니다 (첫 열 20.4 m/s, HOGEUN 쪽 14.3 m/s). 추적이 박스 없이 3초까지 메워 넣은 구간에서도 높은 속도가 나올 수 있습니다 (20:00부터 3분 구간의 8.7 m/s). 추적 v2 쪽 문제라 이번에 고치지 않았습니다.
- `calibration/`은 60초 화면으로 만든 보정이라, cam1은 2:40 이후 (-5.8, -9.2) px, cam2는 46:13 이후 (+12.2, -1.3) px 옮겨 씁니다 (측정 방법은 `src/futsal/nearfar.py`).
- 첫 열의 카메라 움직임 값(`experiments/full_match.py`의 `KNOCKS`, cam1 2:40 전 (41, 19) px, cam2 46:13 뒤 (13.8, 1.3) px)은 예전 값입니다. 부딪힌 지 1.5초 뒤에 잰 값인데, cam1은 그때 아직 흔들리는 중이었습니다. 이번에 흔들림이 멎은 뒤 잰 값(cam1 (5.8, 9.2) px, cam2 (12.2, -1.3) px, `nearfar.POSES_20261008`)과 cam1은 35 px 넘게, cam2는 3 px 정도 다릅니다. 가운데 열은 같은 탭 보정에 이번 값을 쓴 결과입니다 (임시 복사본으로 돌림, 저장소 코드는 예전 값 그대로). 움직임 값만 바꿔도 ID가 32개에서 29개로 줄어서, 보정끼리의 차이 중 일부는 이 값 때문입니다.

**예전 방식으로 되돌리기:** 우리 검출(`pipeline/detect.py`)과 우리 탭 보정(`calib/<cam>/calib.json`)을 부르던 줄은 `src/futsal/trackview.py`, `experiments/full_match.py`에 주석으로 남겨 두었습니다. 주석에 적힌 대로 풀면 됩니다. 어느 쪽을 기본으로 할지는 팀이 정합니다.

- `calib/`는 저장소에 없습니다(`.gitignore`). 우리 탭 보정으로 되돌리려면 각자 PC에서 [tracking-check.md](docs/tracking-check.md) 1번(기준점 탭)으로 먼저 만들어야 합니다.
- 바꾼 곳은 이 두 파일뿐입니다. 연구용 비교 도구(`experiments/harness.py`, `experiments/src_pipeline.py` 같은 `experiments/` 스크립트, `boxes.cams_from_cache`)는 지금도 우리 검출 캐시와 우리 탭 보정(`calib/`)을 씁니다. `server/`(개발용 PC 도구)도 우리 검출기(`pipeline/detect.py`)와 앱에서 탭한 보정을 씁니다.

## 빠른 시작 (PC)

```bash
pip install -e ".[server,sim,dev]"        # 실제 영상 검출까지 하려면 ".[vision]" 추가 (scipy는 기본 의존성)
# 추적 v2 (ReID): ".[vision,reid]" 설치 후 pip install --no-deps boxmot  (docs/tracking-check.md)
pytest                                     # 전체 테스트

uvicorn server.app.main:app --reload       # 개발용 도구: http://localhost:8000 → "데모 경기 만들기"
```

서버 없이 뷰어만 보려면: `python -m http.server -d web/viewer 8080` 후 `http://localhost:8080/?src=sample_tracks.json`

시뮬레이터:
```bash
python -m futsal.sim figures --court 40x20 --twist 5 --out docs/img   # 규격 도면, 커버리지, 데모
python -m futsal.sim fov --out docs/img                                # 화각 그림
python -m futsal.sim twist --court 42x25                               # 그 구장에서 안전한 회전각
python -m futsal.sim accuracy --hfov 67.3                              # 합성 경기 위치 오차
```

## 문서
- [시스템 구조 (폰에서 모두 처리)](docs/architecture.md)
- [촬영 가이드 (설치 위치·각도·설정)](docs/capture-guide.md)
- [시뮬레이션 방법과 결과](docs/simulation.md)
- [로드맵과 다음 단계](docs/roadmap.md)
- [팀원 폰 YOLO 검출 속도 측정 방법](docs/phone-benchmark.md)
- [선수 ID 추적: 기존 방법 조사와 우리 선택](docs/tracking-survey.md)
- [추적 v2: 방법, 결과, 알게 된 것 (10/8 실제 영상)](docs/tracking-v2.md)
- [추적 결과 영상 만들기 (실제 영상, 한 번에 돌리기)](docs/tracking-check.md)
- [공 검출: 학습 계획과 도구 (라벨링·Colab 학습·조각 검출)](docs/ball-detection.md)
- [2026-10-08 테스트 촬영 기록 (영상 점검·시간 맞추기 결과)](docs/capture-2026-10-08.md)
