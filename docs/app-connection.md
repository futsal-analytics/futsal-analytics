# 앱 ↔ 분석 연결 (약속과 검증 기록)

앱(앱 저장소: `android/` 녹화 앱 + `backend/` 서버)과 이 저장소(보정·공·파이프라인)는 **저장소를 따로 두고, 주고받는 형식만 약속**합니다.
약속을 바꿀 때는 두 저장소 담당이 함께 정하고 이 문서를 같이 고칩니다.

```
[앱] 녹화 ──① 영상 파일──▶ [이 저장소] calibration/auto_match.py → python -m futsal.trackview → tracks.json
                                                                                     │
[앱 결과 화면] ◀── GET /summary ── [앱 서버] ◀──② 위치 기록 (tools/app_server)────────┘
```

## ① 녹화 영상 (앱 → 분석)

| 항목 | 약속 | 이유 |
|---|---|---|
| 해상도·fps | 1920×1080, 30 fps 고정 | 사람·공 모델과 코트 보정이 모두 1080p 기준. 720p로 줄인 영상은 결과가 나빴음 |
| 코덱 | H.264 (High), 8비트 | 보급형 폰(Buddy4)이 HEVC 10비트를 못 읽음 |
| 비트레이트 | 앱 설정 13 Mbps (삼성 인코더 실측 약 16 Mbps) | 팀 분석 영상이 12.5~20.5 Mbps. 높이면 4 GB 한도에 빨리 닿음 |
| 소리 | AAC 48 kHz 모노 128 kbps | 두 카메라 시간 맞추기(소리)에 씀. 설정이 없으면 안드로이드 기본값(저음질) |
| 방향 | 폰을 **가로로** 세움 (회전 0°) | 세로로 찍으면 영상에 90° 회전 표시가 붙음 |
| 파일 길이 | 하프마다 따로, 한 파일 35분 이하 | 안드로이드 녹화기는 파일이 약 4 GB가 되면 멈춤 (16 Mbps면 약 35분) |
| 파일 이름 | `Playbook_<경기번호>_<A\|B>_<숫자>.mp4` (A = cam1, B = cam2) | 파이프라인은 `calibration/out/<경기>/sync.json`에 적힌 이름으로 영상을 찾음. auto_match가 이 이름을 그대로 적음 |
| 옮기는 법 | 폰 갤러리(`Movies/Playbook`)에서 USB나 외장 SSD로 복사 | 메신저로 보내면 다시 압축되어 이름·크기가 바뀌고 캐시를 못 알아봄 |

앱 쪽 녹화 설정 변경(1280×720 → 1920×1080, 비트레이트, 소리)은 앱 저장소의 별도 PR로 넣습니다.

## 새 경기 분석 (이 저장소)

```powershell
cd calibration
python auto_match.py ..\videos\Playbook_<경기>_A_<..>.mp4 ..\videos\Playbook_<경기>_B_<..>.mp4 --match <경기> --yes
cd ..
python -m futsal.trackview --video videos\Playbook_<경기>_A_<..>.mp4 videos\Playbook_<경기>_B_<..>.mp4 --start <초> --duration <초> --out runs\<경기>
```

- auto_match에는 `calibration/train/seg_best.ts`(저장소에 없음)와 torch가 필요합니다. torch가 없으면 경고 한 줄만 남기고 약한 방법으로 넘어갑니다.
- 1분 시험 출력에 `calibration/out/<경기>/sync.json 에서 읽음`과 `시간별 보정 calibration/out/<경기>/calib_cam1.json`이 나와야 합니다. `시간 차를 모르는 영상이라 0으로 둡니다`나 `calibration/out/1008`이 나오면 멈추고 보정 폴더를 확인합니다.
- `calibration/out/<경기>/`에는 auto_match 확인용 오버레이 사진(사람이 보임)이 생깁니다. 저장소에 올리기 전에 빼 주세요.

## ② 위치 기록 (분석 → 앱 서버)

- 입력: `tracks.json` (형식은 `src/futsal/tracks.py`): `fps`, `start`, `players[{id, team, x[], y[], stats}]`, 미터 좌표, 안 보인 프레임은 `null`, 코트 40×20 (원점 `corner_bl`, y 위쪽).
- 서버 API (앱 저장소 `backend/app/api/routes.py`):
  - `POST /api/v1/auth/login` `{username, password}` → `access_token` (Bearer)
  - `POST /api/v1/matches/{id}/positions` `{"positions": [{camera_id, player_id, timestamp_ms, x_m, y_m, confidence}]}`
    - `camera_id`는 대문자 한 글자, `timestamp_ms` ≥ 0, `x_m`·`y_m` ≥ 0이고 그 경기 코트 길이·폭 이내, `player_id`는 그 경기 팀의 선수
    - 한 점이라도 어기면 묶음 전체가 422로 거절됨. 한 번에 1~20,000개, 본문 8 MB 이하
    - **지우는 API 없음**
  - `GET /api/v1/matches/{id}/summary`: 선수별 이동 거리 = 같은 선수의 점을 시간순으로 이은 직선 거리의 합(다듬지 않음)
- 변환 규칙(`tools/app_server/tracks_to_positions.py`): 초당 5개로 줄여 보냄, `null`과 코트 밖 점은 뺌, 매핑 안 된 트랙은 뺌.
- 매핑표 `track_id,player_id`: 같은 실행의 `trackview.mp4`로 만듭니다.
- 앱에 뜨는 거리는 파이프라인 `stats.distance_m`(칼만+RTS로 다듬은 값)보다 2~3 % 깁니다(서버가 다듬지 않은 점을 더하기 때문). `--dry-run`이 두 값을 나란히 보여 줍니다.

## 검증 기록 (2026-11-26, 11/3 경기)

| 단계 | 결과 |
|---|---|
| 앱 녹화 (S26, 1080p 설정을 넣은 개발용 빌드, 30초) | 1920×1080, 30.00 fps, H.264 High 4.0, AAC 48 kHz 모노. 목표 16 Mbps → 실측 20.1 Mbps, 목표 13 Mbps → 실측 16.2 Mbps |
| 앱 파일 이름으로 새 경기 자동 보정 (11/3 _sync 영상을 앱 이름으로 바꿔서) | 클릭 0번, 두 카메라 5/5 시점 일치, 흰 선 일치율 0.95 / 0.74, 약 12분(CPU). 소리 시간 차 −0.147 s (손 보정 기록 약 −0.011 s, 한쪽 폰 근처 소리로 약 0.14 s 밀림) |
| 파이프라인 (위 보정, 10:00부터 1분) | ID 12개(절반 넘게 보인 10개), 두 카메라 위치 차 0.943 → 0.216 m. 같은 구간 손 보정은 12개, 0.939 → 0.168 m |
| tracks.json → 서버 (10분 결과, A팀 트랙 5개) | 13,360개 저장, 서버 거리 = 미리 계산 거리 (예: 782.11 m, 파이프라인 767.7 m) |
| 서버 → 앱 결과 화면 | "분석 기록이 도착했어요", 선수별 이동 거리 0.78 / 0.24 / 0.79 km … 표시 |

## 남은 일

1. 앱 저장소: 녹화 설정 변경 PR(위 ① 표)
2. 영상 옮기기와 "보정 → 분석 → 업로드"를 명령 하나로 묶기
3. 트랙 ↔ 실제 선수 매핑을 쉽게 만드는 도구(확인 영상에서 번호 고르기)
4. `server/`(개발용 PC 도구)는 옛 경로입니다. 앱 서버와 헷갈리지 않게 정리
