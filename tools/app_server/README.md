# tools/app_server: 분석 결과를 앱 서버로 보내기

파이프라인 결과(`tracks.json`)를 Playbook 앱 서버(앱 저장소의 `backend/`, FastAPI)에 위치 기록으로 올립니다.
앱의 "경기 정보 → 선수별 경기 정보" 화면이 이 기록으로 선수별 이동 거리를 보여 줍니다.
규격과 검증 기록은 [docs/app-connection.md](../../docs/app-connection.md)에 있습니다.

파이썬 표준 라이브러리만 씁니다(설치할 것 없음). 관리자 비밀번호는 실행할 때 직접 입력합니다(화면에 안 보이고, 저장하지 않음).
자동화할 때만 환경변수 `PLAYBOOK_ADMIN_PASSWORD`를 씁니다.

## 이미 있는 경기에 올리기: `tracks_to_positions.py`

```powershell
# 1) 미리 계산만 (서버에 안 씀): 앱에 뜰 거리와 파이프라인 거리를 나란히 보여 줌
python tools/app_server/tracks_to_positions.py --tracks runs/<결과>/tracks.json --mapping mapping.csv --server <서버 주소> --match <경기 id> --dry-run
# 2) 실제로 올리기
python tools/app_server/tracks_to_positions.py --tracks runs/<결과>/tracks.json --mapping mapping.csv --server <서버 주소> --match <경기 id>
```

`mapping.csv`는 `track_id,player_id` 두 열입니다(파이프라인 트랙 번호 → 앱 서버의 선수 id). 매핑하지 않은 트랙은 올리지 않습니다.
매핑은 **같은 실행에서 나온 `trackview.mp4`**를 보고 만듭니다(실행마다 트랙 번호가 새로 붙음).

## 새 경기와 선수를 만들고 올리기: `upload_to_server.py`

```powershell
python tools/app_server/upload_to_server.py --server <서버 주소> --tracks runs/<결과>/tracks.json --team <서버의 팀 이름> `
    --tracks-ids 1,2,3,4,5 --opponent "<새 경기 이름>" --label "<선수 이름 앞부분>"
```

무엇을 만들지 먼저 보여 주고 `y`를 입력해야만 씁니다. 같은 팀에 같은 이름의 경기가 있으면 멈춥니다.

## 주의

- 서버에는 위치 기록을 **지우는 API가 없습니다.** 같은 경기에 두 번 올리면 거리가 두 배가 됩니다. 두 스크립트 모두 이미 기록이 있으면 멈춥니다(`--force`로만 넘김).
- 운영 서버에 올리기 전에 로컬 서버(앱 저장소 `backend/`를 `DATABASE_URL=sqlite:///./test.db`로 실행)에서 먼저 해 보기를 권합니다.
- 서버 주소와 비밀번호는 이 저장소(공개)에 적지 않습니다.
