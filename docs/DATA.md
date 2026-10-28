# 데이터 구성

학습기는 준비된 전체 MVImgNet2.0 catalog를 받습니다. 데이터·mask·COLMAP 파일과 모델 가중치는 이 저장소에 포함되지 않습니다.

```text
processed_plan_b_full/bin15_masked_catalog_v2/
  catalog.sqlite
  readiness.json
  preparation_config.json
  mvimgnet2_category.json       # class 이름, 선택
```

`catalog.sqlite`의 instance row에는 원본 RGB/mask 경로, camera pose, intrinsics, 객체 중심, reference image와 각 view의 연속 방향이 들어 있습니다. 원본 RGB·mask·sparse COLMAP pose 파일은 해당 경로에서 읽을 수 있어야 합니다. PNG 파일만 모은 디렉터리는 입력 계약을 충족하지 않습니다.

현재 full trainer는 다음을 확인합니다.

- 41개 archive 전체 범위, instance 제한 없음.
- 압축 CRC 검증과 전체 mask audit 완료.
- 원본 archive의 크기/mtime, mask dataset 코드 해시와 readiness identity.
- 재개 시 checkpoint의 데이터 identity·14개 학습 소스 해시·설정 일치.

다른 서버로 이전하려면 원본 절대경로와 archive metadata를 보존하거나, 그 서버에서 catalog 및 검증 기록을 다시 만들어 새 학습을 시작해야 합니다. 다른 경로로 재구성한 catalog를 기존 checkpoint와 같다고 표시하지 않습니다.

관련 준비 도구는 `prepare_full_bin15.py`, `index_verified_full_bin15.py`, `audit_full_masks.py`, `build_masked_catalog.py`입니다. `python -m plan_b.mvimgnet.<도구명> --help`로 옵션을 확인할 수 있습니다. `index_verified_full_bin15`는 이미 압축 해제 및 CRC 검증을 마친 자료와 verification receipt를 요구합니다. 데이터 취득·전체 archive 해제 작업을 자동으로 대신하지 않습니다.

입력 전처리는 context와 GT 모두 같은 mask/roll 정렬을 사용하고, 물체 긴 변을 화면의 80%로 맞춘 뒤 회색 배경에 중앙 배치합니다. 학습에는 GT가 필요하지만 `mrdt_sample.py`의 생성 입력에는 target RGB가 필요하지 않습니다.
