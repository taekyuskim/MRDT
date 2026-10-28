# 소스 출처와 현재 버전

2026-10-08 서버의 MRDT 연속 방향 학습 코드를 기준으로 만들었습니다. 학습에 사용한 14개 소스는 checkpoint 기록과 SHA256을 대조해 동일함을 확인했습니다. 해당 파일을 포함해 기존 소스의 계산 로직은 수정하지 않았습니다.

- 기반 저장소: `thu-ml/RoboticsDiffusionTransformer`, 원래 checkout HEAD `cd79363a1387e8f81c7724d070ef7e45fd23150f`.
- RDT 기반 코드 및 ReSCENE encoder/codec의 당시 작업 파일을 보존했습니다. Upstream HEAD와 완전히 같은 코드라는 의미는 아닙니다.
- `mrdt_train.py`, `mrdt_sample.py`, 패키지 초기화 파일, requirements와 문서를 공유용으로 추가했습니다.
- `mrdt_train.py`는 현재 continuous trainer를 호출하고 encoder 위치를 전달합니다. 학습 loss·배치·sampling·checkpoint 정책은 기존 trainer를 사용합니다.
- 서버 원본 학습 코드와 중단 checkpoint는 수정하지 않았습니다. 운영 예약/SSH helper/인증 정보/학습 데이터/가중치는 공유 대상에서 제외했습니다.
- `source_manifest.json`은 내려받은 원본 파일의 해시를 기록합니다.

현재 지속 학습 중인 모델은 이전 scratch 실험에서 출발해 12,795 step에 연속 방향 조건으로 전환한 이력이 있습니다. README의 `--scratch` 명령은 현행 연속 조건으로 새 실험을 시작합니다. 두 이력을 구분해 비교하세요.

## 검증 범위

공유 checkout의 CLI help 2개, 학습 entry point import, 작은 4D 모델의 CPU forward/backward·sampling, latent patch roundtrip, 연속 각도/padding, 실제 catalog의 context 5장 전처리가 통과했습니다. RDT/encoder와 준비 모듈을 공유 checkout에서 import하는 것도 확인했습니다. 이 공유 작업에서는 GPU 학습이나 전체 크기 모델의 생성 평가를 새로 실행하지 않았습니다. 새로운 환경에서의 전체 pip 설치는 별도 검증이 필요합니다.

CPU 모델 검사는 `python -m tests_plan_b.smoke_current`로 실행합니다. `tests_plan_b`의 기존 일부 검사는 과거 pilot 데이터 경로가 필요하므로 전체 테스트를 무조건 실행하지 않습니다.
