# Commodity Risk DGRI

**원유·천연가스·전력의 뉴스 기반 위험지수 및 위험 전이 연구**

GDELT 2.0 이벤트와 기사 본문에서 EventRisk를 구성하고 축적 위험 DGRI와 시장 간 VAR/GFEVD를 분석합니다. DGRI 약어의 영문 확장은 임의로 붙이지 않았습니다.

| 단계 | 파일 | 역할 |
|---|---|---|
| 1 | `src/01_collect_event_risk.py` | 이벤트·기사 수집, 시장별 위험·coverage/QC |
| 2 | `src/02_build_dgri.py` | RAW/SOFT-70/STRICT-70, Train phi·기준분포 |
| 3 | `src/03_analyze_risk_transmission.py` | 충격, VAR/GFEVD, TCI, bootstrap, 30일 일별 rolling |

FINAL/V3를 파일명에서 정리하고 단계명을 통일했습니다. 내부 방법론 버전과 계산 논리는 유지합니다. 원본 파일은 변경하지 않았고 원본명·해시는 업로드하지 않는 로컬 목록에 기록합니다.

## 연구 설계

- 전체 2024-10-01~2025-05-31.
- Train 2024-10-01~2025-03-31 / Test 2025-04-01~2025-05-31.
- 메인 SOFT-70, RAW/STRICT-70 민감도 비교.
- 결측 EventRisk에는 새 충격을 넣지 않고 기존 state만 감쇠.
- 30달력일 rolling VAR(1), 최소 유효 연속 관측 25개.
- source missing·partial source·genuine zero 구분.

## 실행

`python -m pip install -r requirements.txt` 후 세 소스 파일을 새 `runs/<id>/`에 함께 복사해 번호순 실행합니다. 결과는 실행 폴더 아래 저장됩니다. 1·3단계 원코드에는 누락 패키지 자동 설치가 있으므로 환경을 먼저 준비합니다.
2단계는 1단계 `dgri_input.csv`, 3단계는 2단계 결과를 요구합니다. 기사 수집은 시간·네트워크가 필요합니다. 이번에는 수집·추정·분석을 재실행하지 않았고 성과를 새로 주장하지 않습니다. 기사 전문·원응답·캐시는 GitHub에서 제외하고 원본 보관과 삭제 승인 정책을 따릅니다.

## 격리 실행 폴더 만들기

먼저 python prepare_run.py --run-id research_001 을 실행하면 소스 복사본과 빈 work/outputs 폴더만 생성됩니다. 연구·다운로드·삭제는 실행하지 않습니다. 이후 위 재현 순서에 맞춰 입력과 작업 디렉터리를 준비하세요.

