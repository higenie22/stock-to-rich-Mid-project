# 실행 환경

검증에 사용한 Conda 환경 이름은 `colab`입니다. Anaconda Prompt에서 다음과 같이 실행합니다.

```bat
conda activate colab
cd /d 'working/directory'
python run_submission.py
```

활성 환경은 `where python`과 `python --version`으로 확인할 수 있습니다. 동일한 패키지 버전을 설치하려면 프로젝트 루트에서 다음 명령을 실행합니다.

```bat
python -m pip install -r requirements.txt
```

`requirements.txt`에는 분석·모델링 패키지뿐 아니라 데이터 수집과 원본 노트북 실행에 필요한 패키지도 포함합니다. 특히 LSTM 실행에는 PyTorch와 Windows DLL 의존성이 정상적으로 설치되어 있어야 합니다.



## 실행 옵션

### 전체 제출 실행

안정성 가설, 수익성 실험, 최종 후보 비교, CV 고정 추천과 보고서를 모두 생성합니다.

```bat
python run_submission.py
```

### 최종 후보·추천만 실행

공통 데이터 로드 후 최종 구성 모델의 CV IC와 ReLU 비중을 계산하고, 후보 OOS 비교와 최신 추천을 저장합니다.

```bat
python run_submission.py --compare-candidates
```

이 옵션은 K·Lookback 등 안정성 전체 가설검증과 기간별·거시·LSTM 개별 실험을 생략합니다. 최종 후보군에는 같은 `StabilityPolicy`가 적용됩니다.

### 안정성 모듈만 점검

```bat
python run_submission.py --skip-profitability
```

이 옵션은 진단용이며 최종 제출 전체 실행에는 사용하지 않습니다.



## 모듈 구성

### 실행·설정·데이터

| 모듈                   | 역할                                                         |
| ---------------------- | ------------------------------------------------------------ |
| `run_submission.py`    | 인자를 해석하고 데이터 → 안정성 → 수익성 → 저장 순서로 실행하는 진입점 |
| `src/config.py`        | 프로젝트 루트, 입출력 경로, Lookback=120·K=3 등 공통 설정과 필수 파일 검사 |
| `src/data_pipeline.py` | `test_validation_report` 기준 공통 분석 패널 구성, 피처 계산, membership 적용, 섹터·거시자료 로드, 입력 해시 수집 |
| `src/research_data.py` | 연구 피처, 구성종목 이력, 분석 범위·coverage 처리            |

### 안정성·수익성 연구

| 모듈                       | 역할                                                         |
| -------------------------- | ------------------------------------------------------------ |
| `src/stability.py`         | 날짜별 Stable/Middle/Risky 생성과 K·Lookback·미래 위험·업종 효과 검증 |
| `src/profitability.py`     | 20/60/120일, 타깃·피처·학습방식, 거시변수, LSTM 등 수익성 실험 조율 |
| `src/research_protocol.py` | 미래 라벨 생성, 신호·진입·청산 계약, 실제 가격 경로 정산, 누적수익·MDD 집계 |
| `src/quant_validation.py`  | 라벨 이용 가능 시점, IC·분위수 성과, 의존성, 통제 후 IC 및 CV 도구 |
| `src/training_scope.py`    | 전체 종목과 Stable 학습 비교, Top-N 선택과 섹터 제한 보조    |

### 최종 앙상블·추천

| 모듈                          | 역할                                                         |
| ----------------------------- | ------------------------------------------------------------ |
| `src/candidate_comparison.py` | 공통 표본 후보 비교, 2020~2022 CV 실행, CV 고정 앙상블 OOS 평가와 최신 추천 생성 |
| `src/recommendation.py`       | 평균 CV IC의 ReLU 비중 계산과 추천 종목별 수치 설명 생성     |

현재 최종 추천 실행 경로는 다음과 같습니다.

```text
profitability.run_final_ensemble()
 → candidate_comparison.run_comparison()
   → 2020·2021·2022 모델별 CV IC 계산
   → recommendation.relu_weights()
   → cv_locked_relu_4model 비중 고정
   → OOS 4개 Fold 평가
   → 최신 시점 모델 재학습
   → 모델별 백분위 순위 가중합
   → 섹터 상한을 적용한 Stable Top-10 선택
   → recommendation.explain_recommendations()
```

`profitability.py`의 `run_original_final_ensemble()`과 `_rank_blend()`는 비교와 추적을 위해 보존된 이전 앙상블 구현입니다. 기본 최종 추천은 `run_final_ensemble()`이 위임하는 `candidate_comparison.py`에서 생성합니다.

### 보고서·시각화

| 모듈                          | 역할                                                         |
| ----------------------------- | ------------------------------------------------------------ |
| `src/reporting.py`            | 모든 DataFrame을 CSV로 저장하고 통합 Markdown 보고서, manifest, 완료 표식 생성 |
| `src/visualization.py`        | 기간·피처·거시·LSTM·포트폴리오·요인·앙상블 기본 그래프 생성  |
| `src/presentation_visuals.py` | 저장된 실험표로 발표용 앙상블·가설 시각화 생성. 기본 실행과 별도 호출 |

### 데이터 수집·전처리

이 모듈들은 데이터 구축용으로 보존됩니다. `python run_submission.py`는 외부 데이터를 내려받거나 원천 데이터를 다시 만들지 않고 저장된 입력 파일을 읽습니다.

| 모듈                    | 역할                                                         |
| ----------------------- | ------------------------------------------------------------ |
| `src/get_tickers.py`    | 현재·과거 S&P 500 구성종목 수집, 티커 정규화와 membership 필터 |
| `src/collect_prices.py` | Yahoo → Yahoo chart → Tiingo 순서의 OHLCV·시장지수 수집      |
| `src/preprocess.py`     | 2016-01-01~2026-06-30 OHLCV 타입·정렬·중복·결측·coverage 처리 |
| `src/feature.py`        | OHLCV와 벤치마크를 이용한 피처 데이터셋 생성                 |
| `src/validate.py`       | 최종 OHLCV 스키마, 자료형, 중복, 정렬, 결측과 수집 성공·실패 정합성 검사 |

Tiingo 폴백을 직접 사용할 때는 실행 환경에 `TIINGO_API_KEY`가 필요합니다. 키가 없어도 Yahoo 계열 수집은 사용할 수 있으며, Tiingo를 시도할 수 없는 상태는 실패 사유로 기록됩니다.

## 입력 데이터

기본 경로는 `src/config.py`에서 프로젝트 루트를 기준으로 결정합니다.

| 상대경로                                      | 용도                        |
| --------------------------------------------- | --------------------------- |
| `data/processed/final_df.parquet`             | 모델링 기준 종목 OHLCV 패널 |
| `data/raw/sp500_beta_df.parquet`              | 벤치마크 가격과 거래일 달력 |
| `data/raw/cache/sp500_membership_history.csv` | 과거 S&P 500 편입·제외 이력 |
| `data/raw/sp500_universe.csv`                 | 종목·섹터 정보              |
| `data/raw/validation_macro_market_raw.csv`    | 거시·시장 피처              |

전처리 파이프라인에서 사용하는 주요 파일은 다음과 같습니다.

| 상대경로                              | 용도                                           |
| ------------------------------------- | ---------------------------------------------- |
| `data/raw/final_df_raw.parquet`       | 수집된 원본 가격 패널                          |
| `data/raw/final_missing_df.csv`       | 수집 폴백 후 실패한 종목과 사유                |
| `data/processed/final_df.parquet`     | 분석에 사용하는 최종 OHLCV                     |
| `data/processed/coverage_df.csv`      | 종목별 기간 coverage와 품질 플래그             |
| `data/processed/final_missing_df.csv` | 수집·기간·전처리 후 사용할 수 없는 종목과 사유 |

`final_missing_df`는 과거 편출·M&A 종목 전체 목록이 아니라 수집 또는 전처리 과정에서 최종 사용이 불가능한 종목 목록입니다. 대용량 parquet와 API 캐시는 원격 저장소 정책에 따라 Git 추적에서 제외할 수 있으며, 이 경우 실행 전에 위 필수 입력을 별도로 배치해야 합니다.

입력·출력 위치를 바꾸려면 환경변수를 사용할 수 있습니다.

```bat
set VALIDATION_PROJECT_ROOT=C:\workspaces\middle\final
set VALIDATION_OUTPUT_DIR=C:\workspaces\middle\final\outputs
python run_submission.py
```

## 실행 결과

실행할 때마다 `outputs/<UTC 실행시각>/`에 새 폴더가 만들어집니다. 기존 실행 결과는 덮어쓰지 않습니다.

```text
outputs/<UTC 실행시각>/
├── UNIFIED_VALIDATION_REPORT.md
├── manifest.json
├── SUCCESS.txt
├── tables/
│   └── *.csv
└── figures/
    └── *.png
```

우선 확인할 결과는 다음과 같습니다.

| 산출물                                            | 내용                                                         |
| ------------------------------------------------- | ------------------------------------------------------------ |
| `tables/profit_final_ensemble_recommendation.csv` | 최신 Top-10, 모델 점수·백분위, CV 비중, 모멘텀·베타·RSI 등 종목별 설명 |
| `tables/cv_locked_weights.csv`                    | 4개 모델의 평균 CV IC와 최종 ReLU 비중                       |
| `tables/cv_locked_component_ic.csv`               | 날짜·모델별 CV IC                                            |
| `tables/cv_locked_training_audit.csv`             | CV Fold 시작일과 학습 라벨 마감 감사                         |
| `tables/profit_final_ensemble_protocol.csv`       | 최종 모델명, 비중 선정 규칙과 검증 역할                      |
| `tables/profit_final_ensemble_summary.csv`        | CV 고정 모델의 OOS 요약 성과                                 |
| `tables/profit_final_ensemble_periods.csv`        | OOS Fold별 신호·진입·청산과 정산 결과                        |
| `tables/profit_final_ensemble_holdings.csv`       | 과거 OOS Fold의 선정 종목                                    |
| `tables/profit_candidate_summary.csv`             | 비교 후보의 OOS 성과와 탐색적 순위                           |
| `UNIFIED_VALIDATION_REPORT.md`                    | 가설·실험·검증 결과와 해석을 모은 통합 보고서                |
| `manifest.json`                                   | 입력 파일 해시, Python·패키지·운영체제, Stable 정책과 한계   |
| `SUCCESS.txt`                                     | 정상 완료 여부                                               |

최신 추천 파일의 `pred`는 모델별 후보군 백분위 순위를 CV 비중으로 합산한 값입니다. 기대수익률 단위가 아닙니다. `recommendation_explanation`은 수치와 선정 규칙을 사람이 읽을 수 있게 풀어 쓴 설명이며, SHAP 기반 기여도나 인과적 설명은 아닙니다.
