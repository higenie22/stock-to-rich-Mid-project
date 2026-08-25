# S&P 500 10-year preprocessing pipeline

현재 시점의 S&P 500 구성종목을 대상으로 최근 10년 OHLCV를 수집하고, ML팀이
재사용할 수 있는 전처리 데이터셋을 만드는 파이프라인입니다. 과거 구성종목 복원,
생존자 편향 보정, 수익률 파생변수 생성, 모델링은 범위에 포함하지 않습니다.

## 실행 순서

프로젝트 루트에서 의존성을 설치한 뒤 `notebooks/`의 노트북을 순서대로 실행합니다.

```powershell
python -m pip install -r requirements.txt
```

1. `01_collect_data.ipynb` — 현재 구성종목과 10년 OHLCV를 API에서 한 번 수집
2. `02_preprocessing.ipynb` — 저장된 raw 파일만 읽어 전처리(반복 실행 가능)
3. `03_check_data.ipynb` — 저장된 결과를 검증하고 상태를 요약(반복 실행 가능)

Tiingo 폴백을 사용하려면 수집 전에 환경 변수 `TIINGO_API_KEY`를 설정합니다. 키가
없어도 Yahoo와 Yahoo chart API 수집은 실행되며, 최종 실패 사유에 Tiingo 키 부재가
기록됩니다.

## 산출물

- `data/raw/final_df_raw.parquet`: 수집 완료 원본 가격 패널
- `data/raw/final_missing_df.csv`: 수집 폴백 후 실패한 종목의 raw 체크포인트
- `data/raw/sp500_universe.csv`: 최초 수집 대상인 현재 S&P 500 종목
- `data/processed/final_df.parquet`: ML팀이 사용하는 최종 OHLCV 데이터셋
- `data/processed/coverage_df.csv`: 종목별 10년 coverage 및 품질 플래그 요약
- `data/processed/final_missing_df.csv`: 수집·기간·전처리 후 최종 사용 불가능한 종목과 사유

대용량 parquet와 수집 캐시는 Git에서 제외됩니다. `final_missing_df`는 과거 편출·M&A
목록이 아니라, 현재 분석 대상 중 최종 수집 실패한 종목만 의미합니다.

## 모듈 역할

- `src/get_tickers.py`: 현재 구성종목 수집, 정규화, 검증된 티커 변경 적용
- `src/collect_prices.py`: yfinance → Yahoo chart → Tiingo 수집 및 `make_df()`
- `src/preprocess.py`: 타입·정렬·중복·결측·기본 품질 플래그·coverage 처리
- `src/validate.py`: 최종 스키마와 universe/성공/실패/coverage 정합성 검증
