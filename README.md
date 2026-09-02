# 📈 skfolio_Catton

<div align="center">
  <img src="https://raw.githubusercontent.com/skfolio/skfolio/main/docs/_static/logo_animate.svg" width="90" alt="skfolio logo">
  <br>
  <h3>scikit-learn 기반 포트폴리오 최적화 & 암호화폐 자산배분 프레임워크</h3>
</div>

<p align="center">
  <a href="https://skfolio.org/auto_examples/index.html">
    <img src="https://raw.githubusercontent.com/skfolio/skfolio/main/docs/_static/expo.jpg" alt="skfolio 4개 핵심 시각화 그래프" width="850">
  </a>
</p>

<p align="center">
  <a href="https://github.com/CattonNyan/skfolio_Catton/actions/workflows/crypto_ci.yml"><img src="https://github.com/CattonNyan/skfolio_Catton/actions/workflows/crypto_ci.yml/badge.svg" alt="Crypto Pipeline CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" alt="License: BSD 3-Clause"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg" alt="Python 3.10+"></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-compatible-orange.svg" alt="Scikit-Learn Compatible"></a>
</p>

> **`skfolio_Catton`**은 오픈소스 포트폴리오 최적화 라이브러리인 [`skfolio`](https://github.com/skfolio/skfolio)를 기반으로, **암호화폐(Crypto) 시장과 Freqtrade 자동매매 봇에 맞춤 연계**할 수 있도록 실전 퀀트 파이프라인과 한국어 튜토리얼을 결합한 독립 프로젝트입니다.

---

## 📑 목차
- [🌟 핵심 기능](#-핵심-기능)
- [📊 지원하는 주요 최적화 모델](#-지원하는-주요-최적화-모델)
- [🚀 빠른 시작 (Windows)](#-빠른-시작-windows)
- [💡 실전 활용 가이드](#-실전-활용-가이드)
  - [1. 거래소 실시간 시세 수집 (업비트 / 바이낸스)](#1-거래소-실시간-시세-수집-업비트--바이낸스)
  - [2. HRP 머신러닝 군집화 및 상관계수 분석](#2-hrp-머신러닝-군집화-및-상관계수-분석)
  - [3. Freqtrade 연계 및 설정 파일(config.json) 자동 주입](#3-freqtrade-연계-및-설정-파일configjson-자동-주입)
  - [4. 파이썬 코드로 직접 실행하기](#4-파이썬-코드로-직접-실행하기)
  - [5. 한국어 주피터 노트북 실습](#5-한국어-주피터-노트북-실습)
- [📁 프로젝트 구조](#-프로젝트-구조)
- [🧪 단위 테스트 및 CI/CD](#-단위-테스트-및-cicd)
- [📜 라이선스 및 크레딧](#-라이선스-및-크레딧)

---

## 🌟 핵심 기능

1. **scikit-learn 완벽 호환**:
   - `fit()`, `predict()`, `GridSearchCV` 등 scikit-learn 표준 API를 그대로 사용하여 친숙하고 간결한 코드 작성이 가능합니다.
2. **Freqtrade 트레이딩 봇 자동 연계 (`scripts/crypto_portfolio_optimizer.py`)**:
   - Freqtrade가 수집한 과거 시세 데이터(`.feather`)를 자동 탐색하여 최적 배분 비중을 계산합니다.
   - 계산된 비중을 Freqtrade `config.json`의 `pair_whitelist` 및 `pair_stake_amounts`로 즉시 반영합니다.
3. **계층적 리스크 패리티 (HRP) & 머신러닝 군집화 (`scripts/crypto_hrp_clustering.py`)**:
   - 공분산 행렬의 역행렬을 계산하지 않아, 노이즈와 다중공선성이 심한 코인 시장에서 극도로 안정적인 위험 균등 배분을 구현합니다.
4. **실시간 시세 수집기 (`scripts/fetch_live_crypto.py`)**:
   - API 키 없이도 바이낸스(USDT) 및 **업비트(KRW 원화 마켓)** 시세를 1초 만에 다운로드하여 최적화에 투입합니다.
5. **한국어 인터랙티브 주피터 노트북 (`notebooks/`)**:
   - 퀀트 자산배분 기초부터 암호화폐 리스크 패리티 실습까지 한국어로 단계별 학습을 지원합니다.
6. **Windows 원클릭 실행 환경 (`setup.ps1`)**:
   - 복잡한 빌드 과정 없이 PowerShell 스크립트 한 번으로 가상환경 구축 및 필수 패키지 설치를 완료합니다.

---

## 📊 지원하는 주요 최적화 모델

| 모델명 | 설명 | 특징 및 추천 대상 |
| :--- | :--- | :--- |
| **Max Sharpe Ratio** | 샤프 지수(수익/위험 비율)를 극대화 | 기대 수익 대비 변동성이 가장 우수한 포트폴리오 |
| **Min Variance** | 전체 포트폴리오의 변동성을 극소화 | 급격한 변동성을 피하고 안정적인 자산 보존을 원할 때 |
| **Risk Parity (ERC)** | 각 자산이 전체 위험에 기여하는 비중을 동일하게 배분 | 특정 고변동성 알트코인에 위험이 집중되는 현상 방지 |
| **Min Semi-Variance** | 하방 변동성(낙폭 위험)만을 최소화 | 상승 변동성은 제한하지 않고 급락 위험만 회피 |
| **HRP (Hierarchical Risk Parity)** | 머신러닝 계층적 트리 군집화 기반 위험 배분 | 공분산 역행렬을 구하지 않아 다중공선성에 가장 강인함 |

---

## 🚀 빠른 시작 (Windows)

PowerShell 터미널에서 다음 명령어를 순서대로 실행합니다:

```powershell
# 1. 저장소 폴더 이동
cd "D:\private project\skfolio_Catton"

# 2. Windows 원클릭 가상환경 구축 스크립트 실행
.\setup.ps1

# 3. 가상환경 활성화
.\.venv\Scripts\Activate.ps1

# 4. 환경 설치 검증
python scripts/verify_environment.py
```

---

## 💡 실전 활용 가이드

### 1. 거래소 실시간 시세 수집 (업비트 / 바이낸스)

별도의 API 키 발급 없이 공개 시세를 즉시 다운로드하여 `data/live/`에 저장합니다:

```powershell
# 업비트 원화(KRW) 마켓 1시간봉 수집
python scripts/fetch_live_crypto.py --exchange upbit --pairs BTC/KRW ETH/KRW SOL/KRW XRP/KRW --timeframe 1h

# 바이낸스 USDT 마켓 15분봉 수집
python scripts/fetch_live_crypto.py --exchange binance --pairs BTC/USDT ETH/USDT SOL/USDT --timeframe 15m
```

---

### 2. HRP 머신러닝 군집화 및 상관계수 분석

코인 간의 상관관계 행렬을 계산하고 계층적 트리 군집화(Dendrogram)를 통해 최적 분산 비중을 계산합니다:

```powershell
python scripts/crypto_hrp_clustering.py --timeframe 1h
```

---

### 3. Freqtrade 연계 및 설정 파일(config.json) 자동 주입

skfolio가 계산한 최적 자산 비중을 Freqtrade 설정 파일로 자동 내보내어, 봇이 해당 비중대로 주문을 넣도록 연동합니다:

```powershell
python scripts/crypto_portfolio_optimizer.py `
  --timeframe 15m `
  --export-freqtrade ../freqtrade/user_data/config.json `
  --export-model "Risk Parity (ERC)" `
  --wallet-size 10000
```
> **결과**: `config.json`의 `pair_whitelist`가 최적 코인들로 자동 갱신되며, 총 투자금(10,000 USDT)에 따른 코인별 `pair_stake_amounts`가 자동으로 계산되어 저장됩니다. 기존 JSON은 검증 후 원자적으로 갱신되며, 손상된 설정 파일은 덮어쓰지 않습니다.

계산된 비중을 실제 주문금액에 반영하려면 `freqtrade_integration/skfolio_callbacks.py`를 Freqtrade 프로젝트에서 import할 수 있는 위치에 두고 기존 전략에 믹스인을 추가합니다:

```python
from freqtrade.strategy import IStrategy
from freqtrade_integration import SkfolioFreqtradeMixin

class MyStrategy(SkfolioFreqtradeMixin, IStrategy):
    # 기존 전략 설정과 populate_* 메서드
    pass
```

`SkfolioFreqtradeMixin.custom_stake_amount()`는 `pair_stake_amounts`를 우선 사용하고, 해당 값이 없으면 사용 가능한 지갑 금액에 `pair_weights`를 적용합니다. 설정이 잘못되었거나 목록에 없는 페어는 기본적으로 주문금액 `0`을 반환하여 신규 진입을 차단합니다. 합성데이터 결과는 Freqtrade 설정으로 내보낼 수 없습니다.

---

### 4. 파이썬 코드로 직접 실행하기

```python
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import MeanVariance, ObjectiveFunction, RiskBudgeting
from skfolio.preprocessing import prices_to_returns

# 1. 가격 데이터 로드 및 수익률 변환
prices = load_sp500_dataset()[["AAPL", "MSFT", "AMZN", "JNJ"]]
returns = prices_to_returns(prices)

# 2. 리스크 패리티(위험 균등 배분) 모델 학습
model = RiskBudgeting()
model.fit(returns)

# 3. 최적 자산 비중 출력
for asset, weight in zip(returns.columns, model.weights_):
    print(f"{asset}: {weight * 100:.2f}%")
```

---

### 5. 한국어 주피터 노트북 실습

브라우저에서 대화형으로 실습할 수 있는 튜토리얼 노트북을 제공합니다:

```powershell
jupyter lab
```
- [notebooks/01_quickstart_portfolio_optimization.ipynb](notebooks/01_quickstart_portfolio_optimization.ipynb): 포트폴리오 최적화 기본 개념 및 모델 비교
- [notebooks/02_crypto_risk_parity_allocation.ipynb](notebooks/02_crypto_risk_parity_allocation.ipynb): 변동성 큰 암호화폐를 위한 리스크 패리티 자산배분

---

### 6. 인터랙티브 웹 대시보드 (Streamlit)

마우스 클릭만으로 실시간 자산 배분 비중(파이 차트), 코인 간 상관계수 히트맵, 누적 수익률 시뮬레이션을 시각적으로 확인하고 Freqtrade로 원클릭 내보내기를 수행합니다:

```powershell
streamlit run app_dashboard.py
```
- 브라우저에서 대시보드가 열리며 알고리즘별 성과 비교 및 Freqtrade `config.json` 자동 반영이 가능합니다.

---

### 7. 주기적 포트폴리오 리밸런싱(Rebalancing) 롤링 백테스트

일정 주기(예: 50개 캔들 또는 7일)마다 자산 비중을 재조정(Rebalancing)할 때, 단순 보유(Buy & Hold) 대비 수익률과 낙폭(MDD)이 어떻게 개선되는지 워크포워드 시뮬레이션을 수행합니다:

```powershell
# 리스크 패리티 모델 기반 롤링 리밸런싱 백테스트 실행 (거래 수수료 0.1% 반영)
python scripts/crypto_rebalancing_backtest.py --model "Risk Parity" --timeframe 15m --fee 0.001

# 합성 데이터로 빠른 검증 실행
python scripts/crypto_rebalancing_backtest.py --use-synthetic --model "Equal Weight"
```
- **산출 지표**: 누적 수익률(Total Return), 최대 낙폭(MDD), 연환산 샤프 지수, 회전율(Turnover Rate), 리밸런싱 횟수 비교 요약표 출력

---

### 8. 독립 실행형 인터랙티브 HTML 퀀트 리포트 내보내기

오프라인 브라우저에서도 열어볼 수 있는 단일 HTML 보고서 파일(`reports/crypto_portfolio_report.html`)을 생성합니다:

```powershell
python scripts/export_html_report.py --output reports/crypto_portfolio_report.html --model "Risk Parity (ERC)"
```
- **포함 내용**: KPI 성과 지표 카드, 최적 자산 배분 도넛 차트, 코인 간 상관관계 히트맵, 누적 수익률 시뮬레이션 인터랙티브 차트, Freqtrade 설정 코드

---

### 9. 변동성 기반 동적 손절/익절(SL/TP) 가이드 계산기

각 코인의 하방 변동성(Semi-Deviation)과 위험-보상 비율(RR Ratio)을 분석하여 개별 코인별 권장 손절폭(Stoploss)과 익절폭(Take-Profit)을 자동 산출합니다:

```powershell
# 분석용 리스크 JSON 생성
python scripts/crypto_risk_budget_calculator.py `
  --risk-mult 2.0 `
  --rr-ratio 2.0 `
  --export-json user_data/risk_params.json

# 기존 Freqtrade config.json에 콜백용 종목별 SL/TP를 안전하게 주입
python scripts/crypto_risk_budget_calculator.py `
  --risk-mult 2.0 `
  --rr-ratio 2.0 `
  --freqtrade-config ../freqtrade/user_data/config.json
```

### 10. 실시간 시세 원스톱 최적화 및 Freqtrade 동적 주문금액 연동

1. **실시간 시세 수집 후 즉시 최적화 실행 (`--optimize`)**:
   ```powershell
   # 바이낸스 최신 캔들을 받아와서 즉시 포트폴리오 최적화 실행
   python scripts/fetch_live_crypto.py --exchange binance --optimize
   ```

2. **Freqtrade 실전 전략 주문 금액 동적 할당 (`scripts/freqtrade_stake_allocator.py`)**:
   skfolio가 산출한 비중을 Freqtrade 전략(`custom_stake_amount`)에서 단 한 줄로 불러와 실제 매수 주문 금액을 코인별 비중대로 자동 집행합니다:
   ```python
   from scripts.freqtrade_stake_allocator import get_custom_stake_amount

   def custom_stake_amount(self, pair: str, current_time, current_rate: float,
                           proposed_stake: float, min_stake: float | None, max_stake: float | None,
                           entry_tag: str | None, side: str, **kwargs) -> float:
       return get_custom_stake_amount(
           pair=pair,
           proposed_stake=proposed_stake,
           total_wallet=self.wallets.get_total_stake_amount(),
           min_stake=min_stake,
           max_stake=max_stake,
           config_path="user_data/config.json",
       )
   ```

---

## 📁 프로젝트 구조

```text
skfolio_Catton/
├── .github/workflows/
│   └── crypto_ci.yml                  # GitHub Actions 자동 테스트 CI 워크플로우
├── notebooks/                         # 한국어 대화형 주피터 노트북
│   ├── 01_quickstart_portfolio_optimization.ipynb
│   └── 02_crypto_risk_parity_allocation.ipynb
├── scripts/                           # 실전 자동화 스크립트 모음
│   ├── crypto_portfolio_optimizer.py  # Freqtrade 연계 크립토 최적화 파이프라인
│   ├── crypto_hrp_clustering.py       # HRP 계층적 트리 군집화 및 상관계수 분석
│   ├── crypto_rebalancing_backtest.py # 주기적 포트폴리오 롤링 리밸런싱 백테스트
│   ├── export_html_report.py          # 인터랙티브 HTML 퀀트 보고서 생성기
│   ├── crypto_risk_budget_calculator.py # 변동성 기반 동적 SL/TP 리스크 계산기
│   ├── freqtrade_stake_allocator.py   # Freqtrade 전략 동적 주문금액 연동 브릿지
│   ├── fetch_live_crypto.py           # 거래소(바이낸스/업비트) 실시간 시세 수집기
│   └── verify_environment.py          # 환경 검증 스크립트
├── src/skfolio/                       # skfolio 핵심 최적화 알고리즘 엔진
├── tests/                             # 단위 테스트 모음
│   ├── test_crypto_suite.py           # 통합 테스트 스위트 러너 (Python 3.10~3.14 호환)
│   ├── test_crypto_optimizer.py
│   ├── test_hrp_clustering.py
│   ├── test_live_fetcher.py
│   ├── test_dashboard.py
│   ├── test_rebalancing.py
│   ├── test_html_report.py
│   ├── test_risk_calculator.py
│   ├── test_freqtrade_integration.py
│   └── test_stake_allocator.py
├── requirements-local.txt             # 로컬 개발 및 퀀트 연구용 패키지 목록
├── setup.ps1                          # Windows PowerShell 원클릭 설치 스크립트
├── app_dashboard.py                   # Streamlit 인터랙티브 웹 대시보드
├── README.md                          # 프로젝트 메인 한국어 안내 문서
└── LICENSE                            # BSD 3-Clause 라이선스 전문
```

---

## 🧪 단위 테스트 및 CI/CD

프로젝트의 모든 핵심 모듈은 통합 테스트 스위트를 통해 검증됩니다:

```powershell
python tests/test_crypto_suite.py
```
```text
test_export_freqtrade_allocation ... ok
test_find_freqtrade_data_dirs ... ok
test_synthetic_data_generation ... ok
test_correlation_matrix_computation ... ok
test_save_market_data ... ok

Ran 5 tests in 0.026s -> OK
```

코드가 `main` 브랜치에 푸시될 때마다 **GitHub Actions**가 클라우드에서 위 테스트를 자동으로 수행하여 코드 무결성을 보장합니다.

---

## 📜 라이선스 및 크레딧

- 본 프로젝트는 [skfolio developers](https://github.com/skfolio/skfolio)가 개발한 오픈소스 `skfolio` 라이브러리를 기반으로 합니다.
- 원본 소프트웨어는 **BSD 3-Clause License**를 따르며, 라이선스 전문은 [LICENSE](LICENSE) 파일에서 확인하실 수 있습니다.
- *Copyright (c) 2023-2026 The skfolio developers. All rights reserved.*
