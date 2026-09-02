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
  - [6. 인터랙티브 퀀트 웹 대시보드 (Streamlit)](#6-인터랙티브-퀀트-웹-대시보드-streamlit)
  - [7. 주기적 포트폴리오 리밸런싱 롤링 백테스트](#7-주기적-포트폴리오-리밸런싱rebalancing-롤링-백테스트)
  - [8. 인터랙티브 HTML 퀀트 분석 보고서 생성](#8-인터랙티브-html-퀀트-분석-보고서-생성)
  - [9. 변동성 기반 동적 손절/익절 리스크 계산기](#9-변동성-기반-동적-손절익절-리스크-계산기)
  - [10. 최적 자산배분 비중 엑셀(CSV) 내보내기](#10-최적-자산배분-비중-엑셀csv-내보내기)
  - [11. 블랙-리터만 베이지안 자산배분](#11-블랙-리터만black-litterman-베이지안-자산배분)
  - [12. 역사적 크립토 블랙스완 스트레스 테스트](#12-역사적-크립토-블랙스완-스트레스-테스트stress-testing)
  - [13. 김치 프리미엄 & 환율 차익 분석기](#13-김치-프리미엄--환율-차익-분석기)
  - [14. Freqtrade 멀티 전략 자금 배분 최적화기](#14-freqtrade-멀티-전략-자금-배분-최적화기)
  - [15. 몬테카를로 미래 자산 경로 및 VaR 리스크 시뮬레이터](#15-몬테카를로-미래-자산-경로-및-var-리스크-시뮬레이터)
  - [16. 공포·탐욕 지수 기반 거시 국면 동적 현금 비중 조절기](#16-공포탐욕-지수-기반-거시-국면-동적-현금usdt-비중-조절기)
  - [17. 퀀트 멀티 팩터 분석 및 스마트 베타 유니버스 스크리너](#17-퀀트-멀티-팩터-분석-및-스마트-베타-유니버스-스크리너)
  - [18. Freqtrade 완전체 실전 전략 샘플](#18-freqtrade-완전체-실전-전략-샘플-strategiesskfolioenhancedatrstrategypy)
  - [19. 상관계수 붕괴 & 디커플링 감지기](#19-상관계수-붕괴--디커플링decoupling-감지기)
  - [20. 가상자산 세후 순수익률 & 세금 시뮬레이터](#20-가상자산-세후-순수익률--세금-시뮬레이터)
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

### 11. 블랙-리터만(Black-Litterman) 베이지안 자산배분 모델

시장 균형 수익률(Prior)과 트레이더의 주관적 시장 전망(Views)을 베이지안 통계로 결합하여 안정적이고 극단값 없는 최적 비중을 계산합니다:

```powershell
# 상대적 뷰(BTC가 ETH보다 +2% 초과 상승)와 절대적 뷰(SOL +5% 상승) 반영
python scripts/crypto_black_litterman.py `
  --views "BTC/USDT>ETH/USDT:0.02" "SOL/USDT:0.05" `
  --risk-aversion 2.5
```

---

### 12. 역사적 크립토 블랙스완 스트레스 테스트(Stress Testing)

2020 코로나 빔, 2022 루나 폭락, FTX 파산, 2021 중국 채굴 금지 등 크립토 역사상 최악의 쇼크 시나리오를 현재 포트폴리오에 가상 주입하여 자산 방어력과 최대 낙폭을 평가합니다:

```powershell
# 현재 포트폴리오 자본금 $10,000 기준 스트레스 테스트 시뮬레이션
python scripts/crypto_stress_tester.py --wallet-size 10000 --export-json reports/stress_test.json
```

---

### 13. 김치 프리미엄 & 실시간 환율 차익 분석기

국내 거래소(업비트 KRW)와 해외 거래소(바이낸스 USDT)의 실시간 시세를 환율(USD/KRW)로 정밀 환산하여 코인별 김치 프리미엄(%)과 차익 스프레드를 계산합니다:

```powershell
python scripts/crypto_kimchi_premium.py --usdt-krw 1350.0 --export-json reports/kimchi_premium.json
```

---

### 14. Freqtrade 멀티 전략 자금 배분 최적화기

여러 Freqtrade 매매 전략(추세추종, 역추세, 볼린저 밴드 돌파 등)의 백테스트 손익 곡선을 바탕으로, 전체 계좌의 낙폭(MDD)을 최소화하는 전략별 최적 자금 배분 비율을 도출합니다:

```powershell
# 백테스트 결과 기반 전략 간 리스크 패리티 가중치 산출
python scripts/freqtrade_strategy_optimizer.py --capital 10000 --model "Risk Parity"
```

---

### 15. 몬테카를로 미래 자산 경로 및 VaR 리스크 시뮬레이터

기하 브라운 운동(GBM)을 기반으로 향후 90일간 발생 가능한 1,000개의 가상 가격 경로를 시뮬레이션하여 95% 신뢰구간 콘 차트와 최대 손실액(VaR, CVaR), 원금 손실 확률을 계산합니다:

```powershell
python scripts/crypto_monte_carlo.py --days 90 --sims 1000 --capital 10000
```

---

### 16. 공포·탐욕 지수 기반 거시 국면 동적 현금(USDT) 비중 조절기

Alternative.me 실시간 공포·탐욕 지수를 읽어와 시장 과열(Extreme Greed) 시 최대 40%의 현금 버퍼를 강제 확보하여 자산을 보호하고, 공포 국면에서는 적극 투자로 스위칭합니다:

```powershell
python scripts/crypto_macro_regime.py --wallet-size 10000
```

---

### 17. 퀀트 멀티 팩터 분석 및 스마트 베타 유니버스 스크리너

모멘텀(Momentum), 저변동성(Low Volatility), 추세 강도(Trend Strength) 팩터를 Z-score로 정규화 합산하여 상위 우량 코인을 자동 선별(Smart Beta Universe)합니다:

```powershell
python scripts/crypto_factor_analyzer.py --lookback 60 --top-n 3
```

---

### 18. Freqtrade 완전체 실전 전략 샘플 (`strategies/SkfolioEnhancedAtrStrategy.py`)

`freqtrade-vibe-strategies`에 바로 투입할 수 있는 완전체 전략 파일입니다.
- **동적 주문 금액**: `custom_stake_amount()`에서 skfolio 최적 비중을 읽어와 코인별 매수 주문액을 자동 차등 집행
- **동적 손절/익절**: `custom_stoploss()`에서 코인별 하방 변동성 맞춤 손절폭 적용 및 수익 발생 시 브레이크이븐 트레일링

---

### 19. 상관계수 붕괴 & 디커플링(Decoupling) 감지기

비트코인(BTC)과 알트코인 간의 롤링 상관계수를 실시간 모니터링하여, 동조화가 깨지는 이상 현상(역상관, 상관계수 급락)과 분산 투자 강화 기회를 감지합니다:

```powershell
python scripts/crypto_correlation_breakdown.py --window 30 --threshold 1.8
```

---

### 20. 가상자산 세후 순수익률 & 세금 시뮬레이터

한국 가상자산 소득세(연간 기본공제 250만 원, 22% 분리과세) 규정을 반영하여, 포트폴리오 운용 및 리밸런싱에 따른 실현 손익 통산과 세금 잠식률(Tax Drag), 최종 세후 순수익률을 계산합니다:

```powershell
python scripts/crypto_tax_calculator.py --profit 12000000 --capital 50000000
```

---

## 📁 프로젝트 구조

```text
skfolio_Catton/
├── .github/workflows/
│   └── crypto_ci.yml                  # GitHub Actions 자동 테스트 CI 워크플로우
├── .streamlit/
│   └── config.toml                    # 핀테크 다크 테마 대시보드 환경설정
├── notebooks/                         # 한국어 대화형 주피터 노트북
│   ├── 01_quickstart_portfolio_optimization.ipynb
│   └── 02_crypto_risk_parity_allocation.ipynb
├── scripts/                           # 실전 자동화 스크립트 모음
│   ├── crypto_portfolio_optimizer.py  # Freqtrade 연계 크립토 최적화 파이프라인 (제약조건 지원)
│   ├── crypto_hrp_clustering.py       # HRP 계층적 트리 군집화 및 상관계수 분석
│   ├── crypto_rebalancing_backtest.py # 주기적 포트폴리오 롤링 리밸런싱 백테스트
│   ├── export_html_report.py          # 인터랙티브 HTML 퀀트 보고서 생성기
│   ├── crypto_risk_budget_calculator.py # 변동성 기반 동적 SL/TP 리스크 계산기
│   ├── crypto_black_litterman.py      # 블랙-리터만 베이지안 포트폴리오 최적화기
│   ├── crypto_stress_tester.py        # 역사적 블랙스완 스트레스 테스터
│   ├── crypto_kimchi_premium.py       # 김치 프리미엄 & 환율 차익 분석기
│   ├── freqtrade_strategy_optimizer.py # 멀티 전략 간 자금 배분 최적화기
│   ├── crypto_monte_carlo.py          # 몬테카를로 미래 자산 경로 및 VaR 시뮬레이터
│   ├── crypto_macro_regime.py         # 공포·탐욕 지수 기반 동적 현금 비중 조절기
│   ├── crypto_factor_analyzer.py      # 퀀트 멀티 팩터 분석 및 스마트 베타 스크리너
│   ├── crypto_correlation_breakdown.py # 상관계수 붕괴 & 디커플링 감지기
│   ├── crypto_tax_calculator.py       # 가상자산 세후 순수익률 & 세금 시뮬레이터
│   ├── freqtrade_stake_allocator.py   # Freqtrade 전략 동적 주문금액 연동 브릿지
│   ├── fetch_live_crypto.py           # 거래소(바이낸스/업비트) 실시간 시세 수집기
│   └── verify_environment.py          # 환경 검증 스크립트
├── strategies/                        # Freqtrade 실전 전략 모음
│   └── SkfolioEnhancedAtrStrategy.py  # skfolio 동적 비중/리스크 완전 연동 실전 전략
├── src/skfolio/                       # skfolio 핵심 최적화 알고리즘 엔진
├── tests/                             # 단위 테스트 모음 (총 80개 테스트)
│   ├── test_crypto_suite.py           # 통합 테스트 스위트 러너 (Python 3.10~3.14 호환)
│   ├── test_crypto_optimizer.py
│   ├── test_hrp_clustering.py
│   ├── test_live_fetcher.py
│   ├── test_dashboard.py
│   ├── test_rebalancing.py
│   ├── test_html_report.py
│   ├── test_risk_calculator.py
│   ├── test_freqtrade_integration.py
│   ├── test_stake_allocator.py
│   ├── test_black_litterman.py
│   ├── test_stress_tester.py
│   ├── test_kimchi_premium.py
│   ├── test_strategy_optimizer.py
│   ├── test_monte_carlo.py
│   ├── test_macro_regime.py
│   ├── test_factor_analyzer.py
│   ├── test_enhanced_strategy.py
│   ├── test_correlation_breakdown.py
│   └── test_tax_calculator.py
├── requirements-local.txt             # 로컬 개발 및 퀀트 연구용 패키지 목록
├── setup.ps1                          # Windows PowerShell 원클릭 설치 스크립트
├── app_dashboard.py                   # Streamlit 인터랙티브 웹 대시보드 (다크 테마 & 캐싱)
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
