# 📈 skfolio_Catton

[![Crypto Pipeline CI](https://github.com/CattonNyan/skfolio_Catton/actions/workflows/crypto_ci.yml/badge.svg)](https://github.com/CattonNyan/skfolio_Catton/actions/workflows/crypto_ci.yml)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)

> **포트폴리오 최적화(Portfolio Optimization) & Freqtrade 연계 암호화폐 자산배분 연구 저장소**

본 저장소는 오픈소스 포트폴리오 최적화 라이브러리인 [`skfolio`](https://github.com/skfolio/skfolio)를 기반으로, **암호화폐(Crypto) 및 Freqtrade 자동매매 봇 연계 자산배분**을 실습하고 연구하기 위한 독립 프로젝트입니다.

---

## 🌟 주요 기능 및 특징

1. **scikit-learn 호환 최적화 엔진**:
   - **Mean-Variance**: 최대 샤프 지수(Maximum Sharpe Ratio), 최소 분산(Minimum Variance)
   - **Risk Budgeting (Risk Parity)**: 자산별 위험 기여도를 동일하게 맞추는 위험 균등 배분
   - **Minimum Semi-Variance**: 하방 위험(급락 손실)만을 최소화하는 비대칭 리스크 관리
2. **계층적 리스크 패리티 (HRP) & 머신러닝 군집화 (`scripts/crypto_hrp_clustering.py`)**:
   - 코인 간 상관관계 거리 행렬(Dendrogram) 트리 군집 분석
   - 공분산 역행렬을 사용하지 않아 암호화폐 시장의 노이즈와 다중공선성에 강인한 차세대 자산배분 모델
3. **Freqtrade 트레이딩 봇 자동 연계 파이프라인 (`scripts/crypto_portfolio_optimizer.py`)**:
   - Freqtrade에서 다운로드한 OHLCV 시장 데이터(`.feather`)를 자동 감지 및 로드
   - 산출된 최적 비중을 Freqtrade `config.json`의 `pair_whitelist` 및 `pair_stake_amounts`로 즉시 내보내기 지원 (`--export-freqtrade`)
4. **실시간 거래소 시세 수집기 (`scripts/fetch_live_crypto.py`)**:
   - API 키 없이도 바이낸스(USDT 마켓) 및 **업비트(KRW 원화 마켓)** 공개 캔들 시세를 즉시 다운로드하여 모델에 투입
5. **한국어 인터랙티브 튜토리얼 노트북 (`notebooks/`)**:
   - `01_quickstart_portfolio_optimization.ipynb`: 주식/일반 자산군 포트폴리오 최적화 기초
   - `02_crypto_risk_parity_allocation.ipynb`: 암호화폐 특화 리스크 패리티 자산배분 실습
6. **Windows 원클릭 실행 & GitHub Actions CI**:
   - PowerShell 원클릭 설치 스크립트(`setup.ps1`) 및 GitHub 푸시 시 자동 단위 테스트 파이프라인 구축

---

## 🚀 빠른 시작 (Quick Start)

### 1. 환경 설정 (Windows PowerShell)

```powershell
# Windows 자동 설치 스크립트 실행 (가상환경 .venv 생성 및 패키지 설치)
.\setup.ps1

# 가상환경 활성화
.\.venv\Scripts\Activate.ps1

# 설치 검증
python scripts/verify_environment.py
```

### 2. 크립토 포트폴리오 최적화 및 Freqtrade 설정 자동 내보내기

```powershell
# 1) Freqtrade 데이터 자동 탐색 및 최적화 실행 (리스크 패리티 비중을 Freqtrade config로 내보내기)
python scripts/crypto_portfolio_optimizer.py --timeframe 15m --export-freqtrade ../freqtrade/user_data/config.json --wallet-size 10000

# 2) 계층적 리스크 패리티(HRP) 트리 군집화 및 상관계수 분석
python scripts/crypto_hrp_clustering.py --timeframe 15m

# 3) 실시간 업비트 원화(KRW) 마켓 시세 수집
python scripts/fetch_live_crypto.py --exchange upbit --pairs BTC/KRW ETH/KRW SOL/KRW --timeframe 1h
```

### 3. 주피터 노트북 실행

```powershell
jupyter lab
```
- 브라우저가 열리면 `notebooks/` 폴더 내의 한국어 튜토리얼을 단계별로 실행해볼 수 있습니다.

---

## 📁 디렉토리 구조

```text
skfolio_Catton/
├── .github/workflows/
│   └── crypto_ci.yml                  # GitHub Actions 자동 테스트 CI 파이프라인
├── notebooks/                         # 한국어 실습 및 튜토리얼 주피터 노트북
│   ├── 01_quickstart_portfolio_optimization.ipynb
│   └── 02_crypto_risk_parity_allocation.ipynb
├── scripts/                           # 실행 및 자동화 스크립트
│   ├── crypto_portfolio_optimizer.py  # Freqtrade 연동 크립토 포트폴리오 최적화 파이프라인
│   ├── crypto_hrp_clustering.py       # HRP 계층적 군집화 및 상관관계 분석
│   ├── fetch_live_crypto.py           # 거래소(바이낸스/업비트) 실시간 데이터 수집기
│   └── verify_environment.py          # 환경 검증 스크립트
├── src/skfolio/                       # skfolio 핵심 최적화 알고리즘 엔진
├── tests/                             # 단위 테스트 모음
│   ├── test_crypto_optimizer.py
│   ├── test_hrp_clustering.py
│   └── test_live_fetcher.py
├── requirements-local.txt             # 로컬 실행 및 연구용 의존성 정의
├── setup.ps1                          # Windows 가상환경 원클릭 설치 스크립트
└── README_KR.md                       # 프로젝트 한국어 문서
```

---

## 📜 라이선스 및 출처 (Credits & License)

- 본 프로젝트는 [skfolio developers](https://github.com/skfolio/skfolio)의 `skfolio` 라이브러리를 기반으로 합니다.
- 원본 소프트웨어는 **BSD 3-Clause License**를 따르며, 전체 라이선스 본문은 [LICENSE](LICENSE) 파일에서 확인하실 수 있습니다.
- *Copyright (c) 2023-2026 The skfolio developers. All rights reserved.*
