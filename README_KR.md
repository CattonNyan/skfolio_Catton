# 📈 skfolio_Catton

> **포트폴리오 최적화(Portfolio Optimization) & Freqtrade 연계 암호화폐 자산배분 연구 저장소**

본 저장소는 오픈소스 포트폴리오 최적화 라이브러리인 [`skfolio`](https://github.com/skfolio/skfolio)를 기반으로, **암호화폐(Crypto) 및 Freqtrade 자동매매 봇 연계 자산배분**을 실습하고 연구하기 위한 독립 프로젝트입니다.

---

## 🌟 주요 기능 및 특징

1. **scikit-learn 호환 포트폴리오 최적화 엔진**:
   - **Mean-Variance**: 최대 샤프 지수(Maximum Sharpe Ratio), 최소 분산(Minimum Variance)
   - **Risk Budgeting (Risk Parity)**: 자산별 위험 기여도를 동일하게 맞추는 위험 균등 배분
   - **Minimum Semi-Variance**: 하방 위험(급락 손실)만을 최소화하는 비대칭 리스크 관리
2. **Freqtrade 시세 데이터 연동 파이프라인 (`scripts/crypto_portfolio_optimizer.py`)**:
   - Freqtrade에서 다운로드한 OHLCV 시장 데이터(`.feather`)를 자동 감지 및 로드
   - 멀티 코인(BTC, ETH, SOL 등)의 최적 투자 비중 산출 및 성과 비교 요약 제공
3. **한국어 인터랙티브 튜토리얼 노트북 (`notebooks/`)**:
   - `01_quickstart_portfolio_optimization.ipynb`: 주식/일반 자산군 포트폴리오 최적화 기초
   - `02_crypto_risk_parity_allocation.ipynb`: 암호화폐 특화 리스크 패리티 자산배분 실습
4. **Windows 원클릭 실행 환경 (`setup.ps1`)**:
   - PowerShell 스크립트로 가상환경 생성 및 의존성 패키지 자동 설치 지원

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

### 2. 크립토 포트폴리오 최적화 실행

Freqtrade 데이터가 존재하면 자동으로 탐색하여 로드하며, 없는 경우 현실적인 암호화폐 시뮬레이션 데이터를 바탕으로 최적 비중을 계산합니다:

```powershell
# Freqtrade 15분봉 데이터 자동 탐색 및 최적화 실행
python scripts/crypto_portfolio_optimizer.py --timeframe 15m

# 합성 크립토 샘플 데이터로 강제 실행
python scripts/crypto_portfolio_optimizer.py --use-synthetic
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
├── notebooks/                     # 한국어 실습 및 튜토리얼 주피터 노트북
│   ├── 01_quickstart_portfolio_optimization.ipynb
│   └── 02_crypto_risk_parity_allocation.ipynb
├── scripts/                       # 실행 및 자동화 스크립트
│   ├── crypto_portfolio_optimizer.py  # Freqtrade 연동 크립토 포트폴리오 최적화
│   └── verify_environment.py          # 환경 검증 스크립트
├── src/skfolio/                   # skfolio 핵심 최적화 알고리즘 엔진
├── tests/                         # 단위 테스트 모음
├── requirements-local.txt         # 로컬 실행 및 연구용 의존성 정의
├── setup.ps1                      # Windows 가상환경 원클릭 설치 스크립트
└── README_KR.md                   # 프로젝트 한국어 문서
```

---

## 📜 라이선스 및 출처 (Credits & License)

- 본 프로젝트는 [skfolio developers](https://github.com/skfolio/skfolio)의 `skfolio` 라이브러리를 기반으로 합니다.
- 원본 소프트웨어는 **BSD 3-Clause License**를 따르며, 전체 라이선스 본문은 [LICENSE](LICENSE) 파일에서 확인하실 수 있습니다.
- *Copyright (c) 2023-2026 The skfolio developers. All rights reserved.*
