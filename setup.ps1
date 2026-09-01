# Setup script for skfolio_Catton (Windows PowerShell)
[CmdletBinding()]
param (
    [switch]$ForceReinstall = $false
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  skfolio_Catton Windows Setup Script" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Error "Python이 시스템 PATH에 등록되어 있지 않습니다. Python 3.10 이상을 설치해주세요."
}

$pyVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "[+] 감지된 Python 버전: $pyVersion" -ForegroundColor Green

# 2. Virtual Environment Setup
$venvPath = Join-Path $PSScriptRoot ".venv"
if (-not (Test-Path $venvPath) -or $ForceReinstall) {
    if (Test-Path $venvPath) {
        Write-Host "[*] 기존 가상환경 제거 중..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venvPath
    }
    Write-Host "[*] 새로운 가상환경(.venv) 생성 중..." -ForegroundColor Yellow
    python -m venv $venvPath
} else {
    Write-Host "[+] 기존 가상환경(.venv)이 존재합니다." -ForegroundColor Green
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
$venvPip = Join-Path $venvPath "Scripts\pip.exe"

# 3. Upgrade pip
Write-Host "[*] pip 최신화..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip setuptools wheel

# 4. Install requirements
$reqFile = Join-Path $PSScriptRoot "requirements-local.txt"
if (Test-Path $reqFile) {
    Write-Host "[*] 로컬 의존성 패키지 설치 중 (시간이 다소 걸릴 수 있습니다)..." -ForegroundColor Yellow
    & $venvPip install -r $reqFile
}

# 5. Done
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  설치가 완료되었습니다!" -ForegroundColor Green
Write-Host "  가상환경 활성화 명령어:" -ForegroundColor White
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "  주피터 랩 실행:" -ForegroundColor White
Write-Host "  .\.venv\Scripts\jupyter-lab.exe" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan
