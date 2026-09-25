# Instalador del pipeline de migración celular (Windows).
#
#   Doble clic en instalar.bat, o en PowerShell:
#   powershell -ExecutionPolicy Bypass -File instalar.ps1 [-Variante cuda|cpu]
#
# Crea venv\ en esta carpeta, instala PyTorch con CUDA si hay una GPU
# NVIDIA (en Windows PyTorch no ofrece ROCm), el resto de dependencias,
# descarga el modelo Cellpose-SAM (~1.2 GB) y verifica la instalación.
param([ValidateSet("", "cuda", "cpu")][string]$Variante = "")
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$TORCH = "2.14.0"
$TORCHVISION = "0.29.0"

if (-not $Variante) {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { $Variante = "cuda" } else { $Variante = "cpu" }
}
if ($Variante -eq "cuda") {
    $smi = (nvidia-smi | Out-String)
    $m = [regex]::Match($smi, "CUDA Version:\s*(\d+)\.")
    if ($m.Success -and [int]$m.Groups[1].Value -ge 13) { $idx = "cu130" } else { $idx = "cu126" }
    $gpu = (nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | Select-Object -First 1)
    Write-Host ">> GPU NVIDIA: $gpu  ->  PyTorch $idx"
} else {
    $idx = "cpu"
    Write-Host ">> Sin GPU NVIDIA  ->  PyTorch CPU (lento)"
}

# Python 3.10 o más nuevo (lanzador "py" de python.org, o "python")
function Invocar([string[]]$cmd, [string[]]$extra) {
    $resto = @()
    if ($cmd.Length -gt 1) { $resto = $cmd[1..($cmd.Length - 1)] }
    & $cmd[0] @resto @extra
}
$py = $null
foreach ($c in @(@("py", "-3.12"), @("py", "-3.13"), @("py", "-3.11"), @("py", "-3.10"), @("py", "-3"), @("python"))) {
    try {
        $v = Invocar $c @("-c", "import sys; print(sys.version_info >= (3, 10))") 2>$null
        if ($v -eq "True") { $py = $c; break }
    } catch {}
}
if (-not $py) { throw "No se encontró Python 3.10 o más nuevo. Instalarlo desde https://www.python.org (marcar 'Add python.exe to PATH')." }

if (-not (Test-Path venv)) { Invocar $py @("-m", "venv", "venv") }
$vpy = "venv\Scripts\python.exe"
& $vpy -m pip install --upgrade pip wheel
& $vpy -m pip install "torch==$TORCH" "torchvision==$TORCHVISION" --index-url "https://download.pytorch.org/whl/$idx"
& $vpy -m pip install -r requisitos\base.txt
Write-Host ">> verificando la instalación y descargando el modelo (la primera vez tarda)..."
& $vpy scripts\verificar_instalacion.py
if ($LASTEXITCODE -ne 0) { throw "La verificación falló (ver arriba)." }
Write-Host ""
Write-Host "Listo. Para abrir la interfaz: doble clic en iniciar_interfaz.bat"
