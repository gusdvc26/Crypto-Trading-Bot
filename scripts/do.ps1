Param(
  [ValidateSet('setup','ingest','label','dataset','gold','report','smoke','help')]
  [string]$Task = 'help',
  [string]$Start = (Get-Date).ToUniversalTime().ToString('yyyy-MM-dd'),
  [string]$End = $null,
  [string]$Symbols = 'ALL',
  [string]$DatasetOut = $null,
  [string]$GoldRoot = 'data/gold/merged'
)

function Activate-Venv {
  if (-not (Test-Path .\.venv\Scripts\Activate.ps1)) {
    Write-Host "[setup] Creating virtualenv .venv ..."
    python -m venv .venv | Out-Null
  }
  Write-Host "[setup] Activating .venv ..."
  . .\.venv\Scripts\Activate.ps1
}

if (-not $End) { $End = $Start }
if (-not $DatasetOut) { $DatasetOut = "data/processed/train/dataset_${Start}_${End}" }

switch ($Task) {
  'setup' {
    Activate-Venv
    Write-Host "==> Installing requirements"
    pip install -U pip
    if (Test-Path .\requirements.txt) {
      pip install -r .\requirements.txt
    } else {
      Write-Host "[warn] requirements.txt not found, skipping."
    }
    Write-Host "==> Setup complete"
  }
  'ingest' {
    Activate-Venv
    Write-Host "==> Ingest step (placeholder): ensure decisions/raw data exist for $Start..$End"
    Write-Host "    You can run your live apps (mover_watch/sniper_paper) separately."
  }
  'label' {
    Activate-Venv
    Write-Host "==> Labeling decisions for $Start..$End"
    python -m src.pipeline.build_day --start $Start --end $End --symbols $Symbols
    Write-Host "==> Labeling complete"
  }
  'dataset' {
    Activate-Venv
    Write-Host "==> Building dataset for $Start..$End to $DatasetOut.parquet"
    python -m src.processing.dataset_builder --start $Start --end $End --symbols $Symbols --out $DatasetOut
    Write-Host "==> Dataset build complete"
  }
  'gold' {
    Activate-Venv
    Write-Host "==> Materializing gold from $DatasetOut.parquet to $GoldRoot"
    python -m src.pipeline.materialize_gold --data "$DatasetOut.parquet" --out-root $GoldRoot
    Write-Host "==> Gold materialization complete"
  }
  'report' {
    Activate-Venv
    $parquet = "$DatasetOut.parquet"
    Write-Host "==> Quick report for $parquet (5m, top 10%)"
    python -m src.reports.quick_report --data $parquet --h 5m --top 0.10
  }
  'smoke' {
    Activate-Venv
    Write-Host "==> Running smoke tests (pytest -q)"
    pytest -q
  }
  Default {
    Write-Host "Usage: pwsh -File scripts/do.ps1 -Task <setup|ingest|label|dataset|gold|report|smoke> [-Start YYYY-MM-DD] [-End YYYY-MM-DD] [-Symbols CSV] [-DatasetOut path] [-GoldRoot path]"
  }
}

