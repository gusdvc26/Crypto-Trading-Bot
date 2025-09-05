param(
  [ValidateSet("setup","ingest","label","dataset","gold","report","paper","smoke","help")]
  [string]$Task = "help"
)

function Resolve-Dataset {
  param([string]$Start,[string]$End)
  # Common naming forms:
  $candidates = @(
    "data/processed/train/dataset_${Start}_${End}.parquet",                                      # 2025-08-29_2025-08-30
    "data/processed/train/dataset_$($Start.Replace('-',''))_$($End.Replace('-','')).parquet",    # 20250829_20250830
    "data/processed/train/dataset_$($Start.Substring(5).Replace('-',''))_$($End.Substring(5).Replace('-','')).parquet" # 0829_0830
  )
  foreach ($p in $candidates) {
    if (Test-Path $p) { return (Resolve-Path $p).Path }
  }
  # Fallback: scan all dataset_*.parquet and try to match tokens
  $all = Get-ChildItem -Path "data/processed/train" -Filter "dataset_*.parquet" -File -ErrorAction SilentlyContinue
  if ($all) {
    $tokFull1 = $Start.Replace('-',''); $tokFull2 = $End.Replace('-','')
    $tokMd1   = $Start.Substring(5).Replace('-',''); $tokMd2 = $End.Substring(5).Replace('-','')
    $pick = $all | Where-Object {
      $_.Name -match $tokFull1 -and $_.Name -match $tokFull2 -or
      $_.Name -match $tokMd1   -and $_.Name -match $tokMd2
    } | Select-Object -First 1
    if ($pick) { return $pick.FullName }
  }
  return $null
}

switch ($Task) {
  "setup"   { Write-Host "Create .venv and install -r requirements.txt (manual)." }
  "ingest"  { & .\.venv\Scripts\python.exe -m src.app.core_watch }
  "label"   { & .\.venv\Scripts\python.exe -m src.pipeline.build_day --start $env:START --end $env:END }
  "dataset" { & .\.venv\Scripts\python.exe -m src.processing.dataset_builder --start $env:START --end $env:END --out "data/processed/train/dataset_${env:START}_${env:END}" }
  "gold"    { & .\.venv\Scripts\python.exe -m src.pipeline.materialize_gold --dataset "data/processed/train/dataset_${env:START}_${env:END}.parquet" --date $env:DAY --venue coinbase }
  "report"  { & .\.venv\Scripts\python.exe -m src.reports.quick_report --data "data/processed/train/dataset_${env:START}_${env:END}.parquet" --h 5m --top 0.10 }

  "paper" {
    if (-not $env:START -or -not $env:END) { Write-Error "Set START/END, e.g. `$env:START='2025-08-29'; `$env:END='2025-08-30'"; break }

    function Resolve-Dataset {
      param([string]$Start,[string]$End)
      $candidates = @(
        "data/processed/train/dataset_${Start}_${End}.parquet",
        "data/processed/train/dataset_$($Start.Substring(5).Replace('-',''))_$($End.Substring(5).Replace('-','')).parquet",
        "data/processed/train/dataset_$($Start.Replace('-',''))_$($End.Replace('-','')).parquet"
      )
      foreach ($p in $candidates) { if (Test-Path $p) { return (Resolve-Path $p).Path } }
      $all = Get-ChildItem -Path "data/processed/train" -Filter "dataset_*.parquet" -File -ErrorAction SilentlyContinue
      if ($all) {
        $tokFull1 = $Start.Replace('-',''); $tokFull2 = $End.Replace('-','')
        $tokMd1   = $Start.Substring(5).Replace('-',''); $tokMd2 = $End.Substring(5).Replace('-','')
        $pick = $all | Where-Object { ($_.Name -match $tokFull1 -and $_.Name -match $tokFull2) -or ($_.Name -match $tokMd1 -and $_.Name -match $tokMd2) } | Select-Object -First 1
        if ($pick) { return $pick.FullName }
      }
      return $null
    }

    $signals = Resolve-Dataset -Start $env:START -End $env:END
    if (-not $signals) {
      Write-Error "No dataset parquet found for START=$env:START END=$env:END"
      break
    }
    Write-Host "Using dataset: $signals"
    $suffix = [System.IO.Path]::GetFileNameWithoutExtension($signals); if ($suffix.StartsWith("dataset_")) { $suffix = $suffix.Substring(8) }
    $mids = "data/processed/train/mids_$suffix.parquet"

    if (-not (Test-Path $mids)) {
      Write-Host "Building mids → $mids"
      $argsList = @("--dataset", $signals, "--out", $mids)
      if ($env:MID_PRICE_COL) { $argsList += @("--price-col", $env:MID_PRICE_COL) }
      if ($env:MID_BID_COL -and $env:MID_ASK_COL) { $argsList += @("--bid-col", $env:MID_BID_COL, "--ask-col", $env:MID_ASK_COL) }
      if ($env:MID_CONSTANT) { $argsList += @("--constant", $env:MID_CONSTANT) }

      & .\.venv\Scripts\python.exe -m src.tools.build_mids @argsList
      if ($LASTEXITCODE -ne 0 -or -not (Test-Path $mids)) {
        Write-Warning "Mids build failed. Listing numeric candidates to help choose overrides…"
        & .\.venv\Scripts\python.exe -m src.tools.build_mids --dataset $signals --out $mids --list
        Write-Host "Set one of: `$env:MID_PRICE_COL='<col>'  OR  `$env:MID_BID_COL='<bid>'; `$env:MID_ASK_COL='<ask>'  OR  `$env:MID_CONSTANT='100'  then re-run: ./scripts/do.ps1 paper"
        break
      }
    } else {
      Write-Host "Found mids: $mids"
    }

    $outdir = "data/exec/paper/$suffix"
    & .\.venv\Scripts\python.exe -m src.exec.paper_router --signals $signals --mid $mids --outdir $outdir
  }

  "smoke" {
    $env:DATASET_TIME_TOL_MS="2500"; $env:ENFORCE_QA="1"
    & .\.venv\Scripts\python.exe -m pytest -q
    & .\.venv\Scripts\python.exe -m src.pipeline.build_day --start $env:START --end $env:END
    & .\.venv\Scripts\python.exe -m src.processing.dataset_builder --start $env:START --end $env:END --out "data/processed/train/dataset_${env:START}_${env:END}"
    & .\.venv\Scripts\python.exe -m src.pipeline.materialize_gold --dataset "data/processed/train/dataset_${env:START}_${env:END}.parquet" --date $env:DAY --venue coinbase
    & .\.venv\Scripts\python.exe -m src.alpha.eval --data "data/processed/train/dataset_${env:START}_${env:END}.parquet" --top 0.10 --h 5m
  }

  default { Write-Host "Tasks: setup | ingest | label | dataset | gold | report | paper | smoke | help" }
}