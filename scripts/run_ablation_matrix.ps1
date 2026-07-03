param(
    [string]$Python = "python",
    [string]$Config = "configs/parc_sam_ssl_3class.yaml",
    [int]$MaxIterations = 10000,
    [string]$Device = "cuda",
    [string]$OutputRoot = "outputs/ablation_matrix"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$runs = @(
    @{
        Name = "unimatch_style_student"
        Args = @("--target-mode", "hard", "--disable-sam", "--disable-prototype", "--disable-correlation", "--disable-alignment", "--disable-foreground-guard")
    },
    @{
        Name = "sam_hard_pseudo"
        Args = @("--target-mode", "hard", "--disable-prototype", "--disable-correlation", "--disable-alignment", "--disable-foreground-guard")
    },
    @{
        Name = "conformal_single"
        Args = @("--target-mode", "conformal_single")
    },
    @{
        Name = "no_foreground_guard"
        Args = @("--disable-foreground-guard")
    },
    @{
        Name = "full_v2"
        Args = @()
    }
)

foreach ($run in $runs) {
    $outDir = Join-Path $OutputRoot $run.Name
    $cmdArgs = @(
        "train.py",
        "--config", $Config,
        "--device", $Device,
        "--max-iterations", $MaxIterations,
        "--output-dir", $outDir
    )
    $cmdArgs += $run.Args
    & $Python @cmdArgs
}
