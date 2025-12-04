# LCDP-Sim Quick Start Script for Windows PowerShell

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "LCDP-Sim Quick Start" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan

# Check if conda is available
Write-Host "`nChecking conda installation..." -ForegroundColor Yellow
try {
    $condaVersion = conda --version 2>&1
    Write-Host "Found $condaVersion" -ForegroundColor Green
}
catch {
    Write-Host "Error: Conda is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Anaconda or Miniconda first" -ForegroundColor Yellow
    exit 1
}

# Create conda environment
Write-Host "`nCreating conda environment 'lcdp'..." -ForegroundColor Yellow
Write-Host "Python version: 3.8" -ForegroundColor Gray
conda create -n lcdp python=3.8 -y

# Activate conda environment
Write-Host "`nActivating conda environment..." -ForegroundColor Yellow
conda activate lcdp

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
pip install --upgrade pip

# Install requirements
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Gray
pip install -r requirements.txt

# Install package in editable mode
Write-Host "`nInstalling LCDP-Sim..." -ForegroundColor Yellow
pip install -e .

# Create necessary directories
Write-Host "`nCreating necessary directories..." -ForegroundColor Yellow
$directories = @("data", "checkpoints", "logs", "videos", "visualizations")
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "  Created: $dir" -ForegroundColor Gray
    }
}

Write-Host "`n==================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Cyan

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "To activate the environment in the future, run:" -ForegroundColor White
Write-Host "   conda activate lcdp" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Collect demonstration data:" -ForegroundColor White
Write-Host "   python scripts/collect_data.py --env PickCube-v0 --num-episodes 100" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Train the model:" -ForegroundColor White
Write-Host "   python scripts/train.py --config configs/train_config.yaml" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Evaluate the policy:" -ForegroundColor White
Write-Host "   python scripts/eval.py --checkpoint checkpoints/best.pth --num-episodes 50" -ForegroundColor Gray
Write-Host ""
Write-Host "For more information, see docs/USAGE.md" -ForegroundColor Yellow
Write-Host ""

# Keep the window open
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
