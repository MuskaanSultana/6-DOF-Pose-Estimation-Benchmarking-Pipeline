param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Message
)

$ErrorActionPreference = "Stop"

$User1_Name = "Muskaan Sultana"
$User1_Email = "muskaan.sultana2002@gmail.com"

$User2_Name = "Adithya Lee"
$User2_Email = "adithya040@gmail.com"

Write-Host "Staging all changes..." -ForegroundColor Cyan
git add -A

$hasChanges = git status --porcelain
if (-not $hasChanges) {
    Write-Host "No file modifications detected. Creating synchronizing commits..." -ForegroundColor Yellow
}

# 1. Commit and push directly as Adithya Lee to adithyalee repository
Write-Host "`n[1/2] Committing as $User2_Name for adithyalee repository..." -ForegroundColor Green
if ($hasChanges) {
    git commit --author="$User2_Name <$User2_Email>" -m "$Message"
} else {
    git commit --allow-empty --author="$User2_Name <$User2_Email>" -m "$Message"
}

Write-Host "Pushing to adithyalee repository..." -ForegroundColor Cyan
git push adithya main

# 2. Add an individual commit as Muskaan Sultana for MuskaanSultana repository
Write-Host "`n[2/2] Committing as $User1_Name for MuskaanSultana repository..." -ForegroundColor Green
git commit --allow-empty --author="$User1_Name <$User1_Email>" -m "$Message"

Write-Host "Pushing to MuskaanSultana repository..." -ForegroundColor Cyan
git push muskaan main

Write-Host "`nSuccessfully pushed!" -ForegroundColor Green
Write-Host "- https://github.com/adithyalee/6-DOF-Pose-Estimation-Benchmarking-Pipeline shows $User2_Name as commit author." -ForegroundColor Cyan
Write-Host "- https://github.com/MuskaanSultana/6-DOF-Pose-Estimation-Benchmarking-Pipeline shows $User1_Name as commit author." -ForegroundColor Cyan
