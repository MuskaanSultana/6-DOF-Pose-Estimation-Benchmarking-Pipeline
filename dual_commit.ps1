param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Message
)

$ErrorActionPreference = "Stop"

$User1_Name = "Muskaan Sultana"
$User1_Email = "muskaan.sultana2002@gmail.com"

$User2_Name = "Adithya Lee"
$User2_Email = "adithya040@gmail.com"

# Check if inside a Git repository
$gitRev = git rev-parse --is-inside-work-tree 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Not a git repository." -ForegroundColor Red
    exit 1
}

# Determine current branch
$branch = (git branch --show-current).Trim()
if (-not $branch) {
    $branch = "main"
}

# Inspect existing remotes
$remotes = git remote
if (-not $remotes) {
    Write-Host "Error: No git remotes configured for this repository." -ForegroundColor Red
    exit 1
}

# Determine repository name dynamically from existing remote URL
$repoUrl = git remote get-url origin 2>$null
if (-not $repoUrl) {
    $firstRemote = ($remotes -split "`r?`n")[0].Trim()
    $repoUrl = git remote get-url $firstRemote 2>$null
}

$repoName = ""
if ($repoUrl -match "[:/]([^/:]+)/([^/:]+?)(\.git)?$") {
    $repoName = $matches[2]
}

if (-not $repoName) {
    $repoName = (Get-Item -Path ".").Name
}

# Ensure adithya and muskaan remotes exist in this repo
$hasAdithya = $remotes -contains "adithya"
$hasMuskaan = $remotes -contains "muskaan"

if (-not $hasAdithya) {
    $adithyaUrl = "https://github.com/adithyalee/$repoName.git"
    git remote add adithya $adithyaUrl 2>$null
    Write-Host "Configured adithya remote: $adithyaUrl" -ForegroundColor Gray
}

if (-not $hasMuskaan) {
    $muskaanUrl = "https://github.com/MuskaanSultana/$repoName.git"
    git remote add muskaan $muskaanUrl 2>$null
    Write-Host "Configured muskaan remote: $muskaanUrl" -ForegroundColor Gray
}

Write-Host "Staging all changes..." -ForegroundColor Cyan
git add -A

$hasChanges = git status --porcelain

# 1. Commit as Adithya Lee
Write-Host "`n[1/2] Committing as $User2_Name for adithyalee..." -ForegroundColor Green
if ($hasChanges) {
    git commit --author="$User2_Name <$User2_Email>" -m "$Message"
} else {
    git commit --allow-empty --author="$User2_Name <$User2_Email>" -m "$Message"
}

Write-Host "Pushing to adithyalee repository ($branch)..." -ForegroundColor Cyan
try {
    git push -u adithya $branch
} catch {
    Write-Host "Push to adithyalee failed: $_" -ForegroundColor Yellow
}

# 2. Commit as Muskaan Sultana
Write-Host "`n[2/2] Committing as $User1_Name for MuskaanSultana..." -ForegroundColor Green
git commit --allow-empty --author="$User1_Name <$User1_Email>" -m "$Message"

Write-Host "Pushing to MuskaanSultana repository ($branch)..." -ForegroundColor Cyan
try {
    git push -u muskaan $branch
} catch {
    Write-Host "Push to MuskaanSultana failed: $_" -ForegroundColor Yellow
}

Write-Host "`nSuccessfully finished dual sync for project '$repoName' on branch '$branch'!" -ForegroundColor Green
