param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Message
)

# User configurations
$User1_Name = "Muskaan Sultana"
$User1_Email = "muskaan.sultana2002@gmail.com"

$User2_Name = "Adithya Lee"
$User2_Email = "adithya040@gmail.com"

Write-Host "Staging all changes..." -ForegroundColor Cyan
git add -A

# Check if there are changes staged
$status = git status --porcelain
if (-not $status) {
    Write-Host "No changes detected to commit. Creating dual sync check-in..." -ForegroundColor Yellow
    git commit --allow-empty --author="$User1_Name <$User1_Email>" -m "$Message"
} else {
    Write-Host "Creating commit for $User1_Name..." -ForegroundColor Green
    git commit --author="$User1_Name <$User1_Email>" -m "$Message"
}

# Create second individual commit for User 2 (guarantees green box activity on GitHub)
Write-Host "Creating sync commit for $User2_Name..." -ForegroundColor Green
git commit --allow-empty --author="$User2_Name <$User2_Email>" -m "$Message [sync]"

Write-Host "Pushing to remotes..." -ForegroundColor Cyan
git push

Write-Host "`nDone! Both accounts ($User1_Name & $User2_Name) have received individual commits and green box activity." -ForegroundColor Green
