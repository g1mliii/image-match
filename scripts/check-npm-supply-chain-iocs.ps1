param(
    [string]$Root = "."
)

$ErrorActionPreference = "Stop"

$resolvedRoot = Resolve-Path -LiteralPath $Root

$targets = @(
    @{ Name = "axios"; Versions = @("1.14.1", "0.30.4") },
    @{ Name = "plain-crypto-js"; Versions = @("4.2.1") }
)

$iocStrings = @(
    "sfrclak.com",
    "142.11.206.73",
    "f7d335205b8d7b20208fb3ef93ee6dc817905dc3ae0c10a0b164f4e7d07121cd",
    "617b67a8e1210e4fc87c92d1d1da45a2f311c08d26e89b12307cf583c900d101",
    "92ff08773995ebc8d55ec4b8e1a225d0d1e51efa4ef88b8849d0071230c9645a"
)

$findings = New-Object System.Collections.Generic.List[object]

Get-ChildItem -LiteralPath $resolvedRoot -Recurse -File -Filter package.json -ErrorAction SilentlyContinue | ForEach-Object {
    try {
        $packageJson = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json -AsHashtable
    } catch {
        return
    }

    foreach ($target in $targets) {
        if ($packageJson.name -eq $target.Name -and $target.Versions -contains $packageJson.version) {
            $findings.Add([pscustomobject]@{
                Type = "PackageVersion"
                Path = $_.FullName
                Detail = "$($target.Name)@$($packageJson.version)"
            })
        }
    }
}

$searchRoots = @(
    (Join-Path $resolvedRoot "node_modules"),
    (Join-Path $resolvedRoot "package-lock.json"),
    (Join-Path $resolvedRoot "npm-shrinkwrap.json"),
    (Join-Path $resolvedRoot "pnpm-lock.yaml"),
    (Join-Path $resolvedRoot "yarn.lock")
) | Where-Object { Test-Path -LiteralPath $_ }

foreach ($searchRoot in $searchRoots) {
    Get-ChildItem -LiteralPath $searchRoot -Recurse -File -ErrorAction SilentlyContinue | ForEach-Object {
        foreach ($ioc in $iocStrings) {
            $matches = Select-String -LiteralPath $_.FullName -SimpleMatch -Pattern $ioc -ErrorAction SilentlyContinue
            foreach ($match in $matches) {
                $findings.Add([pscustomobject]@{
                    Type = "IOC"
                    Path = $match.Path
                    Detail = "$ioc (line $($match.LineNumber))"
                })
            }
        }
    }
}

if ($findings.Count -eq 0) {
    Write-Output "No known matches for the targeted malicious package versions or published IOCs were found under $resolvedRoot."
    exit 0
}

$findings | Sort-Object Type, Path, Detail | Format-Table -AutoSize
exit 1
