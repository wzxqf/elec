param(
    [string]$SourcePath = "",
    [string]$OutputPath = ""
)

trap {
    Write-Error $_
    exit 1
}

$ErrorActionPreference = "Stop"

function Invoke-ComMethod {
    param(
        [Parameter(Mandatory = $true)]$ComObject,
        [Parameter(Mandatory = $true)][string]$MethodName,
        [object[]]$Arguments = @()
    )

    return $ComObject.GetType().InvokeMember(
        $MethodName,
        [System.Reflection.BindingFlags]::InvokeMethod,
        $null,
        $ComObject,
        $Arguments
    )
}

function Get-DefaultSource {
    $items = Get-ChildItem -Path . -Recurse -File -Filter "*.docx" |
        Where-Object {
            $_.Name -notlike "~$*" -and
            $_.BaseName -notlike "*formula_rendered*" -and
            $_.BaseName -notlike "*公式渲染版*"
        } |
        Sort-Object LastWriteTime -Descending

    if ($items.Count -lt 1) {
        throw "No DOCX source file found."
    }
    return $items[0].FullName
}

if ([string]::IsNullOrWhiteSpace($SourcePath)) {
    $SourcePath = Get-DefaultSource
}

$sourceItem = Get-Item -LiteralPath $SourcePath
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $sourceItem.DirectoryName ($sourceItem.BaseName + "_formula_rendered.docx")
}

Copy-Item -LiteralPath $sourceItem.FullName -Destination $OutputPath -Force

$wdAlignParagraphCenter = 1
$word = $null
$doc = $null

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0

    $doc = $word.Documents.Open((Resolve-Path -LiteralPath $OutputPath).Path)
    $convertedDisplay = 0
    $convertedInline = 0
    $removedMarkers = 0
    $buildUpFailures = 0

    for ($i = $doc.Paragraphs.Count - 2; $i -ge 1; $i--) {
        $openText = $doc.Paragraphs.Item($i).Range.Text.Trim()
        if ($openText -eq "\[") {
            $formulaPara = $doc.Paragraphs.Item($i + 1)
            $closePara = $doc.Paragraphs.Item($i + 2)
            $closeText = $closePara.Range.Text.Trim()
            if ($closeText -eq "\]") {
                $formulaRange = $formulaPara.Range
                if ($formulaRange.End -gt $formulaRange.Start) {
                    $formulaRange.End = $formulaRange.End - 1
                }
                $formulaPara.Range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
                $math = $doc.OMaths.Add($formulaRange)
                try {
                    Invoke-ComMethod -ComObject $math -MethodName "BuildUp" | Out-Null
                }
                catch {
                    $buildUpFailures += 1
                }
                $convertedDisplay += 1

                $closePara.Range.Delete() | Out-Null
                $doc.Paragraphs.Item($i).Range.Delete() | Out-Null
                $removedMarkers += 2
            }
        }
    }

    for ($p = 1; $p -le $doc.Paragraphs.Count; $p++) {
        $para = $doc.Paragraphs.Item($p)
        $text = $para.Range.Text
        $matches = [regex]::Matches($text, "\\\((.+?)\\\)")
        for ($m = $matches.Count - 1; $m -ge 0; $m--) {
            $match = $matches.Item($m)
            $expr = $match.Groups.Item(1).Value
            if ([string]::IsNullOrWhiteSpace($expr)) {
                continue
            }

            $start = $para.Range.Start + $match.Index
            $end = $para.Range.Start + $match.Index + $match.Length
            $range = $doc.Range($start, $end)
            $range.Text = $expr
            $math = $doc.OMaths.Add($range)
            try {
                Invoke-ComMethod -ComObject $math -MethodName "BuildUp" | Out-Null
            }
            catch {
                $buildUpFailures += 1
            }
            $convertedInline += 1
        }
    }

    $doc.Save()
    $doc.Close($true)
    $doc = $null

    [pscustomobject]@{
        source = $sourceItem.FullName
        output = (Resolve-Path -LiteralPath $OutputPath).Path
        converted_display = $convertedDisplay
        converted_inline = $convertedInline
        removed_marker_paragraphs = $removedMarkers
        build_up_failures = $buildUpFailures
    } | ConvertTo-Json -Compress
}
finally {
    if ($doc -ne $null) {
        try { $doc.Close($false) | Out-Null } catch {}
    }
    if ($word -ne $null) {
        try { $word.Quit() | Out-Null } catch {}
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    }
}
