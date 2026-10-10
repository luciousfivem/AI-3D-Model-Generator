# Puts a "Forge3D" shortcut on your desktop.
$root = Split-Path -Parent $PSScriptRoot
$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Desktop")) "Forge3D.lnk"))
$link.TargetPath = Join-Path $root "scripts\Forge3D.bat"
$link.WorkingDirectory = $root
$link.WindowStyle = 7   # minimised: the launcher closes immediately
$link.Description = "Forge3D - AI models for 3D printing"
$link.Save()
Write-Host "Shortcut created on your desktop."
