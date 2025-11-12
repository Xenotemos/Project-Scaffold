# Usage: `. .\scripts\llama_env.ps1` to export llama.cpp paths for the current PowerShell session.

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $projectRoot

$llamaServer = Join-Path $projectRoot "third_party\llama.cpp\build\bin\Release\llama-server.exe"
$llmModel = "D:\AI\LLMs\mistral-7b-instruct-v0.2.Q4_K_M.gguf"

$defaultVulkanRoot = "C:\VulkanSDK"
if (-not $env:VULKAN_SDK -and (Test-Path $defaultVulkanRoot)) {
    $latestSdk = Get-ChildItem $defaultVulkanRoot -Directory | Sort-Object Name -Descending | Select-Object -First 1
    if ($latestSdk) {
        $env:VULKAN_SDK = $latestSdk.FullName
    }
}
if ($env:VULKAN_SDK) {
    $vulkanBin = Join-Path $env:VULKAN_SDK "Bin"
    if (Test-Path $vulkanBin) {
        $pathParts = ($env:PATH -split ";")
        if ($pathParts -notcontains $vulkanBin) {
            $env:PATH = "$vulkanBin;$env:PATH"
        }
    }
}

if (-not (Test-Path $llamaServer)) {
    Write-Error "llama-server binary not found at $llamaServer. Run scripts\build_llama_cpp.py first."
    return
}

if (-not (Test-Path $llmModel)) {
    Write-Error "Model GGUF not found at $llmModel. Update scripts\llama_env.ps1 with your model path."
    return
}

$env:LLAMA_SERVER_BIN = $llamaServer
$env:LLAMA_MODEL_PATH = $llmModel
$env:LLAMA_SERVER_HOST = "127.0.0.1"
$env:LLAMA_SERVER_PORT = "8080"
$env:LLAMA_MODEL_ALIAS = "mistral-local"
$env:LLAMA_SERVER_ARGS = "--ctx-size 4096 --no-webui"
$env:LLAMA_SERVER_READY_TIMEOUT = "120"
$env:LIVING_LLM_TIMEOUT = "60"
$env:LLAMA_SERVER_TIMEOUT = "60"

Write-Host "[llama-env] Exported LLAMA_* variables for this session."
