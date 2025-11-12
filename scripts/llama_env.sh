# Usage: source scripts/llama_env.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_SERVER_BIN="$PROJECT_ROOT/third_party/llama.cpp/build/bin/Release/llama-server"
LLAMA_MODEL_PATH="/mnt/d/AI/LLMs/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Best-effort Vulkan SDK discovery (mirrors Windows env setup).
if [ -z "${VULKAN_SDK:-}" ] && [ -d "/mnt/c/VulkanSDK" ]; then
  latest_sdk="$(ls -1 /mnt/c/VulkanSDK 2>/dev/null | sort -r | head -n 1)"
  if [ -n "$latest_sdk" ] && [ -d "/mnt/c/VulkanSDK/$latest_sdk" ]; then
    export VULKAN_SDK="/mnt/c/VulkanSDK/$latest_sdk"
  fi
fi
if [ -n "${VULKAN_SDK:-}" ] && [ -d "$VULKAN_SDK/Bin" ]; then
  case ":$PATH:" in
    *":$VULKAN_SDK/Bin:"*) ;;
    *) export PATH="$VULKAN_SDK/Bin:$PATH" ;;
  esac
fi

if [ ! -f "$LLAMA_SERVER_BIN" ]; then
  echo "llama-server binary not found at $LLAMA_SERVER_BIN. Build it via scripts/build_llama_cpp.py first."
  return 1 2>/dev/null || exit 1
fi

if [ ! -f "$LLAMA_MODEL_PATH" ]; then
  echo "Model GGUF not found at $LLAMA_MODEL_PATH. Update scripts/llama_env.sh with your model path."
  return 1 2>/dev/null || exit 1
fi

export LLAMA_SERVER_BIN
export LLAMA_MODEL_PATH
export LLAMA_SERVER_HOST="127.0.0.1"
export LLAMA_SERVER_PORT="8080"
export LLAMA_MODEL_ALIAS="mistral-local"
export LLAMA_SERVER_ARGS="--ctx-size 4096 --no-webui"
export LLAMA_SERVER_READY_TIMEOUT="120"
export LIVING_LLM_TIMEOUT="60"
export LLAMA_SERVER_TIMEOUT="60"

echo "[llama-env] Exported LLAMA_* variables for this shell."
