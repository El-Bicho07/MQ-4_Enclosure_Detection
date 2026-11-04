#!/usr/bin/env bash
# ==============================================================================
# Script: capture_mq4_webcam_screenshots.sh
# Description: Automated full-screen screenshot capture tool during live
#              MQ-4 enclosure YOLOv8n webcam detection on Raspberry Pi 5.
# ==============================================================================

set -Eeuo pipefail

# Configurable parameters via environment variables with safe defaults
SCREENSHOT_COUNT="${SCREENSHOT_COUNT:-8}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-10}"
INITIAL_WAIT_SECONDS="${INITIAL_WAIT_SECONDS:-8}"

# Resolve project directories relative to script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
cd "${PROJECT_ROOT}"

VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python3"
MODEL_CHECKPOINT="${PROJECT_ROOT}/models/best.pt"
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
OUTPUT_DIR="${PROJECT_ROOT}/evidence/deployment/webcam_screen_captures/${TIMESTAMP}"

# ------------------------------------------------------------------------------
# Utility: Detect supported screenshot tool
# ------------------------------------------------------------------------------
detect_screenshot_tool() {
    # Check for Wayland utilities first if running under Wayland
    if [[ -n "${WAYLAND_DISPLAY:-}" ]] || [[ "${XDG_SESSION_TYPE:-}" == "wayland" ]]; then
        if command -v grim &>/dev/null; then
            echo "grim"
            return 0
        elif command -v gnome-screenshot &>/dev/null; then
            echo "gnome-screenshot"
            return 0
        fi
    fi

    # Check for X11 / standard Linux screenshot utilities
    if command -v scrot &>/dev/null; then
        echo "scrot"
        return 0
    elif command -v gnome-screenshot &>/dev/null; then
        echo "gnome-screenshot"
        return 0
    elif command -v import &>/dev/null; then
        echo "import"
        return 0
    elif command -v grim &>/dev/null; then
        echo "grim"
        return 0
    fi

    return 1
}

# ------------------------------------------------------------------------------
# Utility: Capture a single full-screen screenshot
# ------------------------------------------------------------------------------
capture_screenshot() {
    local target_path="$1"
    local tool="$2"

    case "${tool}" in
        grim)
            grim "${target_path}"
            ;;
        gnome-screenshot)
            gnome-screenshot -f "${target_path}"
            ;;
        scrot)
            scrot "${target_path}"
            ;;
        import)
            import -window root "${target_path}"
            ;;
        *)
            echo "Error: Unsupported capture tool '${tool}'" >&2
            return 1
            ;;
    esac
}

# ------------------------------------------------------------------------------
# Pre-flight Checks
# ------------------------------------------------------------------------------
echo "=================================================================="
echo "    MQ-4 WEBCAM SCREENSHOT AUTOMATION FOR EVIDENCE RECORDING"
echo "=================================================================="
echo "Platform:       Raspberry Pi 5 (ARM Cortex-A76, 4 GB RAM)"
echo "Camera Target:  Logitech HD Webcam C270 (/dev/video0)"
echo "Model Checkpoint: ${MODEL_CHECKPOINT}"
echo "Screenshot Tool: Detecting..."

if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "Error: Virtual environment python not found at '${VENV_PYTHON}'." >&2
    echo "Please initialize the environment: python3 -m venv --system-site-packages .venv" >&2
    exit 1
fi

if [[ ! -f "${MODEL_CHECKPOINT}" ]]; then
    echo "Error: Model checkpoint not found at '${MODEL_CHECKPOINT}'." >&2
    exit 1
fi

if [[ -z "${DISPLAY:-}" ]] && [[ -z "${WAYLAND_DISPLAY:-}" ]]; then
    echo "Warning: Neither \$DISPLAY nor \$WAYLAND_DISPLAY is set." >&2
    echo "Full-screen screenshots require an active desktop session (X11 or Wayland)." >&2
fi

SCREENSHOT_TOOL="$(detect_screenshot_tool || true)"
if [[ -z "${SCREENSHOT_TOOL}" ]]; then
    echo ""
    echo "Error: No supported screenshot utility detected!" >&2
    echo "To capture full-screen screenshots on this system, please install a compatible tool:" >&2
    if [[ -n "${WAYLAND_DISPLAY:-}" ]] || [[ "${XDG_SESSION_TYPE:-}" == "wayland" ]]; then
        echo "  - For Wayland: sudo apt-get update && sudo apt-get install -y grim" >&2
    else
        echo "  - For X11:     sudo apt-get update && sudo apt-get install -y scrot" >&2
    fi
    echo "  - Alternative: sudo apt-get install -y gnome-screenshot" >&2
    exit 1
fi

echo "Selected Tool:  ${SCREENSHOT_TOOL}"
echo "Count:          ${SCREENSHOT_COUNT} captures"
echo "Interval:       ${INTERVAL_SECONDS} seconds between captures"
echo "Initial Wait:   ${INITIAL_WAIT_SECONDS} seconds for window initialization"
echo "Output Directory: ${OUTPUT_DIR}"
echo "=================================================================="
echo ""

# Ensure evidence destination directory exists
mkdir -p "${OUTPUT_DIR}"

# ------------------------------------------------------------------------------
# Process Management & Safe Signal Handling
# ------------------------------------------------------------------------------
WEBCAM_PID=""

cleanup() {
    local exit_code=$?
    echo ""
    if [[ -n "${WEBCAM_PID}" ]] && kill -0 "${WEBCAM_PID}" 2>/dev/null; then
        echo "[Cleanup] Stopping webcam detection process (PID: ${WEBCAM_PID})..."
        # Send SIGINT first to allow OpenCV and Python to release camera and windows cleanly
        kill -INT "${WEBCAM_PID}" 2>/dev/null || kill -TERM "${WEBCAM_PID}" 2>/dev/null
        
        # Wait up to 5 seconds for clean termination
        local wait_count=0
        while kill -0 "${WEBCAM_PID}" 2>/dev/null && [[ ${wait_count} -lt 5 ]]; do
            sleep 1
            ((wait_count++))
        done
        
        # Force kill if still lingering
        if kill -0 "${WEBCAM_PID}" 2>/dev/null; then
            echo "[Cleanup] Forcing shutdown with SIGKILL..."
            kill -KILL "${WEBCAM_PID}" 2>/dev/null || true
        fi
        echo "[Cleanup] Webcam process terminated cleanly."
    fi

    if [[ ${exit_code} -eq 0 ]]; then
        echo "=================================================================="
        echo "SUCCESS: Screenshot automation sequence completed."
        echo "Evidence directory: ${OUTPUT_DIR}"
        echo "=================================================================="
    else
        echo "=================================================================="
        echo "NOTICE: Script finished with exit code ${exit_code}."
        echo "Screenshots preserved in: ${OUTPUT_DIR}"
        echo "=================================================================="
    fi
    exit "${exit_code}"
}

trap cleanup EXIT INT TERM

# ------------------------------------------------------------------------------
# 1. Launch Webcam Detection Application
# ------------------------------------------------------------------------------
echo "[1/3] Launching MQ-4 webcam detection pipeline..."
"${VENV_PYTHON}" src/camera.py --device 0 --imgsz 320 --conf 0.25 &
WEBCAM_PID=$!
echo "      Webcam detection process running with PID: ${WEBCAM_PID}"

# ------------------------------------------------------------------------------
# 2. Wait for Window Initialization
# ------------------------------------------------------------------------------
echo "[2/3] Waiting ${INITIAL_WAIT_SECONDS} seconds for camera and GUI window to initialize..."
for ((s = INITIAL_WAIT_SECONDS; s > 0; s--)); do
    if ! kill -0 "${WEBCAM_PID}" 2>/dev/null; then
        echo "Error: Webcam detection application exited unexpectedly during startup." >&2
        exit 1
    fi
    sleep 1
done
echo "      Webcam detection window should now be visible on screen."
echo "      (Note: Keep the detection window visible and do NOT minimize it)"
echo ""

# ------------------------------------------------------------------------------
# 3. Automated Screenshot Capture Loop
# ------------------------------------------------------------------------------
echo "[3/3] Starting automatic screenshot capture sequence..."
CAPTURED_COUNT=0

for ((i = 1; i <= SCREENSHOT_COUNT; i++)); do
    # Check that webcam process is alive prior to capture
    if ! kill -0 "${WEBCAM_PID}" 2>/dev/null; then
        echo "Error: Webcam process is no longer running (PID ${WEBCAM_PID} terminated). Aborting captures." >&2
        exit 1
    fi

    FILENAME="$(printf "screen_%02d.png" "${i}")"
    TARGET_FILE="${OUTPUT_DIR}/${FILENAME}"

    echo " -> [Capture ${i}/${SCREENSHOT_COUNT}] Capturing full-screen: ${FILENAME}..."
    if ! capture_screenshot "${TARGET_FILE}" "${SCREENSHOT_TOOL}"; then
        echo "Error: Failed to execute screenshot command with '${SCREENSHOT_TOOL}'." >&2
        exit 1
    fi

    # Verify that the screenshot was actually created and is non-empty
    if [[ ! -s "${TARGET_FILE}" ]]; then
        echo "Error: Captured file '${TARGET_FILE}' is missing or empty (0 bytes)." >&2
        exit 1
    fi

    FILE_SIZE="$(du -h "${TARGET_FILE}" | cut -f1)"
    echo "    Saved successfully (${FILE_SIZE}) at $(date +'%H:%M:%S')"
    ((CAPTURED_COUNT++))

    # Wait interval between screenshots if not the final capture
    if [[ ${i} -lt ${SCREENSHOT_COUNT} ]]; then
        echo "    Waiting ${INTERVAL_SECONDS} seconds until next screenshot..."
        for ((w = INTERVAL_SECONDS; w > 0; w--)); do
            if ! kill -0 "${WEBCAM_PID}" 2>/dev/null; then
                echo "Error: Webcam process exited prematurely during wait interval." >&2
                exit 1
            fi
            sleep 1
        done
    fi
done

echo ""
echo "All ${CAPTURED_COUNT} screenshots captured successfully."

