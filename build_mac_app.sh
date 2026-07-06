#!/bin/bash

# macOS Native App Builder for Subtitle Remover
# Creates a standalone SubtitleRemover.app with PyInstaller

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VSR_VENV_DIR:-${SCRIPT_DIR}/venv}"
APP_NAME="SubtitleRemover"
MAIN_SCRIPT="${SCRIPT_DIR}/gui.py"
ICON_FILE="${SCRIPT_DIR}/resource/icon.svg"  # Fallback to default if exists
BUILD_DIR="${SCRIPT_DIR}/build"
DIST_DIR="${SCRIPT_DIR}/dist"
STAGE_DIR="${SCRIPT_DIR}/.context/pyinstaller-data"
STAGE_BACKEND="${STAGE_DIR}/backend"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔨 Building macOS app for ${APP_NAME}...${NC}"

# Activate venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: venv not found at $VENV_DIR"
    exit 1
fi

source "${VENV_DIR}/bin/activate"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export MPLCONFIGDIR="${HOME}/Library/Caches/SubtitleRemover/matplotlib"
mkdir -p "$MPLCONFIGDIR"

# Clean previous builds
rm -rf "$BUILD_DIR" "$DIST_DIR" "build" "dist" "${APP_NAME}.spec" "$STAGE_DIR"
mkdir -p "$STAGE_BACKEND/ffmpeg" "$STAGE_DIR/design"

echo -e "${BLUE}🧩 Preparing minimal bundle data...${NC}"
python - <<'PY'
from backend.tools.model_config import ModelConfig

model_config = ModelConfig()
model_config.ensure_lama_model()
model_config.ensure_propainter_model()
PY
rsync -a --exclude='__pycache__' "${SCRIPT_DIR}/backend/interface/" "${STAGE_BACKEND}/interface/"
rsync -a --exclude='__pycache__' "${SCRIPT_DIR}/backend/ffmpeg/macos/" "${STAGE_BACKEND}/ffmpeg/macos/"
rsync -a \
    --exclude='__pycache__' \
    --exclude='big-lama_[0-9].pt' \
    --exclude='ProPainter_[0-9].pth' \
    "${SCRIPT_DIR}/backend/models/" "${STAGE_BACKEND}/models/"
cp "${SCRIPT_DIR}/design/vsr.ico" "${STAGE_DIR}/design/vsr.ico"

# Build with PyInstaller
echo -e "${BLUE}📦 Running PyInstaller...${NC}"

# Determine if icon exists
ICON_FLAG=""
if [ -f "$ICON_FILE" ]; then
    ICON_FLAG="--icon=$ICON_FILE"
fi

# PyInstaller command for macOS
pyinstaller \
    --name="${APP_NAME}" \
    --onedir \
    --windowed \
    --add-data="${STAGE_BACKEND}:backend" \
    --add-data="${SCRIPT_DIR}/ui:ui" \
    --add-data="${STAGE_DIR}/design:design" \
    --hidden-import=PySide6 \
    --hidden-import=qfluentwidgets \
    --hidden-import=paddleocr \
    --hidden-import=paddlex \
    --hidden-import=cv2 \
    --hidden-import=torch \
    --hidden-import=torchvision \
    --hidden-import=backend.inpaint.propainter_inpaint \
    --hidden-import=scipy._cyutility \
    --hidden-import=pypdfium2 \
    --hidden-import=pypdfium2_raw \
    --hidden-import=pyclipper \
    --hidden-import=PIL \
    --collect-all=paddleocr \
    --collect-all=paddlex \
    --collect-all=modelscope \
    --collect-all=torch \
    --collect-all=torchvision \
    --collect-all=pypdfium2 \
    --collect-all=pypdfium2_raw \
    --collect-all=pyclipper \
    --copy-metadata=opencv-python \
    --copy-metadata=opencv-contrib-python \
    --copy-metadata=pypdfium2 \
    --copy-metadata=pyclipper \
    $ICON_FLAG \
    --osx-bundle-identifier="com.todoipictures.subtitleremover" \
    --codesign-identity="-" \
    "$MAIN_SCRIPT"

# Move app to Applications folder (optional)
APP_PATH="${DIST_DIR}/${APP_NAME}.app"
if [ -d "$APP_PATH" ]; then
    echo -e "${GREEN}✅ App created: ${APP_PATH}${NC}"
    echo -e "${BLUE}📍 Launch with: open '${APP_PATH}'${NC}"
    echo -e "${BLUE}📍 Or copy to /Applications: cp -r '${APP_PATH}' /Applications/${NC}"

    # Verify it's executable
    if [ -x "${APP_PATH}/Contents/MacOS/${APP_NAME}" ]; then
        echo -e "${GREEN}✅ Binary is executable${NC}"
    fi
else
    echo "❌ App build failed"
    exit 1
fi

echo -e "${GREEN}✨ Build complete!${NC}"
