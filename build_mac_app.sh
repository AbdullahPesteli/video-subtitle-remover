#!/bin/bash

# macOS Native App Builder for Subtitle Remover
# Creates a standalone SubtitleRemover.app with PyInstaller

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"
APP_NAME="SubtitleRemover"
MAIN_SCRIPT="${SCRIPT_DIR}/gui.py"
ICON_FILE="${SCRIPT_DIR}/resource/icon.svg"  # Fallback to default if exists
BUILD_DIR="${SCRIPT_DIR}/build"
DIST_DIR="${SCRIPT_DIR}/dist"

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

# Clean previous builds
rm -rf "$BUILD_DIR" "$DIST_DIR" "build" "dist" "${APP_NAME}.spec"

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
    --add-data="${SCRIPT_DIR}/backend:backend" \
    --add-data="${SCRIPT_DIR}/ui:ui" \
    --add-data="${SCRIPT_DIR}/design:design" \
    --hidden-import=PySide6 \
    --hidden-import=qfluentwidgets \
    --hidden-import=paddleocr \
    --hidden-import=paddlex \
    --hidden-import=cv2 \
    --hidden-import=torch \
    --hidden-import=torchvision \
    --hidden-import=PIL \
    --collect-all=paddleocr \
    --collect-all=paddlex \
    --collect-all=modelscope \
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
