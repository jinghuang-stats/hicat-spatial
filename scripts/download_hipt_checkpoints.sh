#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="${1:-checkpoints/hipt}"

SOURCE_256="https://upenn.box.com/shared/static/p0hc12l1bpu5c7fzieotv1d6592btv1l.pth"
SOURCE_4K="https://upenn.box.com/shared/static/8qayhxzmdjpcr5loi88xtkfbqomag8a9.pth"

TARGET_256="${CHECKPOINT_DIR}/vit256_small_dino.pth"
TARGET_4K="${CHECKPOINT_DIR}/vit4k_xs_dino.pth"

mkdir -p "${CHECKPOINT_DIR}"

echo "Downloading HIPT ViT-256 checkpoint..."
wget -c "${SOURCE_256}" -O "${TARGET_256}"

echo "Downloading HIPT ViT-4K checkpoint..."
wget -c "${SOURCE_4K}" -O "${TARGET_4K}"

echo "HIPT checkpoints saved to:"
echo "  ${TARGET_256}"
echo "  ${TARGET_4K}"