#!/usr/bin/env bash
# scripts/pull_from_drive.sh
# Pulls checkpoints from Google Drive to local checkpoints/ directory.

set -euo pipefail

GDRIVE_REMOTE="gdrive:fq-ssiuu-research"
LOCAL_CHECKPOINTS="./checkpoints"

echo "Pulling checkpoints from Google Drive..."
rclone sync "$GDRIVE_REMOTE/checkpoints" "$LOCAL_CHECKPOINTS" --progress || {
    echo "rclone not configured. See scripts/sync_to_drive.sh for setup instructions."
}
