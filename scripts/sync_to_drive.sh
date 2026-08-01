#!/usr/bin/env bash
# scripts/sync_to_drive.sh
# Syncs local results and logs to Google Drive.
# Run after each experiment to preserve outputs.
# Requires: rclone configured with Google Drive remote named 'gdrive'

set -euo pipefail

GDRIVE_REMOTE="gdrive:fq-ssiuu-research"
LOCAL_RESULTS="./results"
LOCAL_LOGS="./logs"

echo "Syncing results to Google Drive..."
rclone sync "$LOCAL_RESULTS" "$GDRIVE_REMOTE/results" --progress || {
    echo "rclone not configured. Manual sync instructions:"
    echo "  1. Install rclone: https://rclone.org/install/"
    echo "  2. Configure: rclone config (add Google Drive remote named 'gdrive')"
    echo "  3. Or manually upload from Google Drive web interface."
}
