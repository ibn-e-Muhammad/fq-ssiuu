"""
src/fq_ssiuu/drive_upload.py

Google Drive artifact upload sink.

Uploads experiment run directories to a shared Google Drive folder using
service account authentication. The upload status is recorded separately
from experiment results — upload failures never corrupt experiment data.

Authentication sources (in order):
    1. GDRIVE_SERVICE_ACCOUNT_JSON env var (path to service account JSON)
    2. Kaggle Secrets (key: 'GDRIVE_SERVICE_ACCOUNT_JSON')

Requires:
    pip install google-api-python-client google-auth

Usage:
    from fq_ssiuu.drive_upload import upload_run_directory
    upload_run_directory("/path/to/run/output", folder_id="your_drive_folder_id")
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

# Default retry settings
MAX_RETRIES = 3
BACKOFF_BASE = 2.0  # seconds


def _get_drive_service():
    """Authenticate and return a Google Drive API service object.

    Returns:
        Google Drive API service resource.

    Raises:
        ImportError: If google-api-python-client is not installed.
        RuntimeError: If no credentials can be found.
    """
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError:
        raise ImportError(
            "Google Drive upload requires google-api-python-client and google-auth. "
            "Install with: pip install google-api-python-client google-auth"
        )

    # Try to get service account JSON path
    sa_json_path = None
    try:
        from fq_ssiuu.secrets import get_secret

        sa_json_path = get_secret("GDRIVE_SERVICE_ACCOUNT_JSON", required=False)
    except Exception:
        sa_json_path = os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON")

    if not sa_json_path:
        raise RuntimeError(
            "No Google Drive credentials found. Set GDRIVE_SERVICE_ACCOUNT_JSON "
            "environment variable to the path of your service account JSON file. "
            "See handbook/DRIVE_SETUP_GUIDE.md for setup instructions."
        )

    if os.path.isfile(sa_json_path):
        creds = service_account.Credentials.from_service_account_file(
            sa_json_path,
            scopes=["https://www.googleapis.com/auth/drive.file"],
        )
    else:
        # Try parsing as inline JSON
        try:
            sa_info = json.loads(sa_json_path)
            creds = service_account.Credentials.from_service_account_info(
                sa_info,
                scopes=["https://www.googleapis.com/auth/drive.file"],
            )
        except (json.JSONDecodeError, ValueError):
            raise RuntimeError(
                "GDRIVE_SERVICE_ACCOUNT_JSON is neither a valid file path nor valid inline JSON."
            )

    service = build("drive", "v3", credentials=creds)
    logger.info("Google Drive service authenticated successfully.")
    return service


def _upload_file_with_retry(
    service,
    local_path: str,
    parent_folder_id: str,
    max_retries: int = MAX_RETRIES,
) -> dict:
    """Upload a single file to Google Drive with exponential backoff.

    Args:
        service: Google Drive API service.
        local_path: Path to the local file.
        parent_folder_id: Google Drive folder ID.
        max_retries: Maximum number of retry attempts.

    Returns:
        Dict with 'id', 'name', 'size' of the uploaded file.

    Raises:
        RuntimeError: If all retry attempts fail.
    """
    from googleapiclient.http import MediaFileUpload

    file_name = os.path.basename(local_path)
    file_size = os.path.getsize(local_path)

    file_metadata = {
        "name": file_name,
        "parents": [parent_folder_id],
    }

    media = MediaFileUpload(local_path, resumable=True)

    for attempt in range(max_retries + 1):
        try:
            file = (
                service.files()
                .create(
                    body=file_metadata,
                    media_body=media,
                    fields="id,name,size",
                )
                .execute()
            )

            uploaded_size = int(file.get("size", 0))
            if uploaded_size != file_size:
                logger.warning(
                    "Size mismatch for '%s': local=%d, uploaded=%d",
                    file_name,
                    file_size,
                    uploaded_size,
                )

            logger.info(
                "Uploaded '%s' (%d bytes) -> Drive ID: %s",
                file_name,
                file_size,
                file["id"],
            )
            return file

        except Exception as e:
            if attempt < max_retries:
                wait = BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    "Upload attempt %d/%d failed for '%s': %s. Retrying in %.1f seconds.",
                    attempt + 1,
                    max_retries + 1,
                    file_name,
                    str(e),
                    wait,
                )
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Failed to upload '{file_name}' after {max_retries + 1} attempts: {e}"
                )


def _create_drive_folder(
    service,
    folder_name: str,
    parent_folder_id: str,
) -> str:
    """Create a folder in Google Drive.

    Args:
        service: Google Drive API service.
        folder_name: Name for the new folder.
        parent_folder_id: Parent folder ID.

    Returns:
        The ID of the created folder.
    """
    file_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }
    folder = (
        service.files()
        .create(
            body=file_metadata,
            fields="id",
        )
        .execute()
    )
    logger.info("Created Drive folder '%s' (ID: %s)", folder_name, folder["id"])
    return folder["id"]


def upload_run_directory(
    local_dir: str,
    folder_id: str | None = None,
    run_id: str | None = None,
) -> dict:
    """Upload an experiment run directory to Google Drive.

    Creates a subfolder named after the run_id (or directory basename)
    and uploads all files recursively.

    Args:
        local_dir: Path to the local run directory.
        folder_id: Google Drive folder ID. If None, reads from
            GDRIVE_FOLDER_ID env var or Kaggle Secret.
        run_id: Optional run ID for the subfolder name.

    Returns:
        Dict with upload summary: folder_id, file_count, total_bytes, errors.
    """
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(f"Run directory not found: {local_dir}")

    if not folder_id:
        try:
            from fq_ssiuu.secrets import get_secret

            folder_id = get_secret("GDRIVE_FOLDER_ID", required=True)
        except Exception:
            folder_id = os.environ.get("GDRIVE_FOLDER_ID")

    if not folder_id:
        raise RuntimeError(
            "No Google Drive folder ID found. Set GDRIVE_FOLDER_ID "
            "environment variable or Kaggle Secret."
        )

    service = _get_drive_service()

    # Create run subfolder
    subfolder_name = run_id or os.path.basename(local_dir)
    drive_folder_id = _create_drive_folder(service, subfolder_name, folder_id)

    # Upload all files
    uploaded_files: list[dict] = []
    errors: list[str] = []
    total_bytes = 0

    for root, dirs, files in os.walk(local_dir):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            try:
                result = _upload_file_with_retry(service, fpath, drive_folder_id)
                uploaded_files.append(result)
                total_bytes += os.path.getsize(fpath)
            except Exception as e:
                error_msg = f"Failed to upload {fpath}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

    summary = {
        "drive_folder_id": drive_folder_id,
        "drive_folder_name": subfolder_name,
        "file_count": len(uploaded_files),
        "total_bytes": total_bytes,
        "error_count": len(errors),
        "errors": errors,
    }

    logger.info(
        "Upload complete: %d files (%d bytes), %d errors",
        summary["file_count"],
        summary["total_bytes"],
        summary["error_count"],
    )

    return summary
