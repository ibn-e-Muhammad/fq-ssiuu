# Google Drive Artifact Storage — Setup Guide

This guide describes how to configure Google Drive for FQ-SSIUU experiment
artifact upload.

## Overview

All experiment artifacts (checkpoints, evaluation results, provenance records)
are uploaded to a shared Google Drive folder after each run. This provides:
- Off-device backup (critical for Kaggle ephemeral storage)
- Shared access for collaborators
- Immutable artifact archive

---

## Step 1: Create a GCP Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the **Google Drive API**:
   - Navigate to APIs & Services > Library
   - Search for "Google Drive API"
   - Click Enable

## Step 2: Create a Service Account

1. Go to IAM & Admin > Service Accounts
2. Click "Create Service Account"
3. Name: `fq-ssiuu-drive-upload`
4. Click "Create and Continue"
5. Skip the optional steps, click "Done"
6. Click on the new service account
7. Go to Keys tab > Add Key > Create New Key > JSON
8. Download the JSON key file
9. **Store securely — never commit this file**

## Step 3: Create and Share a Drive Folder

1. In Google Drive, create a folder: `FQ-SSIUU-Artifacts`
2. Right-click > Share
3. Add the service account email (from step 2, e.g. `fq-ssiuu-drive-upload@project.iam.gserviceaccount.com`)
4. Give **Editor** permission
5. Copy the folder ID from the URL:
   `https://drive.google.com/drive/folders/FOLDER_ID_HERE`

## Step 4: Configure Credentials

### For Local Development

Add to your `.env` file:
```
GDRIVE_SERVICE_ACCOUNT_JSON=/path/to/service-account.json
GDRIVE_FOLDER_ID=your_folder_id_here
```

### For Kaggle

1. Go to your Kaggle notebook
2. Click "Add-ons" > "Secrets"
3. Add two secrets:
   - Name: `GDRIVE_SERVICE_ACCOUNT_JSON`, Value: *paste the entire JSON content*
   - Name: `GDRIVE_FOLDER_ID`, Value: *paste the folder ID*

## Step 5: Verify

```python
from fq_ssiuu.drive_upload import upload_run_directory

# Create a test directory
import tempfile, os
with tempfile.TemporaryDirectory() as d:
    with open(os.path.join(d, "test.txt"), "w") as f:
        f.write("FQ-SSIUU upload test")
    result = upload_run_directory(d, run_id="test-upload")
    print(result)
```

If successful, you'll see the test file in your Drive folder.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError` | Install: `pip install google-api-python-client google-auth` |
| `RuntimeError: No credentials` | Check GDRIVE_SERVICE_ACCOUNT_JSON is set |
| `HttpError 404` | Verify the folder ID and that the service account has access |
| `HttpError 403` | The service account needs Editor permission on the folder |
