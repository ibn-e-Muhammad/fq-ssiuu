# FQ-SSIUU Overlay Locks

## Lock Strategy

* **Do Not Lock Full Environment:** A full `pip freeze` lockfile must not be generated or used across environments because the Kaggle-native compute image is separately fingerprinted and maintained by Kaggle.
* **Stage 1I Overlay Lock:** Following successful installation and verification in Kaggle during Stage 1I, an overlay-only lockfile (`kaggle-overlay.lock`) will be generated.
* **Non-Mutating Verification:** Before installing any overlay, `verify_pip_report.py` validates the pip dry-run report against `requirements/kaggle-native-baseline.json` to ensure zero changes to hard-protected or native-stability packages.
