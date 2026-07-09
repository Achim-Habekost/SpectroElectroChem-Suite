# GitHub release guide

1. Create a new GitHub repository, for example:
   `SpectroElectroChem-Suite`

2. Upload all files in this folder.

3. Replace the placeholder `YOUR_GITHUB_NAME` in:
   - `src/spectroelectrochem_suite/updater.py`
   - `CITATION.cff`

4. Commit and push.

5. Create a release tag:
   ```bash
   git tag v2.0.0
   git push origin v2.0.0
   ```

6. GitHub Actions will build the Windows application automatically.

7. Connect the GitHub repository with Zenodo to obtain a DOI.
