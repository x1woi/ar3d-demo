# Upload To GitHub

Target repository:

[https://github.com/x1woi/ar3d-demo](https://github.com/x1woi/ar3d-demo)

This folder is already cleaned for upload. It contains code, demo assets, README files, and experiment summaries. It does not contain virtual environments, raw videos, large generation weights, or private tokens.

## Recommended: use Git command line

Open PowerShell in this folder:

```powershell
cd D:\tool3\py1\pythonProject6\paper_repro_outputs\competition_delivery\github_repo_upload\ar3d-demo
```

Then run:

```powershell
git init
git branch -M main
git add .
git commit -m "Initial release for AR 3D demo"
git remote add origin https://github.com/x1woi/ar3d-demo.git
git push -u origin main
```

If Git asks for login, use GitHub browser login or a GitHub personal access token. Do not paste tokens into project files.

## Alternative: GitHub web upload

1. Open [https://github.com/x1woi/ar3d-demo](https://github.com/x1woi/ar3d-demo).
2. Click `Add file` -> `Upload files`.
3. Drag the contents of this folder into the upload area.
4. Do not upload the local `__pycache__` folder if it is visible.
5. Commit changes.

Command-line upload is safer because `.gitignore` automatically excludes cache files.

## After upload

Check that GitHub shows:

- `README.md`
- `plus.py`
- `demo_static_server.py`
- `cache_policy_loader.py`
- `runtime_assets/cloud_realtime_demo.html`
- `runtime_assets/competition_demo_models_hq/*.glb`
- `docs/experiments/`
- `docs/competition/competition_overview.pdf`

Also check that the repository does not contain:

- `.venv/`
- `.venv_public/`
- `.venv_sf3d/`
- Hugging Face token
- API key
- raw videos
- Stable Fast 3D / Hunyuan3D weights
