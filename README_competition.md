# Competition README

## Project

造物π：基于手势姿态估计与图文双相似度缓存复用的 AR 3D 生成系统。

## Demo Link

[https://zaowupi-ar3d-demo.serveousercontent.com](https://zaowupi-ar3d-demo.serveousercontent.com)

If the public tunnel is unavailable, run the lightweight local demo:

```powershell
.\.venv\Scripts\python.exe demo_static_server.py
```

Then open:

```text
http://127.0.0.1:5000
```

For the full local pipeline, run:

```powershell
$env:PYTHONIOENCODING="utf-8"
$env:ENABLE_POLICY_CACHE="1"
$env:CACHE_POLICY_PATH="runtime_assets/cache_policy.json"
.\.venv\Scripts\python.exe plus.py
```

## Demo Flow

1. Open the camera page.
2. Allow browser camera permission.
3. Use hand gesture or ROI area to select an object.
4. Load a GLB model in the browser.
5. Show cache policy behavior:
   - high-confidence `auto_hit`
   - low-confidence candidate prompt
   - safe `miss` fallback

## Key Result

The project validates that reusing cached GLB assets can reduce repeated 3D generation waiting time. In a single-sample speed validation, the auto-hit cache path skipped TripoSR and reduced latency by about 47.83 seconds compared with the baseline generation path.

## Notes

This GitHub version is a clean review package. It does not include virtual environments, large model weights, raw videos, private tokens, or full intermediate experiment data.
