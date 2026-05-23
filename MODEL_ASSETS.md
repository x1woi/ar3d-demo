# Model And Asset Notes

This repository includes only small demo assets that are practical for GitHub review.

## Included assets

`runtime_assets/competition_demo_models_hq/` contains curated GLB files for browser demonstration:

- `sunglasses_hq.glb`
- `water_bottle_hq.glb`
- `boombox_hq.glb`
- `antique_camera_hq.glb`
- `damaged_helmet_hq.glb`

These files are used to demonstrate model loading, AR-style placement, and cache reuse behavior. Source and license notes are kept beside the GLB files.

The lightweight MediaPipe hand landmarker file `runtime_assets/hand_landmarker.task` is included for gesture-related local demo compatibility. If distribution policy changes, replace it with an official download step.

## Not included

The following are intentionally not included:

- Qwen model weights or API tokens
- TripoSR model weights
- Stable Fast 3D weights
- Hunyuan3D weights
- Hugging Face tokens
- Local virtual environments
- Temporary videos, raw captures, and full experiment intermediate data

## Why

The project contribution is the AR interaction flow and the text-image fusion cache reuse strategy. Large 3D generation models are treated as replaceable generation backends and are not required for the lightweight GitHub demo.

