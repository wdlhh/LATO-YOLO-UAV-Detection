# Source baseline

The bundled framework source was prepared from the `ultralytics==8.4.92` Python package source tree and then modified locally to integrate LATO-YOLO.

Custom/modified integration points:

- `ultralytics/nn/modules/lato.py` — added
- `ultralytics/utils/lato_loss.py` — added
- `ultralytics/nn/modules/__init__.py` — exports custom modules
- `ultralytics/nn/tasks.py` — YAML parser hooks and HPL criterion selection
- `ultralytics/cfg/models/12/lato-yolo.yaml` — added

The original Ultralytics license is included as `LICENSE-ULTRALYTICS.txt`.
