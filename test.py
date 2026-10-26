"""Live Rs.50 banknote detection from a webcam.

    python test.py --model model.onnx

Thin entry point: all logic lives in src/detector/app.py so it can be unit-tested.
Needs only the packages in requirements.txt (no PyTorch, no server).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    try:
        from detector.app import main
    except ImportError as exc:  # pragma: no cover - friendly message for a missing dependency
        sys.exit(f"error: {exc}. Activate your virtual environment and run: pip install -r requirements.txt")
    sys.exit(main())
