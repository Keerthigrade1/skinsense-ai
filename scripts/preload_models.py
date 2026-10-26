"""Download all models ahead of time (used by the Docker build).

Run: python scripts/preload_models.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skinsense import nlp  # noqa: E402

failed = []
for key, spec in nlp.MODELS.items():
    m = nlp.load(key)
    print(f"{'ok  ' if m.ok else 'FAIL'} {spec.model_id}" + ("" if m.ok else f"  ({m.error})"))
    if not m.ok:
        failed.append(spec.model_id)
sys.exit(1 if failed else 0)
