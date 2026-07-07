import sys
import types

# Pytest imports the repo root __init__.py as a top-level test package before
# per-test ComfyUI stubs are installed; keep collection isolated from that file.
sys.modules.setdefault("__init__", types.ModuleType("__init__"))
