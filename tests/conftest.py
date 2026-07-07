import sys
import types

sys.modules.setdefault("__init__", types.ModuleType("__init__"))
