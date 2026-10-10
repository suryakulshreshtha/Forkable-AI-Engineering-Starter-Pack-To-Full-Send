"""Optional pytest support. fullsend.py needs none of this; it is here for people who like pytest.

    pytest levels/L1-numbers-that-learn            # judge your drill.py files
    FULLSEND_TARGET=reference pytest               # judge the reference answers instead

Each test_drill.py function takes one argument, `m`: the module under test.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_counter = 0


@pytest.fixture
def m(request):
    global _counter
    folder = Path(str(getattr(request, "path", None) or request.fspath)).parent
    name = "reference.py" if os.environ.get("FULLSEND_TARGET") == "reference" else "drill.py"
    target = folder / name
    if not target.exists():
        pytest.skip(f"{target.relative_to(ROOT)} is missing; run python fullsend.py init")
    _counter += 1
    spec = importlib.util.spec_from_file_location(f"_pytest_fullsend_{_counter}", str(target))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
