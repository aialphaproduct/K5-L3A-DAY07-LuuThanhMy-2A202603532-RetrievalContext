"""Nạp package `src` của Day 6 dưới tên `day6src` để không trùng với code Day 7 và không sửa repo Day 6."""
from __future__ import annotations

import importlib
import importlib.util
import sys

from .config import DAY6_STARTER


def load():
    if "day6src" in sys.modules:
        return sys.modules["day6src"]
    init = DAY6_STARTER / "src" / "__init__.py"
    spec = importlib.util.spec_from_file_location("day6src", init,
                                                  submodule_search_locations=[str(DAY6_STARTER / "src")])
    module = importlib.util.module_from_spec(spec)
    sys.modules["day6src"] = module
    spec.loader.exec_module(module)
    return module


def submodule(name: str):
    load()
    return importlib.import_module(f"day6src.{name}")
