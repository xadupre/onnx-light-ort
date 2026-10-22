"""Runs a build-tree Python script with the native test DLL search paths."""

from contextlib import ExitStack
import os
from pathlib import Path
import runpy
import sys


def main():
    script = Path(sys.argv[1]).resolve(strict=True)
    sys.argv = sys.argv[1:]
    sys.path[0] = str(script.parent)
    with ExitStack() as directories:
        if sys.platform == "win32":
            # Python 3.8+ ignores PATH when resolving extension-module DLLs.
            for path in dict.fromkeys(os.get_exec_path()):
                if os.path.isabs(path) and os.path.isdir(path):
                    directories.enter_context(os.add_dll_directory(path))
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
