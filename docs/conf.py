import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def setup(app):
    build = ROOT / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(ROOT),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython3_EXECUTABLE={sys.executable}",
            "-DONNX_LIGHT_ORT_BUILD_PYTHON=ON",
            "-DONNX_LIGHT_ORT_BUILD_HOST=ON",
            "-DONNX_LIGHT_ORT_BUILD_PLUGIN=ON",
            "-DONNX_LIGHT_ORT_BUILD_EXAMPLES=ON",
        ],
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build),
            "--config",
            "Release",
            "--parallel",
            str(os.cpu_count() or 1),
            "--target",
            "onnx_light_ort_python",
            "onnx_light_ort_custom_ops",
            "onnx_light_ort_bridge_ops_ort_called_by_light",
            "onnx_light_ort_ort_custom_ops",
        ],
        check=True,
    )
    with (build / "Release" / "gallery_environment.json").open(encoding="utf-8") as stream:
        environment = json.load(stream)
    python_dir = environment.pop("ONNX_LIGHT_ORT_MODEL_PYTHON_DIR")
    sys.path.insert(0, python_dir)
    # Gallery workers must use the same model bindings as the Sphinx process.
    python_path = [str(ROOT), python_dir]
    if os.environ.get("PYTHONPATH"):
        python_path.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(python_path)
    if sys.platform == "win32":
        dll_directories = dict.fromkeys(
            [str(Path(python_dir) / "onnx_light" / "onnx_py"), *os.get_exec_path()]
        )
        app._onnx_light_dll_directories = [
            os.add_dll_directory(path)
            for path in dll_directories
            if os.path.isabs(path) and os.path.isdir(path)
        ]
    for name, path in environment.items():
        os.environ.setdefault(name, path)


project = "onnx-light-ort"
author = "onnx-light-ort contributors"
extensions = ["sphinx_gallery.gen_gallery"]
sphinx_gallery_conf = {
    "examples_dirs": "examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"/plot_",
    "abort_on_example_error": True,
    "run_stale_examples": True,
    "download_all_examples": False,
    "image_scrapers": ("matplotlib",),
    "remove_config_comments": True,
}
html_theme = "pydata_sphinx_theme"
html_title = project
html_logo = "_static/onnx-light-ort.svg"
html_theme_options = {
    "github_url": "https://github.com/xadupre/onnx-light-ort",
    "show_prev_next": False,
}
exclude_patterns = ["_build", "examples/README.rst"]
nitpicky = True
