from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    "image_scrapers": (),
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
