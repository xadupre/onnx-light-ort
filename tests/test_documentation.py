"""Checks the project diagram, documentation destinations and custom-op examples."""

from html.parser import HTMLParser
from pathlib import Path
import runpy
import tomllib
import unittest
import xml.etree.ElementTree as ET

DOCS = Path(__file__).resolve().parents[1] / "docs"
SVG = "{http://www.w3.org/2000/svg}"
PROJECTS = {
    "bridge_ops_light_called_by_ort": ("30", "95"),
    "bridge_ops_ort_called_by_light": ("490", "95"),
    "bridge_ep_light_called_by_ort": ("30", "240"),
    "bridge_ep_ort_called_by_light": ("490", "240"),
}


class HtmlElements(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.elements = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


class TestDocumentation(unittest.TestCase):
    def test_homepage_version(self):
        config = runpy.run_path(str(DOCS / "conf.py"))
        with (DOCS.parent / "pyproject.toml").open("rb") as stream:
            version = tomllib.load(stream)["project"]["version"]
        self.assertEqual(config["release"], version)
        homepage = (DOCS / "index.rst").read_text(encoding="utf-8")
        self.assertIn("Version: |release|", homepage)

    def test_svg_zones(self):
        diagram = DOCS / "_static" / "projects.svg"
        links = ET.parse(diagram).getroot().findall(f"{SVG}a")
        self.assertEqual(len(links), 4)
        self.assertEqual({Path(link.attrib["href"]).stem for link in links}, set(PROJECTS))
        for link in links:
            with self.subTest(href=link.attrib["href"]):
                page = diagram.parent / link.attrib["href"]
                self.assertTrue(page.with_suffix(".rst").is_file())
                self.assertEqual(link.attrib["target"], "_top")
                self.assertTrue(link.attrib["aria-label"])
                rect = link.find(f"{SVG}rect")
                self.assertIsNotNone(rect)
                self.assertEqual((rect.attrib["x"], rect.attrib["y"]), PROJECTS[page.stem])
                self.assertEqual((rect.attrib["width"], rect.attrib["height"]), ("440", "125"))

    def test_interactive_embedding_and_fallback(self):
        elements = HtmlElements((DOCS / "index.rst").read_text(encoding="utf-8")).elements
        objects = [attrs for tag, attrs in elements if tag == "object"]
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["type"], "image/svg+xml")
        self.assertEqual(objects[0]["data"], "_static/projects.svg")
        self.assertTrue(objects[0]["aria-label"])
        self.assertEqual(
            {attrs["href"] for tag, attrs in elements if tag == "a"},
            {f"components/{name}.html" for name in PROJECTS},
        )

    def test_four_project_pages(self):
        components = DOCS / "components"
        pages = {page.stem for page in components.glob("*.rst")} - {"index"}
        self.assertEqual(pages, set(PROJECTS))
        navigation = (components / "index.rst").read_text(encoding="utf-8")
        for name in PROJECTS:
            with self.subTest(project=name):
                self.assertIn(f"   {name}\n", navigation)
                source = (components / f"{name}.rst").read_text(encoding="utf-8")
                self.assertIn("\nDesign\n------\n", source)
                self.assertIn("\nRequirements\n------------\n", source)
                for line in source.splitlines():
                    if line.startswith(".. include:: "):
                        source += (components / line.removeprefix(".. include:: ")).read_text(
                            encoding="utf-8"
                        )
                self.assertRegex(
                    source, r":language: (python|cpp)|\.\. code-block:: (python|cpp)"
                )

    def test_custom_op_pages(self):
        components = DOCS / "components"
        for name in ("bridge_ops_light_called_by_ort", "bridge_ops_ort_called_by_light"):
            with self.subTest(project=name):
                source = (components / f"{name}.rst").read_text(encoding="utf-8")
                self.assertIn("\nC ABI contract\n--------------\n", source)
                for symbol in (
                    "RegisterCustomOps",
                    "GetApi",
                    "CreateKernelV2",
                    "KernelComputeV2",
                    "KernelDestroy",
                ):
                    self.assertIn(f"``{symbol}``", source)
                for line in source.splitlines():
                    if line.startswith(".. include:: "):
                        source += (components / line.removeprefix(".. include:: ")).read_text(
                            encoding="utf-8"
                        )
                for language in ("python", "cpp"):
                    self.assertRegex(
                        source, rf":language: {language}|\.\. code-block:: {language}\n"
                    )
                examples = []
                for line in source.splitlines():
                    if line.startswith(".. literalinclude:: "):
                        example = components / line.removeprefix(".. literalinclude:: ")
                        self.assertTrue(example.is_file(), str(example))
                        self.assertTrue(example.read_text(encoding="utf-8").strip())
                        examples.append(example)
                self.assertTrue(any(example.suffix == ".py" for example in examples))
                self.assertTrue(any(example.suffix in (".cc", ".cpp") for example in examples))


if __name__ == "__main__":
    unittest.main()
