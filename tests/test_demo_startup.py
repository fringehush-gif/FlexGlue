"""Exercise the notebook's checkout guard without IPython or shell effects."""

import json
import os
import tempfile
import unittest
from pathlib import Path


class DemoStartupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = Path(__file__).resolve().parents[1] / "demo.ipynb"
        cells = json.loads(notebook.read_text())["cells"]
        startup = next(cell for cell in cells if cell["cell_type"] == "code")
        lines = []
        for line in startup["source"]:
            if line.strip() == "import torch":
                break
            if line.lstrip().startswith(("!", "%")):
                indent = line[: len(line) - len(line.lstrip())]
                line = f"{indent}calls.append({line.strip()!r})\n"
            lines.append(line)
        cls.startup = compile("".join(lines), str(notebook), "exec")

    def run_startup(self, dirname, checkout):
        calls = []
        with tempfile.TemporaryDirectory() as folder:
            cwd = Path(folder) / dirname
            cwd.mkdir()
            if checkout:
                (cwd / "flexglue").mkdir()
                (cwd / "flexglue" / "__init__.py").touch()
            previous = Path.cwd()
            try:
                os.chdir(cwd)
                exec(self.startup, {"calls": calls})
            finally:
                os.chdir(previous)
        return calls

    def assert_bootstrap(self, calls):
        self.assertEqual(len(calls), 3)
        self.assertTrue(calls[0].startswith("!git clone "))
        self.assertEqual(calls[1], "%cd FlexGlue")
        self.assertTrue(calls[2].startswith("%pip install "))

    def test_existing_checkout_can_have_another_directory_name(self):
        for dirname in ("FlexGlue", "flexglue-work"):
            with self.subTest(dirname=dirname):
                self.assertEqual(self.run_startup(dirname, True), [])

    def test_matching_directory_name_without_package_bootstraps(self):
        self.assert_bootstrap(self.run_startup("FlexGlue", False))

    def test_fresh_colab_directory_bootstraps(self):
        self.assert_bootstrap(self.run_startup("content", False))


if __name__ == "__main__":
    unittest.main()
