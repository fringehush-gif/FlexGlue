"""Headless plotting contracts used by the public notebook."""

import ast
import json
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from flexglue.utils import pack_matches, rbd, viz2d  # noqa: E402


class PlottingTest(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_empty_matches_leave_images_and_axes_unchanged(self):
        for empty in (np.empty((0, 2)), torch.empty(0, 2, requires_grad=True)):
            for lw, ps in ((0, 0), (1.5, 4)):
                with self.subTest(tensor=isinstance(empty, torch.Tensor), lw=lw):
                    viz2d.plot_images([np.zeros((8, 12, 3)), np.ones((6, 10, 3))])
                    fig = plt.gcf()
                    axes = tuple(fig.axes)
                    limits = [(ax.get_xlim(), ax.get_ylim()) for ax in axes]
                    for color in (None, "red", np.empty((0, 4))):
                        viz2d.plot_matches(empty, empty, color=color, lw=lw, ps=ps)
                    self.assertEqual(len(fig.artists), 0)
                    self.assertTrue(all(len(ax.collections) == 0 for ax in axes))
                    self.assertEqual(
                        [(ax.get_xlim(), ax.get_ylim()) for ax in axes], limits
                    )
                    self.assertTrue(all(len(ax.images) == 1 for ax in axes))
                    with self.assertRaisesRegex(ValueError, "Got 0 and 1"):
                        viz2d.plot_matches(empty, np.ones((1, 2)), axes=axes)
                    fig.canvas.draw()
                    plt.close(fig)

    def test_notebook_can_display_zero_matches(self):
        notebook = json.loads(
            (Path(__file__).resolve().parents[1] / "demo.ipynb").read_text(
                encoding="utf-8"
            )
        )
        helpers = [
            node
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            if "def plot_matches(" in "".join(cell["source"])
            for node in ast.parse("".join(cell["source"])).body
            if isinstance(node, ast.FunctionDef) and node.name == "plot_matches"
        ]
        self.assertEqual(len(helpers), 1)
        namespace = {
            "torch": torch,
            "viz2d": viz2d,
            "rbd": rbd,
            "pack_matches": pack_matches,
            "image0": torch.zeros(3, 8, 12),
            "image1": torch.ones(3, 6, 10),
        }
        code = ast.Module(body=helpers, type_ignores=[])
        exec(compile(code, "demo.ipynb", "exec"), namespace)
        pred = {
            "matches0": torch.full((1, 3), -1, dtype=torch.int64),
            "matching_scores0": torch.zeros(1, 3),
        }
        namespace["plot_matches"](
            pred, torch.zeros(3, 2), torch.zeros(4, 2), "generated", lw=0
        )
        fig = plt.gcf()
        self.assertEqual(fig.axes[0].texts[0].get_text(), "generated: 0 matches")
        self.assertEqual(len(fig.artists), 0)
        fig.canvas.draw()

    def test_images_preserve_pixels_and_gradient_graph(self):
        leaf = torch.linspace(0, 1, 3 * 5 * 7).reshape(3, 5, 7).requires_grad_()
        image = (leaf * 0.75).transpose(1, 2)
        mono = leaf[0]
        array = np.linspace(0, 1, 5 * 7 * 3).reshape(5, 7, 3)[:, ::-1]
        array.setflags(write=False)
        before = leaf.detach().clone()
        expected = [
            image.detach().permute(1, 2, 0).numpy(),
            mono.detach().numpy(),
            array.copy(),
        ]
        viz2d.plot_images([image, mono, array])
        for ax, pixels in zip(plt.gcf().axes, expected):
            np.testing.assert_array_equal(ax.images[0].get_array(), pixels)
            self.assertEqual(
                ax.images[0].get_extent(), [0, pixels.shape[1], pixels.shape[0], 0]
            )
        self.assertIsNone(leaf.grad)
        torch.testing.assert_close(leaf.detach(), before, rtol=0, atol=0)
        image.sum().backward()
        torch.testing.assert_close(
            leaf.grad, torch.full_like(leaf, 0.75), rtol=0, atol=0
        )
        self.assertFalse(array.flags.writeable)
        plt.gcf().canvas.draw()

    def test_nonempty_matches_use_the_requested_figure(self):
        fig, axes = plt.subplots(1, 2)
        unrelated = plt.figure()
        points0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        points1 = points0 + [2.0, 1.0]
        before0, before1 = points0.copy(), points1.copy()
        viz2d.plot_matches(points0, points1, axes=tuple(axes), lw=2, ps=9)
        self.assertIs(plt.gcf(), unrelated)
        self.assertEqual(len(unrelated.artists), 0)
        self.assertEqual(len(fig.artists), 2)
        for i, line in enumerate(fig.artists):
            self.assertIs(line.figure, fig)
            np.testing.assert_array_equal(line.xy1, points0[i])
            np.testing.assert_array_equal(line.xy2, points1[i])
            self.assertEqual(line.get_linewidth(), 2)
        np.testing.assert_array_equal(fig.artists[0].get_edgecolor(), [1, 1, 0, 1])
        np.testing.assert_array_equal(fig.artists[1].get_edgecolor(), [0, 0, 1, 1])
        for ax, points in zip(axes, (points0, points1)):
            np.testing.assert_array_equal(ax.collections[0].get_offsets(), points)
            np.testing.assert_array_equal(ax.collections[0].get_sizes(), [9])
        np.testing.assert_array_equal(points0, before0)
        np.testing.assert_array_equal(points1, before1)
        fig.canvas.draw()


if __name__ == "__main__":
    unittest.main()
