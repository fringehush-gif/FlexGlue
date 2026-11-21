import hashlib
import importlib.util
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch


class ONNXDownloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("onnxruntime") is None:
            raise unittest.SkipTest("optional dependency onnxruntime is unavailable")
        spec = importlib.util.spec_from_file_location(
            "flexglue_onnx_match_pair",
            Path(__file__).resolve().parents[1] / "onnx" / "match_pair.py",
        )
        cls.match_pair = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.match_pair)

    def test_concurrent_downloads_publish_verified_model(self):
        name, payload = "model.onnx", b"verified model fixture"
        written, hashed = threading.Barrier(2), threading.Barrier(2)
        checksum = self.match_pair.sha256sum

        def download(url, destination):
            Path(destination).write_bytes(payload)
            written.wait(timeout=5)

        def checked_hash(path):
            result = checksum(path)
            hashed.wait(timeout=5)
            return result

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            with (
                patch.dict(
                    self.match_pair.MODELS, {name: hashlib.sha256(payload).hexdigest()}
                ),
                patch.object(self.match_pair.urllib.request, "urlretrieve", download),
                patch.object(self.match_pair, "sha256sum", checked_hash),
                ThreadPoolExecutor(max_workers=2) as pool,
            ):
                results = list(
                    pool.map(lambda _: self.match_pair.get_model(name, cache), (0, 1))
                )
            self.assertEqual(results, [cache / name] * 2)
            self.assertEqual((cache / name).read_bytes(), payload)
            self.assertEqual(list(cache.iterdir()), [cache / name])

    def test_failed_download_does_not_leave_files(self):
        for transport_failure in (False, True):
            with self.subTest(transport_failure=transport_failure):

                def download(url, destination):
                    Path(destination).write_bytes(b"incomplete model")
                    if transport_failure:
                        raise OSError("fixture download interrupted")

                with tempfile.TemporaryDirectory() as directory:
                    cache = Path(directory)
                    name = next(iter(self.match_pair.MODELS))
                    with patch.object(
                        self.match_pair.urllib.request, "urlretrieve", download
                    ):
                        with self.assertRaises(
                            OSError if transport_failure else RuntimeError
                        ):
                            self.match_pair.get_model(name, cache)
                    self.assertEqual(list(cache.iterdir()), [])

    def test_existing_custom_model_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            cache, payload = Path(directory), b"custom exported model"
            path = cache / next(iter(self.match_pair.MODELS))
            path.write_bytes(payload)
            with patch.object(
                self.match_pair.urllib.request, "urlretrieve"
            ) as download:
                self.assertEqual(self.match_pair.get_model(path.name, cache), path)
            download.assert_not_called()
            self.assertEqual(path.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
