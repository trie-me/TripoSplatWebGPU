import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1] / "src" / "result_bundle.py"
    spec = importlib.util.spec_from_file_location("result_bundle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeGaussian:
    def __init__(self, count):
        self.count = count

    def save_ply(self, path):
        Path(path).write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex " + str(self.count).encode("ascii") + b"\nend_header\n")

    def save_splat(self, path):
        Path(path).write_bytes(b"x" * (self.count * 32))


class ResultBundleTests(unittest.TestCase):
    def test_bundle_has_exact_browser_contract_and_store_mode_members(self):
        bundle = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = bundle.build_result_bundle(directory, FakeGaussian(32), 32, 7, "sdk", {"backend": "cuda"})
            self.assertEqual(manifest["format"], "triposplat-webgpu-result")
            self.assertEqual(manifest["count"], 32)
            with zipfile.ZipFile(directory / "result.zip") as archive:
                self.assertEqual(archive.namelist(), ["scene.ply", "scene.splat", "manifest.json"])
                self.assertEqual(archive.getinfo("scene.ply").compress_type, zipfile.ZIP_STORED)
                self.assertEqual(json.loads(archive.read("manifest.json"))["version"], 1)

    def test_bundle_rejects_wrong_splat_record_count(self):
        bundle = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "scene.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 32\nend_header\n")
            (directory / "scene.splat").write_bytes(b"bad")
            with self.assertRaisesRegex(RuntimeError, "32 bytes"):
                bundle.validate_scene_files(directory / "scene.ply", directory / "scene.splat", 32)


if __name__ == "__main__":
    unittest.main()

