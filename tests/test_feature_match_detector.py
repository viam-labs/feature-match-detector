"""Tests for feature-match-detector.

Run from the repository root:

    python -m unittest tests.test_feature_match_detector
"""

import asyncio
import gzip
import io
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from viam.media.video import CameraMimeType, ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.service.vision import Detection
from viam.utils import dict_to_struct

import src  # noqa: E402

fmd = sys.modules["src.featureMatchDetector"]
featureMatchDetector = fmd.featureMatchDetector
ORIGINAL_URLOPEN = fmd.urllib.request.urlopen


def checker_svg(phase: int = 0) -> bytes:
    cells = []
    for y in range(0, 400, 40):
        for x in range(0, 400, 40):
            if (x // 40 + y // 40 + phase) % 2 == 0:
                cells.append(f'<rect x="{x}" y="{y}" width="40" height="40" fill="black"/>')
    body = "".join(cells)
    return (
        '<?xml version="1.0"?>'
        '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400" viewBox="0 0 400 400">'
        '<rect width="400" height="400" fill="white"/>'
        f"{body}</svg>"
    ).encode()


BLANK_SVG = (
    b'<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">'
    b'<rect width="40" height="40" fill="white"/></svg>'
)


def config_for(**attrs) -> ComponentConfig:
    return ComponentConfig(name="fd", attributes=dict_to_struct(attrs))


def handler_for(directory: Path):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, fmt, *args):
            return

    return Handler


class FeatureMatchDetectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="fmd-test-"))
        cls.web = cls.root / "web"
        cls.web.mkdir()
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 255, (240, 320), dtype=np.uint8)
        cls.png_path = cls.root / "ref.png"
        Image.fromarray(noise).save(cls.png_path)
        cls.png_bytes = cls.png_path.read_bytes()
        cls.svg = checker_svg(0)
        cls.svg_v2 = checker_svg(1)
        cls.svg_path = cls.root / "ref.svg"
        cls.svg_path.write_bytes(cls.svg)
        cls.svgz_path = cls.root / "ref.svgz"
        cls.svgz_path.write_bytes(gzip.compress(cls.svg))
        (cls.web / "v1.svg").write_bytes(cls.svg)
        (cls.web / "v2.svg").write_bytes(cls.svg_v2)
        (cls.web / "blank.svg").write_bytes(BLANK_SVG)
        (cls.web / "ref.svg").write_bytes(cls.svg)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(cls.web))
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        raster = io.BytesIO()
        fmd._svg_bytes_to_pil(cls.svg).save(raster, format="PNG")
        cls.svg_png = raster.getvalue()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        fmd.urllib.request.urlopen = ORIGINAL_URLOPEN
        shutil.rmtree(cls.root)

    def setUp(self):
        self.cache = Path(tempfile.mkdtemp(prefix="fmd-cache-"))
        fmd.CACHE_DIR = self.cache
        self.downloads = 0

        def counting_urlopen(*args, **kwargs):
            self.downloads += 1
            return ORIGINAL_URLOPEN(*args, **kwargs)

        fmd.urllib.request.urlopen = counting_urlopen

    def tearDown(self):
        fmd.urllib.request.urlopen = ORIGINAL_URLOPEN
        shutil.rmtree(self.cache, ignore_errors=True)

    def url(self, name: str) -> str:
        return f"http://127.0.0.1:{self.port}/{name}"

    def cached_files(self):
        return [path for path in self.cache.iterdir() if path.is_file() and not path.name.startswith(".")]

    def test_validate_config(self):
        deps, optional = featureMatchDetector.validate_config(
            config_for(source_image_path=str(self.png_path), cameras=["cam-a", "cam-a", "cam-b"], min_good_matches=8)
        )
        self.assertEqual(list(deps), ["cam-a", "cam-b"])
        self.assertEqual(list(optional), [])

        featureMatchDetector.validate_config(config_for(source_image_path=self.png_path.as_uri()))
        featureMatchDetector.validate_config(config_for(source_image_path="https://example.com/ref.svg"))
        self.assertEqual(self.downloads, 0)

        with self.assertRaises(Exception):
            featureMatchDetector.validate_config(config_for())
        with self.assertRaises(Exception):
            featureMatchDetector.validate_config(config_for(source_image_path=str(self.root / "missing.jpg")))
        with self.assertRaises(Exception):
            featureMatchDetector.validate_config(config_for(source_image_path="ftp://example.com/a.jpg"))
        with self.assertRaises(Exception):
            featureMatchDetector.validate_config(config_for(source_image_path=str(self.png_path), cameras="cam"))

    def test_local_png_and_file_uri_match(self):
        detector = featureMatchDetector.new(
            config_for(source_image_path=str(self.png_path), cameras=["cam-a", "cam-b"], min_good_matches=8),
            {},
        )
        detections = asyncio.run(detector.get_detections(ViamImage(self.png_bytes, CameraMimeType.PNG)))
        self.assertEqual(len(detections), 1)
        self.assertIsInstance(detections[0], Detection)
        self.assertEqual(detections[0].class_name, "match")
        self.assertGreater(detections[0].confidence, 0)

        properties = asyncio.run(detector.get_properties())
        self.assertEqual(properties.default_camera, "cam-a")
        self.assertTrue(properties.detections_supported)
        self.assertFalse(properties.classifications_supported)

        with self.assertRaises(Exception) as raised:
            asyncio.run(detector.get_detections_from_camera("other-cam"))
        self.assertIn("other-cam", str(raised.exception))
        self.assertIn("cam-a", str(raised.exception))

        file_detector = featureMatchDetector.new(config_for(source_image_path=self.png_path.as_uri(), min_good_matches=8), {})
        file_detections = asyncio.run(file_detector.get_detections(ViamImage(self.png_bytes, CameraMimeType.PNG)))
        self.assertEqual(len(file_detections), 1)

    def test_svg_and_svgz_references(self):
        for source in (str(self.svg_path), str(self.svgz_path)):
            detector = featureMatchDetector.new(config_for(source_image_path=source, min_good_matches=6), {})
            detections = asyncio.run(detector.get_detections(ViamImage(self.svg_png, CameraMimeType.PNG)))
            self.assertEqual(len(detections), 1, source)

        detector = featureMatchDetector.new(config_for(source_image_path=str(self.svg_path), min_good_matches=6), {})
        svg_query = asyncio.run(detector.get_detections(ViamImage(self.svg, CameraMimeType.PNG)))
        self.assertEqual(len(svg_query), 1)

        blank = io.BytesIO()
        Image.new("RGB", (80, 80), "white").save(blank, format="PNG")
        self.assertEqual(asyncio.run(detector.get_detections(ViamImage(blank.getvalue(), CameraMimeType.PNG))), [])

    def test_remote_svg_is_cached_then_reused(self):
        uri = self.url("v1.svg")
        featureMatchDetector.new(config_for(source_image_path=uri, cameras=["cam"], min_good_matches=6), {})
        self.assertEqual(self.downloads, 1)
        cached = self.cached_files()
        self.assertEqual(len(cached), 1)
        self.assertEqual(cached[0].suffix, ".svg")
        self.assertEqual(cached[0].read_bytes(), self.svg)

        featureMatchDetector.new(config_for(source_image_path=uri, min_good_matches=6), {})
        self.assertEqual(self.downloads, 1)

    def test_refetch_reference_replaces_cache(self):
        uri = self.url("ref.svg")
        detector = featureMatchDetector.new(config_for(source_image_path=uri, min_good_matches=6), {})
        self.assertEqual(self.downloads, 1)
        original_features = detector.source_descriptors.tobytes()
        try:
            (self.web / "ref.svg").write_bytes(self.svg_v2)
            featureMatchDetector.new(config_for(source_image_path=uri, min_good_matches=6), {})
            self.assertEqual(self.downloads, 1)

            result = asyncio.run(detector.do_command({"refetch_reference": True}))
            self.assertEqual(result["source_image_path"], uri)
            self.assertEqual(self.downloads, 2)
            self.assertEqual(self.cached_files()[0].read_bytes(), self.svg_v2)
            self.assertNotEqual(detector.source_descriptors.tobytes(), original_features)
            self.assertEqual(detector.source_image_path, uri)

            (self.web / "ref.svg").write_bytes(BLANK_SVG)
            features_before_failure = detector.source_descriptors.tobytes()
            with self.assertRaises(Exception) as raised:
                asyncio.run(detector.do_command({"refetch_reference": True}))
            self.assertIn("enough features", str(raised.exception))
            self.assertEqual(self.cached_files()[0].read_bytes(), self.svg_v2)
            self.assertEqual(detector.source_descriptors.tobytes(), features_before_failure)
        finally:
            (self.web / "ref.svg").write_bytes(self.svg)

    def test_refetch_reference_can_switch_url(self):
        detector = featureMatchDetector.new(config_for(source_image_path=self.url("v1.svg"), min_good_matches=6), {})
        result = asyncio.run(detector.do_command({"refetch_reference": self.url("v2.svg")}))
        self.assertEqual(result["source_image_path"], self.url("v2.svg"))
        self.assertEqual(self.downloads, 2)
        cached = {path.name: path.read_bytes() for path in self.cached_files()}
        self.assertIn(self.svg, cached.values())
        self.assertIn(self.svg_v2, cached.values())
        self.assertEqual(urlparse(detector.source_image_path).path, "/v2.svg")

    def test_refetch_reference_rejects_local_files_and_download_errors(self):
        detector = featureMatchDetector.new(config_for(source_image_path=str(self.png_path), min_good_matches=8), {})
        with self.assertRaises(Exception) as raised:
            asyncio.run(detector.do_command({"refetch_reference": True}))
        self.assertIn("http:// or https://", str(raised.exception))
        self.assertEqual(detector.source_image_path, str(self.png_path))

        remote = featureMatchDetector.new(config_for(source_image_path=self.url("v1.svg"), min_good_matches=6), {})
        with self.assertRaises(Exception) as raised:
            asyncio.run(remote.do_command({"refetch_reference": self.url("missing.svg")}))
        self.assertIn("Failed to download", str(raised.exception))
        self.assertEqual(remote.source_image_path, self.url("v1.svg"))
        self.assertEqual(self.cached_files()[0].read_bytes(), self.svg)

    def test_do_command_updates_source_and_threshold(self):
        detector = featureMatchDetector.new(config_for(source_image_path=str(self.png_path), min_good_matches=8), {})
        asyncio.run(
            detector.do_command(
                {
                    "set": [
                        {"key": "source_image_path", "value": str(self.svg_path)},
                        {"key": "min_good_matches", "value": 6},
                    ]
                }
            )
        )
        self.assertEqual(detector.min_good_matches, 6)
        self.assertEqual(detector.source_image_path, str(self.svg_path))
        detections = asyncio.run(detector.get_detections(ViamImage(self.svg_png, CameraMimeType.PNG)))
        self.assertEqual(len(detections), 1)

    def test_store_reference_replaces_previous_suffix(self):
        uri = "https://example.com/reference"
        fmd._store_reference_bytes(uri, self.svg, "image/svg+xml")
        self.assertEqual([path.suffix for path in self.cached_files()], [".svg"])
        fmd._store_reference_bytes(uri, self.png_bytes, "image/png")
        cached = self.cached_files()
        self.assertEqual([path.suffix for path in cached], [".png"])
        self.assertEqual(cached[0].read_bytes(), self.png_bytes)


if __name__ == "__main__":
    unittest.main()
