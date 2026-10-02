import gzip
import hashlib
import io
import time
import urllib.request
from urllib.error import HTTPError
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar, List, Mapping, Optional, Sequence, Tuple, cast
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import cv2
import numpy as np
import pymupdf
from PIL import Image
from typing_extensions import Self

from viam.components.camera import Camera
from viam.logging import getLogger
from viam.media.utils.pil import viam_to_pil_image
from viam.media.video import CameraMimeType, ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.proto.service.vision import Detection, GetPropertiesResponse
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily
from viam.services.vision import CaptureAllResult, Vision
from viam.utils import ValueTypes, struct_to_dict

DETECTOR = cv2.SIFT_create()
MATCHER = cv2.BFMatcher(cv2.NORM_L2)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
LOGGER = getLogger(__name__)
LOWE_RATIO = 0.7
MIN_INLIER_RATIO = 0.4
MIN_HOMOGRAPHY_POINTS = 4
MIN_TRACKED_FRACTION = 0.3
TRACK_BACKTRACK_ERROR = 3.0
GRACE_FRAMES = 2
DEFAULT_DETECTION_HOLD_SECONDS = 5.0

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
REMOTE_SCHEMES = {"http", "https"}
RASTER_MIME_TYPES = (CameraMimeType.JPEG, CameraMimeType.PNG, CameraMimeType.VIAM_RGBA)
CONTENT_TYPE_SUFFIX = {
    "image/svg+xml": ".svg",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}
KNOWN_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".svg", ".svgz"}


class featureMatchDetector(Vision):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "detector"), "feature-match-detector")

    def __init__(self, name: str):
        super().__init__(name)
        self.source_image_path = ""
        self.cameras: List[str] = []
        self.min_good_matches = 15
        self.detection_hold_seconds = DEFAULT_DETECTION_HOLD_SECONDS
        self.dependencies: Mapping[ResourceName, ResourceBase] = {}
        self.source_keypoints = None
        self.source_descriptors = None
        self.source_shape: Optional[Tuple[int, int]] = None
        self._camera_grace: dict = {}
        self._held_detections: dict = {}
        self._clock = time.monotonic
        self._source_lock = Lock()

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        detector = cls(config.name)
        detector.reconfigure(config, dependencies)
        return detector

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = _attributes(config)
        _validate_source(str(attrs.get("source_image_path", "")))
        _min_good_matches(attrs.get("min_good_matches"))
        _detection_hold_seconds(attrs.get("detection_hold_seconds"))
        return _configured_cameras(attrs.get("cameras")), []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        attrs = _attributes(config)
        self.cameras = list(_configured_cameras(attrs.get("cameras")))
        self.min_good_matches = _min_good_matches(attrs.get("min_good_matches"))
        self.detection_hold_seconds = _detection_hold_seconds(attrs.get("detection_hold_seconds"))
        self.dependencies = dependencies
        self._load_reference(str(attrs.get("source_image_path", "")))

    async def get_cam_image(self, camera_name: str) -> ViamImage:
        if camera_name not in self.cameras:
            configured = ", ".join(self.cameras) if self.cameras else "(none)"
            raise Exception(
                f"Camera '{camera_name}' is not in the configured cameras array. Configured cameras: {configured}"
            )
        resource_name = Camera.get_resource_name(camera_name)
        if resource_name not in self.dependencies:
            raise Exception(f"Camera '{camera_name}' is not available as a dependency")
        cam = cast(Camera, self.dependencies[resource_name])
        images, _ = await cam.get_images()
        if not images:
            raise Exception(f"Camera '{camera_name}' returned no images")
        for mime_type in RASTER_MIME_TYPES:
            for img in images:
                if img.mime_type == mime_type:
                    return img
        raise Exception(f"Camera '{camera_name}' did not return a JPEG, PNG, or RGBA image")

    def _load_reference(self, source: str, *, refresh: bool = False):
        pending_cache = None
        if refresh:
            _require_remote_source(source)
            data, content_type = _fetch_reference(source)
            gray = _bytes_to_gray(data, _suffix_for_image(source, content_type, data))
            pending_cache = (data, content_type)
            path = None
        else:
            path = resolve_reference_image(source)
            gray = _path_to_gray(path)
        keypoints, descriptors = DETECTOR.detectAndCompute(gray, None)
        if descriptors is not None:
            descriptors = np.asarray(descriptors, dtype=np.float32)
        if keypoints is None or descriptors is None or len(descriptors) < 2:
            raise Exception(f"Reference image '{source}' does not contain enough features to match")
        if pending_cache is not None:
            path = _store_reference_bytes(source, pending_cache[0], pending_cache[1])
        with self._source_lock:
            self.source_image_path = source
            self.source_keypoints = keypoints
            self.source_descriptors = descriptors
            self.source_shape = gray.shape[:2]
            self._camera_grace = {}
            self._held_detections = {}
        LOGGER.info("Loaded reference image from %s (%s features)", path, len(keypoints))

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        return self._match_image(await self.get_cam_image(camera_name), camera_name)

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        return self._match_image(image)

    def _match_image(self, image: ViamImage, camera_name: Optional[str] = None) -> List[Detection]:
        with self._source_lock:
            source_keypoints = self.source_keypoints
            source_descriptors = self.source_descriptors
            source_shape = self.source_shape
            minimum = self.min_good_matches
            grace = self._camera_grace.get(camera_name, 0) if camera_name else 0
        if source_keypoints is None or source_descriptors is None or source_shape is None:
            return []

        query_gray = _viam_image_to_gray(image)
        keypoints, descriptors = DETECTOR.detectAndCompute(query_gray, None)
        if keypoints is not None and descriptors is not None and len(descriptors) >= 2:
            descriptors = np.asarray(descriptors, dtype=np.float32)
            matches = MATCHER.knnMatch(source_descriptors, descriptors, 2)
            good = []
            for pair in matches:
                if len(pair) < 2:
                    continue
                first, second = pair
                if first.distance < LOWE_RATIO * second.distance:
                    good.append(first)
            if len(good) >= MIN_HOMOGRAPHY_POINTS:
                src_pts = np.float32([source_keypoints[match.queryIdx].pt for match in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints[match.trainIdx].pt for match in good]).reshape(-1, 1, 2)
                homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, None, 5000, 0.999)
                if homography is not None and mask is not None and np.isfinite(homography).all():
                    inlier_mask = mask.ravel().astype(bool)
                    inliers = int(inlier_mask.sum())
                    accepted, strict = _match_decision(inliers, len(good), minimum, grace)
                    box = _box_from_homography(homography, source_shape, query_gray.shape[:2]) if accepted else None
                    if box is not None:
                        detection = _detection_from_box(box, inliers)
                        return self._remember_match(
                            camera_name,
                            detection,
                            strict,
                            src_pts[inlier_mask].copy(),
                            dst_pts[inlier_mask].copy(),
                            query_gray,
                        )
        return self._follow_hold(camera_name, query_gray, source_shape)

    def _remember_match(
        self,
        camera_name: Optional[str],
        detection: Detection,
        strict: bool,
        reference_points: np.ndarray,
        scene_points: np.ndarray,
        query_gray: np.ndarray,
    ) -> List[Detection]:
        if camera_name is None:
            return [detection]
        now = self._clock()
        with self._source_lock:
            if strict:
                self._camera_grace[camera_name] = GRACE_FRAMES
            else:
                remaining = self._camera_grace.get(camera_name, 0) - 1
                if remaining > 0:
                    self._camera_grace[camera_name] = remaining
                else:
                    self._camera_grace.pop(camera_name, None)
            self._held_detections[camera_name] = {
                "expires_at": now + self.detection_hold_seconds,
                "reference_points": reference_points,
                "scene_points": scene_points,
                "previous_gray": query_gray.copy(),
                "anchor_count": len(scene_points),
            }
        return [detection]

    def _follow_hold(
        self, camera_name: Optional[str], query_gray: np.ndarray, source_shape: Optional[Tuple[int, int]]
    ) -> List[Detection]:
        if camera_name is None or self.detection_hold_seconds <= 0 or source_shape is None:
            return []
        now = self._clock()
        with self._source_lock:
            held = self._held_detections.get(camera_name)
            if held is None or now >= held["expires_at"]:
                self._held_detections.pop(camera_name, None)
                self._camera_grace.pop(camera_name, None)
                return []
            previous_gray = held["previous_gray"]
            scene_points = held["scene_points"]
            reference_points = held["reference_points"]
            anchor_count = held["anchor_count"]
        tracked = _track_held_points(previous_gray, query_gray, reference_points, scene_points, anchor_count, source_shape)
        if tracked is None:
            with self._source_lock:
                self._held_detections.pop(camera_name, None)
                self._camera_grace.pop(camera_name, None)
            return []
        detection, reference_points, scene_points = tracked
        with self._source_lock:
            current = self._held_detections.get(camera_name)
            if current is None or now >= current["expires_at"]:
                self._held_detections.pop(camera_name, None)
                return []
            current["reference_points"] = reference_points
            current["scene_points"] = scene_points
            current["previous_gray"] = query_gray.copy()
            self._camera_grace.pop(camera_name, None)
        return [detection]

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        LOGGER.info(command)
        if "set" in command.keys():
            for item in command["set"]:
                if "key" not in item.keys():
                    continue
                if item["key"] == "source_image_path":
                    self._load_reference(str(item["value"]))
                if item["key"] == "min_good_matches":
                    self.min_good_matches = _min_good_matches(item["value"])
                if item["key"] == "detection_hold_seconds":
                    self.detection_hold_seconds = _detection_hold_seconds(item["value"])
        response: dict = {"response": "OK", "timestamp": str(datetime.now())}
        if "refetch_reference" in command and command["refetch_reference"] is not False:
            refetch = command["refetch_reference"]
            if refetch is True:
                source = self.source_image_path
            elif isinstance(refetch, str) and refetch != "":
                source = refetch
            else:
                raise Exception("refetch_reference must be true or an http(s) URL")
            self._load_reference(source, refresh=True)
            response["source_image_path"] = self.source_image_path
        return response

    async def get_classifications(self, image: ViamImage, count: int, *, extra=None, timeout=None):
        raise NotImplementedError("feature-match-detector does not support classifications")

    async def get_classifications_from_camera(self, camera_name: str, count: int, *, extra=None, timeout=None):
        raise NotImplementedError("feature-match-detector does not support classifications")

    async def get_object_point_clouds(self, camera_name: str, *, extra=None, timeout=None):
        raise NotImplementedError("feature-match-detector does not support object point clouds")

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        result = CaptureAllResult()
        if return_image or return_detections:
            image = await self.get_cam_image(camera_name)
            if return_image:
                result.image = image
            if return_detections:
                result.detections = self._match_image(image, camera_name)
        return result

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> GetPropertiesResponse:
        properties = GetPropertiesResponse(
            classifications_supported=False,
            detections_supported=True,
            object_point_clouds_supported=False,
        )
        if self.cameras:
            properties.default_camera = self.cameras[0]
        return properties


def _attributes(config: ComponentConfig) -> dict:
    return struct_to_dict(config.attributes)


def _configured_cameras(value: Any) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise Exception("cameras must be an array of camera name strings")
    cameras = []
    for camera in value:
        if not isinstance(camera, str) or camera == "":
            raise Exception("cameras must be an array of non-empty camera name strings")
        if camera not in cameras:
            cameras.append(camera)
    return cameras


def _detection_hold_seconds(value: Any) -> float:
    if value in (None, ""):
        return DEFAULT_DETECTION_HOLD_SECONDS
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise Exception("detection_hold_seconds must be a number") from exc
    if parsed < 0:
        raise Exception("detection_hold_seconds must be greater than or equal to 0")
    return parsed


def _min_good_matches(value: Any) -> int:
    if value in (None, "", 0):
        return 15
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise Exception("min_good_matches must be an integer") from exc
    if parsed < 0:
        raise Exception("min_good_matches must be greater than or equal to 0")
    return parsed


def _validate_source(source: str):
    if source == "":
        raise Exception("A source_image_path must be defined")
    scheme = urlparse(source).scheme.lower()
    if scheme in REMOTE_SCHEMES:
        if urlparse(source).netloc == "":
            raise Exception(f"Invalid source image URI: {source}")
        return
    if scheme == "file":
        path = _file_uri_path(source)
    elif scheme == "":
        path = Path(source)
    else:
        raise Exception(
            f"Unsupported source image URI scheme '{scheme}'. Use a local path, or a file://, http://, or https:// URI."
        )
    if not path.is_file():
        raise Exception(f"Invalid source_image_path: {source}")


def _require_remote_source(source: str):
    scheme = urlparse(source).scheme.lower()
    if scheme not in REMOTE_SCHEMES:
        raise Exception("refetch_reference requires an http:// or https:// source_image_path")
    _validate_source(source)


def resolve_reference_image(source: str) -> Path:
    """Return a local file for a filesystem path or URI. Remote URIs are cached after the first download."""
    _validate_source(source)
    scheme = urlparse(source).scheme.lower()
    if scheme in REMOTE_SCHEMES:
        return _download_reference(source)
    if scheme == "file":
        return _file_uri_path(source)
    return Path(source)


def _file_uri_path(source: str) -> Path:
    return Path(url2pathname(unquote(urlparse(source).path)))


def _download_reference(uri: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()
    cached = _cached_file(digest)
    if cached is not None:
        LOGGER.info("Using cached reference image %s", cached)
        return cached
    data, content_type = _fetch_reference(uri)
    return _store_reference_bytes(uri, data, content_type)


def _fetch_reference(uri: str) -> Tuple[bytes, str]:
    LOGGER.info("Downloading reference image from %s", uri)
    request = urllib.request.Request(uri, headers={"User-Agent": "feature-match-detector"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
            content_type = response.headers.get_content_type()
    except HTTPError as exc:
        exc.close()
        raise Exception(f"Failed to download reference image from {uri}: {exc}") from exc
    except Exception as exc:
        raise Exception(f"Failed to download reference image from {uri}: {exc}") from exc
    if not data:
        raise Exception(f"Reference image download from {uri} was empty")
    return data, content_type


def _store_reference_bytes(uri: str, data: bytes, content_type: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()
    destination = CACHE_DIR / f"{digest}{_suffix_for_image(uri, content_type, data)}"
    temporary = CACHE_DIR / f".{digest}.partial"
    try:
        temporary.write_bytes(data)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    for old in CACHE_DIR.glob(f"{digest}.*"):
        if old.is_file() and old.resolve() != destination.resolve():
            old.unlink()
    LOGGER.info("Cached reference image at %s", destination)
    return destination


def _cached_file(digest: str) -> Optional[Path]:
    matches = sorted(path for path in CACHE_DIR.glob(f"{digest}.*") if path.is_file())
    if not matches:
        return None
    return matches[0]


def _suffix_for_image(uri: str, content_type: str, data: bytes) -> str:
    if content_type == "image/svg+xml" or _looks_like_svg(data):
        return ".svg"
    if data[:2] == b"\x1f\x8b" and Path(urlparse(uri).path).suffix.lower() == ".svgz":
        return ".svgz"
    if content_type in CONTENT_TYPE_SUFFIX:
        return CONTENT_TYPE_SUFFIX[content_type]
    suffix = Path(urlparse(uri).path).suffix.lower()
    if suffix in KNOWN_SUFFIXES:
        return ".jpg" if suffix == ".jpeg" else suffix
    return ".img"


def _looks_like_svg(data: bytes) -> bool:
    if data[:2] == b"\x1f\x8b":
        return False
    return b"<svg" in data[:4096].lstrip().lower()


def _path_to_gray(path: Path) -> np.ndarray:
    return _bytes_to_gray(path.read_bytes(), path.suffix)


def _bytes_to_gray(data: bytes, suffix: str = "") -> np.ndarray:
    if suffix.lower() == ".svgz":
        data = gzip.decompress(data)
    if suffix.lower() in {".svg", ".svgz"} or _looks_like_svg(data):
        image = _svg_bytes_to_pil(data)
    else:
        image = Image.open(io.BytesIO(data))
    return _pil_to_gray(image)


def _viam_image_to_gray(image: ViamImage) -> np.ndarray:
    if _looks_like_svg(image.data):
        pil_image = _svg_bytes_to_pil(image.data)
    else:
        pil_image = viam_to_pil_image(image)
    return _pil_to_gray(pil_image)


def _svg_bytes_to_pil(data: bytes) -> Image.Image:
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    document = pymupdf.open(stream=data, filetype="svg")
    try:
        if document.page_count < 1:
            raise Exception("SVG reference image did not contain a page")
        pixmap = document[0].get_pixmap(dpi=144, alpha=False)
        if pixmap.width == 0 or pixmap.height == 0:
            raise Exception("SVG reference image rasterized to an empty image")
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    finally:
        document.close()


def _detection_from_box(box: Tuple[int, int, int, int], inliers: int) -> Detection:
    min_x, min_y, max_x, max_y = box
    return Detection(
        x_min=min_x,
        y_min=min_y,
        x_max=max_x,
        y_max=max_y,
        confidence=float(min(inliers / 40, 1)),
        class_name="match",
    )


def _track_held_points(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    reference_points: np.ndarray,
    scene_points: np.ndarray,
    anchor_count: int,
    source_shape: Tuple[int, int],
) -> Optional[Tuple[Detection, np.ndarray, np.ndarray]]:
    if previous_gray.shape != current_gray.shape or len(scene_points) < MIN_HOMOGRAPHY_POINTS:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, scene_points, None, winSize=(21, 21), maxLevel=3, criteria=criteria
    )
    if next_points is None or status is None:
        return None
    back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray, previous_gray, next_points, None, winSize=(21, 21), maxLevel=3, criteria=criteria
    )
    if back_points is None or back_status is None:
        return None
    forward_backward = np.linalg.norm((back_points - scene_points).reshape(-1, 2), axis=1)
    keep = status.ravel().astype(bool) & back_status.ravel().astype(bool) & (forward_backward < TRACK_BACKTRACK_ERROR)
    needed = max(MIN_HOMOGRAPHY_POINTS, int(anchor_count * MIN_TRACKED_FRACTION))
    if int(keep.sum()) < needed:
        return None
    homography, mask = cv2.findHomography(
        reference_points[keep], next_points[keep], cv2.RANSAC, 5.0, None, 5000, 0.999
    )
    if homography is None or mask is None or not np.isfinite(homography).all():
        return None
    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    if inliers < needed or inliers / int(keep.sum()) < MIN_INLIER_RATIO:
        return None
    box = _box_from_homography(homography, source_shape, current_gray.shape[:2])
    if box is None:
        return None
    kept_reference = reference_points[keep][inlier_mask].reshape(-1, 1, 2).copy()
    kept_scene = next_points[keep][inlier_mask].reshape(-1, 1, 2).copy()
    return _detection_from_box(box, inliers), kept_reference, kept_scene


def _match_decision(inliers: int, good_count: int, minimum: int, grace: int) -> Tuple[bool, bool]:
    """Return whether the homography is a match, and whether it cleared the full inlier bar."""
    if good_count <= 0 or inliers < MIN_HOMOGRAPHY_POINTS:
        return False, False
    if inliers / good_count < MIN_INLIER_RATIO:
        return False, False
    if inliers >= max(MIN_HOMOGRAPHY_POINTS, minimum):
        return True, True
    relaxed = max(MIN_HOMOGRAPHY_POINTS, minimum // 2)
    if grace > 0 and inliers >= relaxed:
        return True, False
    return False, False


def _box_from_homography(
    homography: np.ndarray, source_shape: Tuple[int, int], query_shape: Tuple[int, int]
) -> Optional[Tuple[int, int, int, int]]:
    height, width = source_shape
    corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, homography)
    if projected is None or not np.isfinite(projected).all():
        return None
    query_height, query_width = query_shape
    min_x = int(np.floor(projected[:, 0, 0].min()))
    min_y = int(np.floor(projected[:, 0, 1].min()))
    max_x = int(np.ceil(projected[:, 0, 0].max()))
    max_y = int(np.ceil(projected[:, 0, 1].max()))
    min_x = max(0, min(min_x, query_width - 1))
    min_y = max(0, min(min_y, query_height - 1))
    max_x = max(0, min(max_x, query_width))
    max_y = max(0, min(max_y, query_height))
    if max_x - min_x < 2 or max_y - min_y < 2:
        return None
    return min_x, min_y, max_x, max_y


def _pil_to_gray(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return CLAHE.apply(gray)
