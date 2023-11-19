from datetime import datetime
from typing import ClassVar, Mapping, Sequence, Any, Dict, Optional, Tuple, Final, List, cast
from typing_extensions import Self

from typing import Any, Final, List, Mapping, Optional, Union

from PIL import Image

from viam.media.video import RawImage
from viam.proto.service.vision import Detection
from viam.resource.types import RESOURCE_NAMESPACE_RDK, RESOURCE_TYPE_SERVICE, Subtype
from viam.utils import ValueTypes

from viam.module.types import Reconfigurable
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily

from viam.services.vision import Vision
from viam.components.camera import Camera
from viam.logging import getLogger

import asyncio
import numpy as np
import cv2

from pathlib import Path

DETECTOR = cv2.SIFT_create()
MATCHER = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
LOGGER = getLogger(__name__)

class featureMatchDetector(Vision, Reconfigurable):
    
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "detector"), "feature-match-detector")
    
    source_image_path: str
    source_keypoints: dict
    source_descriptors: dict

    # Constructor
    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        my_class = cls(config.name)
        my_class.reconfigure(config, dependencies)
        return my_class

    # Validates JSON Configuration
    @classmethod
    def validate(cls, config: ComponentConfig):
        source_image_path = config.attributes.fields["source_image_path"].string_value
        if source_image_path == "":
            raise Exception("A source_image_path must be defined")
        if not Path(source_image_path).exists():
            raise Exception("Invalid source_image_path: " + source_image_path)
        return

    # Handles attribute reconfiguration
    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        self.source_image_path = config.attributes.fields["source_image_path"].string_value
        self.init_source_image()
        self.min_good_matches = config.attributes.fields["min_good_matches"].number_value or 15
        self.DEPS = dependencies
        return

    def init_source_image(self):
        im = Image.open(self.source_image_path)
        imgTrainGray = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2GRAY)
        kp1, des1 = DETECTOR.detectAndCompute(imgTrainGray,None)
        self.source_keypoints = kp1
        self.source_descriptors = des1
        return
    
    async def get_detections_from_camera(
        self, camera_name: str, *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None
    ) -> List[Detection]:
        actual_cam = self.DEPS[Camera.get_resource_name(camera_name)]
        cam = cast(Camera, actual_cam)
        cam_image = await cam.get_image(mime_type="image/jpeg")
        return await self.get_detections(cam_image)

    
    async def get_detections(
        self,
        image: Union[Image.Image, RawImage],
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        detections = []
        imgTrainGray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        kp2, des2 = DETECTOR.detectAndCompute(imgTrainGray,None)
        matches = MATCHER.knnMatch(self.source_descriptors,des2,2)
        good = []
        for m,n in matches:
            if m.distance < .7 * n.distance:
                good.append(m)

        if len(good) >= self.min_good_matches:
            src_pts = np.float32([ self.source_keypoints[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            pts = dst_pts[mask==1]
            if len(pts):
                min_x, min_y = np.int32(pts.min(axis=0))
                max_x, max_y = np.int32(pts.max(axis=0))
                confidence = len(good) / 40
                if confidence > 1:
                    confidence = 1
                detections.append({ "confidence": confidence, "class_name": "match", "x_min": min_x, "y_min": min_y, 
                                        "x_max": max_x, "y_max": max_y } )
        return detections

    # Implements set=[{key=,value=}] to allow for config changes on the fly
    async def do_command(self, input: Mapping[str, ValueTypes], *,
                         timeout: Optional[float] = None,
                         **kwargs) -> Mapping[str, ValueTypes]:
        LOGGER.info(input)
        resp = {
            "response": "OK",
            "timestamp": str(datetime.now())
        }
        if "set" in input.keys():
            for s in input["set"]:
                if "key" in s.keys():
                    if s["key"] == "source_image_path":
                        self.source_image_path = s["value"]
                        self.init_source_image()
                    if s["key"] == "min_good_matches":
                        self.min_good_matches = s["value"]
        return resp
    
    async def get_classifications(self):
        return
    
    async def get_classifications_from_camera(self):
        return
    
    async def get_object_point_clouds(self):
        return