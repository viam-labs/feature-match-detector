"""
This file registers the model with the Python SDK.
"""

from viam.services.vision import Vision
from viam.resource.registry import Registry, ResourceCreatorRegistration

from .featureMatchDetector import featureMatchDetector

Registry.register_resource_creator(
    Vision.API, featureMatchDetector.MODEL, ResourceCreatorRegistration(featureMatchDetector.new, featureMatchDetector.validate_config)
)
