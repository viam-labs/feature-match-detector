
# feature-match-detector

*feature-match-detector* is a Viam modular vision service that uses [OpenCV's implementation of the SIFT algorithm](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html) to identify feature-based matches between a source (reference) image and another image.

## API

The feature-match-detector resource provides the following methods from Viam's built-in [rdk:service:vision API](https://python.viam.dev/autoapi/viam/services/vision/client/index.html)

### get_detections(image=*binary*)

### get_detections_from_camera(camera_name=*string*)

Note: if using this method, any cameras you are using must be set in the `depends_on` array for the service configuration, for example:

```json
      "depends_on": [
        "cam"
      ]
```

### do_command({"set":[{"key":"value"}]})

If you pass set as the key in an object passed to do_command, you may then re-configure this resource on the fly by specifying config attributes to change.

For example, the following passed to do_command would change the source image:

``` json
{
    "set":
    [
        { 
            "key":"source_image_path",
            "value":"/path/to/refImage.png"
        }
    ]
}
```

## Viam Service Configuration

The following attributes may be configured as feature-match-detector config attributes.

For example: the following configuration would configure the source image, and select minimum 20 good key point matches:

``` json
{
  "source_image_path": "/path/to/your_reference_image.jpg",
  "min_good_matches": 20
}
```

### source_image_path

*string (required)*

The path to the reference image to which other images will be matched against.

### min_good_matches

*integer (default: 15)*

The minimum number of "good" keypoint matches (as determined by [Lowe's ratio test](https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html)).
