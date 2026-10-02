# `feature-match-detector` modular service

This module implements the [vision service API](https://docs.viam.com/dev/reference/apis/services/vision/) in a `rdk:service:vision:feature-match-detector` model.
With this model, you can identify feature-based matches between a source (reference) image and another image using [OpenCV's implementation of the SIFT algorithm](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html).

Navigate to the **CONFIGURE** tab of your machine's page.
Click the **+** button, select **Component or service**, then select the `vision / feature-match-detector` model provided by the [`feature-match-detector` module](https://app.viam.com/module/feature-match-detector).
Click **Add module**, enter a name for your vision service, and click **Create**.

## Configure your `feature-match-detector` service

On the new service panel, copy and paste the following attribute template into your service's **Attributes** box:

```json
{
  "source_image_path": "<string>",
  "cameras": ["<camera name>"],
  "min_good_matches": <integer>,
  "detection_hold_seconds": <number>
}
```

### Attributes

The following attributes are available for `rdk:service:vision:feature-match-detector` services:

| Name                | Type             | Inclusion                         | Description |
| ------------------- | ---------------- | --------------------------------- | ----------- |
| `source_image_path` | string           | Required                          | Local filesystem path, or a `file://`, `http://`, or `https://` URI, for the reference image. Remote URIs are downloaded on first use and then read from the module's `.cache` directory. JPEG, PNG, and SVG (including SVGZ) are supported. |
| `cameras`           | array of strings | Required for camera methods       | Names of cameras `get_detections_from_camera` and `capture_all_from_camera` may use. Each name is added as an implicit dependency. The first entry is reported as the default camera. |
| `min_good_matches`  | integer          | Optional                          | The minimum number of homography inliers required to report a match (default: 15). For the next two frames after a match, that camera may pass with about half this count. |
| `detection_hold_seconds` | number    | Optional                          | How long a camera keeps following its last match when later frames miss the feature check (default: 5). The box moves with the tracked object. Set to 0 to drop the match on the first miss. |

### Example Configuration

Local reference image:

```json
{
  "source_image_path": "/path/to/your_reference_image.jpg",
  "cameras": ["cam"],
  "min_good_matches": 20
}
```

Remote reference image (downloaded once, then cached):

```json
{
  "source_image_path": "https://example.com/reference.svg",
  "cameras": ["cam"]
}
```

## Prerequisites

For Linux systems, install the required OpenGL library:

```bash
sudo apt-get install libgl1
```

## API Methods

The `feature-match-detector` service provides the following methods from Viam's built-in [vision service API](https://docs.viam.com/dev/reference/apis/services/vision/):

### `get_detections(image=*binary*)`

### `get_detections_from_camera(camera_name=*string*)`

`camera_name` must be one of the names in the `cameras` array. Those cameras are implicit dependencies, so they do not also need to be listed in `depends_on`.

Frames are contrast-normalized before matching, and the detection box is the reference image projected into the camera frame. After a match, that camera follows those points for `detection_hold_seconds` (default 5) when later frames miss the feature check, so the box stays on the object while it moves. The detection drops when the points can no longer be followed, or when the hold expires.

### `do_command({"set":[{"key":"value"}]})`

You can re-configure this resource on the fly by passing a "set" object to do_command. For example, to change the source image:

```json
{
  "set": [
    {
      "key": "source_image_path",
      "value": "https://example.com/refImage.svg"
    }
  ]
}
```

`source_image_path` accepts a local path or a `file://`, `http://`, or `https://` URI. A remote URI that was already downloaded is read from disk.

To download the current remote reference again and replace the cached file:

```json
{ "refetch_reference": true }
```

You can also pass a URL. That URL is downloaded, cached, and becomes the reference image:

```json
{ "refetch_reference": "https://example.com/refImage.svg" }
```

`refetch_reference` only applies to `http://` and `https://` URIs. The previous cache and loaded features stay in place if the download fails or the new image does not contain enough features.
