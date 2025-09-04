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
  "min_good_matches": <integer>
}
```

### Attributes

The following attributes are available for `rdk:service:vision:feature-match-detector` services:

| Name                | Type    | Inclusion | Description |
| ------------------- | ------- | --------- | ----------- |
| `source_image_path` | string  | Required  | The path to the reference image to which other images will be matched against |
| `min_good_matches`  | integer | Optional  | The minimum number of "good" keypoint matches (default: 15) |

### Example Configuration

```json
{
  "source_image_path": "/path/to/your_reference_image.jpg",
  "min_good_matches": 20
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

Note: If using this method, any cameras you are using must be set in the `depends_on` array for the service configuration:

```json
{
  "depends_on": [
    "cam"
  ]
}
```

### `do_command({"set":[{"key":"value"}]})`

You can re-configure this resource on the fly by passing a "set" object to do_command. For example, to change the source image:

```json
{
  "set": [
    {
      "key": "source_image_path",
      "value": "/path/to/refImage.png"
    }
  ]
}
```
