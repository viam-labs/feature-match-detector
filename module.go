// package main is a module that implements a feature match detection service
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"math"

	"github.com/edaniels/golog"
	"github.com/pkg/errors"
	"go.opencensus.io/trace"
	goutils "go.viam.com/utils"

	"go.viam.com/rdk/config"
	"go.viam.com/rdk/module"
	"go.viam.com/rdk/registry"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/rimage"
	"go.viam.com/rdk/services/vision"
	"go.viam.com/rdk/utils"
	kp "go.viam.com/rdk/vision/keypoints"
	"go.viam.com/rdk/vision/objectdetection"
)

var model = resource.NewModel("viamlabs", "service", "feature-match-detector")

func main() {
	goutils.ContextualMain(mainWithArgs, golog.NewDevelopmentLogger("featureMatchDetectorModule"))
}

func mainWithArgs(ctx context.Context, args []string, logger golog.Logger) (err error) {
	registerDetector()
	modalModule, err := module.NewModuleFromArgs(ctx, logger)

	if err != nil {
		return err
	}
	modalModule.AddModelFromRegistry(ctx, vision.Subtype, model)

	err = modalModule.Start(ctx)
	defer modalModule.Close(ctx)

	if err != nil {
		return err
	}
	<-ctx.Done()
	return nil
}

// helper function to add the detector's constructor and metadata to the component registry, so that we can later construct it.
func registerDetector() {
	registry.RegisterService(
		vision.Subtype,
		model,
		registry.Service{Constructor: func(
			ctx context.Context,
			deps registry.Dependencies,
			config config.Service,
			logger golog.Logger,
		) (interface{}, error) {
			return newFeatureMatchDetector(ctx, config, logger)
		}})
}

// FeatureMatchDetectorParams specifies the fields necessary for creating a feature match detector.
type FeatureMatchDetectorParams struct {
	// this should come from the attributes part of the detector config
	ReferenceImagePath string `json:"reference_image_path"`
	MaxDist            int    `json:"max_match_distance,omitempty"`
}

type OrbKP struct {
	descriptors [][]uint64
	keypoints   kp.KeyPoints
}

// NewFeatureMatchDetector creates an RDK detector given a DetectorConfig. In other words, this
// function returns a function from image-->[]objectdetection.Detection. It does this by making calls to
// a keypoints package and wrapping the result.
func newFeatureMatchDetector(
	ctx context.Context,
	conf config.Service,
	logger golog.Logger,
) (objectdetection.Detector, error) {
	ctx, span := trace.StartSpan(ctx, "service::vision::NewFeatureMatchDetector")
	defer span.End()

	var t FeatureMatchDetectorParams
	fmParams, err := config.TransformAttributeMapToStruct(&t, config.AttributeMap(conf.Attributes))

	if err != nil {
		return nil, errors.New("error getting parameters from config")
	}
	params, ok := fmParams.(*FeatureMatchDetectorParams)
	if !ok {
		err := utils.NewUnexpectedTypeError(params, fmParams)
		return nil, errors.Wrapf(err, "register feature match detector %s", conf.Name)
	}

	// load reference image and compute keypoints
	img, err := rimage.NewImageFromFile(params.ReferenceImagePath)
	if err != nil {
		return nil, errors.Wrapf(err, "something wrong with loading the reference image: %s", params.ReferenceImagePath)
	}
	refKPs, err := getImageKeypoints(ctx, img)
	if err != nil {
		return nil, errors.Wrap(err, "something wrong computing keypoints")
	}

	// This function to be returned is the detector.
	return func(ctx context.Context, img image.Image) ([]objectdetection.Detection, error) {
		maxDist := params.MaxDist
		if maxDist == 0 {
			maxDist = 50
		}
		matchingConf := &kp.MatchingConfig{
			DoCrossCheck: true,
			MaxDist:      maxDist,
		}
		imgKPs, err := getImageKeypoints(ctx, rimage.ConvertImage(img))
		if err != nil {
			return nil, errors.Wrap(err, "something wrong getting image keypoints")
		}
		matches := kp.MatchDescriptors(refKPs.descriptors, imgKPs.descriptors, matchingConf, logger)
		bounds := getBoundingBox(matches, imgKPs.keypoints)

		// Only ever return max one detection
		var detections = []objectdetection.Detection{}
		detections[0] = objectdetection.NewDetection(bounds, 1, "match")
		res2B, _ := json.Marshal(detections)
		fmt.Println(string(res2B))
		return detections, nil
	}, nil
}

// getBoundingBox returns a rectangle based on min/max x,y of matches in the match image
func getBoundingBox(matches []kp.DescriptorMatch, pts kp.KeyPoints) image.Rectangle {
	min := image.Point{math.MaxInt32, math.MaxInt32}
	max := image.Point{0, 0}

	for _, match := range matches {
		m := pts[match.Idx2]
		if m.X < min.X {
			min.X = m.X
		}
		if m.Y < min.Y {
			min.Y = m.Y
		}

		if m.X > max.X {
			max.X = m.X
		}
		if m.Y > max.Y {
			max.Y = m.Y
		}
	}

	return image.Rectangle{min, max}
}

// getImageKeypoints reads an image from the specified path and
// returns descriptors and keypoints, which are cached for detector matching
func getImageKeypoints(ctx context.Context, img *rimage.Image) (*OrbKP, error) {
	_, span := trace.StartSpan(ctx, "service::vision::getImageKeypoints")
	defer span.End()

	orbConf := &kp.ORBConfig{
		Layers:          4,
		DownscaleFactor: 2,
		FastConf: &kp.FASTConfig{
			NMatchesCircle: 9,
			NMSWinSize:     7,
			Threshold:      20,
			Oriented:       true,
			Radius:         16,
		},
		BRIEFConf: &kp.BRIEFConfig{
			N:              512,
			Sampling:       2,
			UseOrientation: true,
			PatchSize:      48,
		},
	}

	imG := rimage.MakeGray(img)
	samplePoints := kp.GenerateSamplePairs(orbConf.BRIEFConf.Sampling, orbConf.BRIEFConf.N, orbConf.BRIEFConf.PatchSize)
	var kps OrbKP
	orb, kp, err := kp.ComputeORBKeypoints(imG, samplePoints, orbConf)
	if err != nil {
		return nil, err
	}
	kps.descriptors = orb
	kps.keypoints = kp

	return &kps, nil
}
