
# Based on https://pyimagesearch.com/2021/02/15/automatic-color-correction-with-opencv-and-python/

# In our adaptation of the pyimagesearch project we're using the
# Pantone color card code not for color correction but for the
# ArUco marker detection and perspective normalization.

# import the necessary packages
from perspective import four_point_transform
import numpy as np
import argparse
import cv2
import sys
from enum import Enum
from ImageUtils import ImageUtils
import matplotlib.pyplot as plt

MIN_SAT_THRESHOLD = 25
MAX_SAT_THRESHOLD = 250

def find_color_card(image):
    # load the ArUCo dictionary, grab the ArUCo parameters, and
    # detect the markers in the input image

    ## with release 4.7 this is the method ...
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
    corners, ids, rejected = detector.detectMarkers(image)

    # pyimagesearch version
    #arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    #arucoParams = cv2.aruco.DetectorParameters_create()
    #(corners, ids, rejected) = cv2.aruco.detectMarkers(image,
    #	arucoDict, parameters=arucoParams)

    # try to extract the coordinates of the color correction card
    try:
        # We've found the four ArUco markers, so we can
        # continue by flattening the ArUco IDs list
        ids = ids.flatten()

        # extract the top-left marker
        i = np.squeeze(np.where(ids == 923))
        topLeft = np.squeeze(corners[i])[0]

        # extract the top-right marker
        i = np.squeeze(np.where(ids == 1001))
        topRight = np.squeeze(corners[i])[1]

        # extract the bottom-right marker
        i = np.squeeze(np.where(ids == 241))
        bottomRight = np.squeeze(corners[i])[2]

        # extract the bottom-left marker
        i = np.squeeze(np.where(ids == 1007))
        bottomLeft = np.squeeze(corners[i])[3]

    # we could not find color correction card, so gracefully return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    # build our list of reference points and apply a perspective
    # transform to obtain a top-down, birds-eye-view of the color
    # matching card
    cardCoords = np.array([topLeft, topRight,
        bottomRight, bottomLeft])
    card = four_point_transform(image, cardCoords)

    # return the color matching card to the calling function
    return card

class SaturationDirection(Enum):
         INITIAL = 0
         INCREMENT = 1
         DECREMENT = 2

# Given an HSV image assumed to contain a single sample,
# this function attempts to determine the cv2.inRange
# parameters that will correctly threshold the sample.
#
# Input parameters are the HSV image, the low hue value,
# the high hue value, the current saturation level, and
# the current value level.

# In this version of the function only the saturation threshold
# will be either increased or decreased.
# Returns Boolean (True == good threshold found), final saturation threshold

##**TODO How to make this function generic so that it will work on both
# HSV and grayscale images - after all, both modify only a single
# parameter: saturation threshold for HSV cv2.inRange() and grayscale
# threshold for cv2.threshold(). A Python lambda should work here;
# Move this function to ImageUtils.
def iterateThreshold(p_hsv_image, p_hsv_hue_low, p_hsv_hue_high, saturation_threshold, value_threshold,
                     saturation_direction,
                     min_sample_area, max_sample_area):
    SAT_THRESHOLD_CHANGE = 5

    # Besides the target sample we'll allow a few contours below
    # the minimum area. These will be filtered out later.
    MAX_CONTOURS_BELOW_MIN_AREA = 10  # some may be zero length or not closed

    # Try inRange with the passed-in saturation and value thresholds.
    # If these pass the non-zero pixel count filter and produce a
    # single clean thresholded binary image with fewer than the maximum
    # number of contours then we're done.

    hsv_thr = ImageUtils.threshold_hsv(p_hsv_image, p_hsv_hue_low, p_hsv_hue_high, saturation_threshold, value_threshold)
    thr_non_zero_count = cv2.countNonZero(hsv_thr)
    print("HSV saturation threshold " + str(saturation_threshold))
    print("Thresholded HSV non-zero pixel count " + str(thr_non_zero_count))
    # cv2.imshow('Thresholded HSV', hsv_thr)
    # cv2.waitKey(0)

    # Filter the contours and rotated rectangles.
    hsv_thr_height, hsv_thr_width = hsv_thr.shape
    filtered_contours = ImageUtils.filter_contours(hsv_thr, hsv_thr_height, hsv_thr_width, min_sample_area, max_sample_area)

    # The image is too sparse if:
    #   the total number of contours is 0
    #   the total number of rotated rectangles within the area range is 0
    # The image is too busy if:
    #   the total number of contours is above MAX_CONTOURS_BELOW_MIN_AREA
    #   the total number of rotated rectangles is > 1

    # Take the desired case first.
    if len(filtered_contours.filtered_binary_output) == 1 and filtered_contours.numBelowMinArea <= MAX_CONTOURS_BELOW_MIN_AREA and filtered_contours.numAboveMaxArea == 0:
        return True, saturation_threshold  # all good

    # If the thresholded image is too sparse then we need to
    # lower the saturation threshold.
    if filtered_contours.numUnfilteredContours == 0:
        if saturation_direction == SaturationDirection.INCREMENT:
            print("Error: reversal of saturation direction from increment to decrement")
            return False, saturation_threshold

        # INITIAL or DECREMENT
        next_sat_threshold = saturation_threshold - SAT_THRESHOLD_CHANGE
        if next_sat_threshold < MIN_SAT_THRESHOLD:
            print("Error: below minimum saturation threshold")
            return False, saturation_threshold

        # call self
        next_sat_direction = SaturationDirection.DECREMENT
        return iterateThreshold(p_hsv_image, p_hsv_hue_low, p_hsv_hue_high, next_sat_threshold, value_threshold,
                                next_sat_direction,
                                min_sample_area, max_sample_area)

    # If the thresholded image is too busy or we've found more than
    # one qualifying rectangle or an oversized blob, then we need to
    # raise the saturation threshold.
    if filtered_contours.numBelowMinArea > MAX_CONTOURS_BELOW_MIN_AREA or len(
            filtered_contours.filtered_binary_output) > 1 or filtered_contours.numAboveMaxArea != 0:
        if saturation_direction == SaturationDirection.DECREMENT:
            print("Error: reversal of saturation direction from decrement to increment")
            return False, saturation_threshold

        # INITIAL or INCREMENT
        next_sat_threshold = saturation_threshold + SAT_THRESHOLD_CHANGE
        if next_sat_threshold > MAX_SAT_THRESHOLD:
            print("Error: above maximum saturation threshold")
            return False, saturation_threshold

        # call self
        next_sat_direction = SaturationDirection.INCREMENT
        return iterateThreshold(p_hsv_image, hsv_hue_low, hsv_hue_high, next_sat_threshold, value_threshold,
                                next_sat_direction,
                                min_sample_area, max_sample_area)

    # failsafe
    print("Unhandled condition in iterateThreshold")
    sys.exit(1)  # OR raise exception

## main ##
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--image_dir", type=str)
ap.add_argument("--aperture", type=str)
ap.add_argument("--target", type=str)
args = vars(ap.parse_args())

image_dir = args["image_dir"]
aperture_calibration_filename = args["aperture"]
target_image_filename = args["target"]

# Read in any Pantone image from pyimagesearch because
# we're going to replace the color in the aperture with
# one extracted from our calibration image.

##**TODO Required:
# Put our own ArUco markers on a sheet of paper into which
# we have cut an aperture; in this way we'll get a larger
# swatch of the color.
pantone_image_filename = "Pantone_01.jpg"
pantone_image_path = image_dir + pantone_image_filename
pantone_image = cv2.imread(pantone_image_path, cv2.IMREAD_COLOR)
if pantone_image is None:
    print('Pantone image file not found')
    sys.exit(1)

# This is an image that is the same size as the aperture in the
# center of the Pantone card, For images of single samples that
# already exist we use Gimp to extract a sub-image that can be
# resized to the same dimensions as the aperture.
aperture_calibration_path = image_dir + aperture_calibration_filename
aperture_calibration_image = cv2.imread(aperture_calibration_path, cv2.IMREAD_COLOR)
if aperture_calibration_image is None:
    print('Aperture calibration image file not found')
    sys.exit(1)

# Test this image against our color calibration.
target_image_path = image_dir + target_image_filename
target_image = cv2.imread(target_image_path, cv2.IMREAD_COLOR)
if target_image is None:
    print('Target image file not found')
    sys.exit(1)

# The Pantone card image is 3024x4032. Use the scaling factor
# from pyimagesearch.
SCALED_CARD_WIDTH = 600
SCALED_CARD_HEIGHT = 600
SCALED_CARD_SIZE = (SCALED_CARD_WIDTH, SCALED_CARD_HEIGHT)
pantone_image = cv2.resize(pantone_image, SCALED_CARD_SIZE, interpolation=cv2.INTER_AREA)

# Find the color matching card in the input image
print("Finding color matching card ...")
imageCard = find_color_card(pantone_image)

# If the color matching card is not found, just in either the reference
# exit.
if imageCard is None:
    print("Could not find color matching card")
    sys.exit(0)

## PY added
# After the Perspective Transform the ArUco markers are
# aligned with the edges of the image. The center of the
# aperture is at the center of the image.

##**TODO Will these numbers hold even if the camera is at a
# different angle? I think they should.

# With the image resized to 600x600 the size of the transformed
# image of the card only (in Gimp) is 339x314 and the size of
# the aperture is width 85 x height 60
APERTURE_WIDTH = 85
APERTURE_HEIGHT = 60

# Get the center point of the card and define the aperture.
height, width, channels = imageCard.shape
card_center_x = width / 2.0
card_center_y = height / 2.0
aperture_x1 = int(card_center_x - (APERTURE_WIDTH / 2.0))
aperture_y1 = int(card_center_y - (APERTURE_HEIGHT / 2.0)) # upper left
aperture_x2 = int(card_center_x + (APERTURE_WIDTH / 2.0))
aperture_y2 = int(card_center_y + (APERTURE_HEIGHT / 2.0)) # lower right

# The aperture calibration image must have the same dimensions
# as the aperture in the calibration card.
ach, acw, _ = aperture_calibration_image.shape
if ach != APERTURE_HEIGHT or acw != APERTURE_WIDTH:
    print("Aperture calibration image size does not match that of the card aperture")
    sys.exit(0)

# Get the minimum and maximum saturation levels from the aperture.
aperture_hsv = cv2.cvtColor(aperture_calibration_image, cv2.COLOR_BGR2HSV)
aperture_h, aperture_sat, aperture_val = cv2.split(aperture_hsv)
aperture_min_sat = np.min(aperture_sat)
aperture_max_sat = np.max(aperture_sat)
aperture_min_val = np.min(aperture_val)
aperture_max_val = np.max(aperture_val)

print("Aperture saturation minimum " + str(aperture_min_sat) + ", maximum " + str(aperture_max_sat))
print("Aperture value minimum " + str(aperture_min_val) + ", maximum " + str(aperture_max_val))

# Legacy images only: replace the color in the aperture with the color
# from the calibration image, which is also 85x60.
imageCard[aperture_y1: aperture_y2, aperture_x1: aperture_x2] = aperture_calibration_image

# Write and show the color matching card adjusted for perspective.
cv2.imwrite(image_dir + "Pantone_01_aruco_.png", imageCard)
cv2.imshow("Pantone card with calibration aperture", imageCard)
cv2.waitKey(0)

# Create a mask for the histogram by drawing a filled rectangle
# over the aperture.
aperture_mask = np.zeros((height, width), dtype=np.uint8)
cv2.rectangle(aperture_mask, (aperture_x1, aperture_y1), (aperture_x2, aperture_y2), 255, cv2.FILLED)

# Get the histogram of the color behind the aperture.
card_hsv = cv2.cvtColor(imageCard, cv2.COLOR_BGR2HSV)
histSize = 180 # one bin for each OpenCV HSV hue value
ranges = [0, 180]

# [0] is the channel = hue; mask is the aperture; [180] is the number of bins; [0, 180] is the range
hist = cv2.calcHist([card_hsv], [0], aperture_mask, [histSize], ranges)

# The bin index and the hue are the same because we've allocated 180 bins,
# one for each hue.
dominant_bin_index = np.argmax(hist)
print("Dominant hue/bin " + str(dominant_bin_index))

plt.plot(hist)
plt.title('Histogram of selected channel')
plt.xlabel('Pixel Intensity')
plt.ylabel('Pixel Count')
plt.show()

hsv_hue_low, hsv_hue_high = ImageUtils.get_hue_range(hist, dominant_bin_index)

# We're using the color in the aperture to get the hue
# range, then we're iterating over the full image to
# get the saturation threshold.

##**TODO Verify assumptions:
# Assumption: the hue range of the Pantone card aperture
# is the same as that of the full sample. In testing this
# has not always been the case so we'll have to create a
# pseudo Pantone card with ArUco markers with an aperture
# that includes the entire sample.

# Assumption: the hue range from the aperture will remain
# valid when applied to the full image of a single sample.

# Assumption: changes to the saturation threshold will not
# affect the hue range.

# Assumption: once we get all of the cv2.inRange() parameters
# for thresholding the full image of a single sample we will
# be able to apply these same parameters to images
# that contain samples of all three colors.

target_hsv = cv2.cvtColor(target_image, cv2.COLOR_BGR2HSV)

# For debugging split and get the minimum saturation for the full image.
target_h, target_sat, target_val = cv2.split(target_hsv)
target_min_sat = np.min(target_sat)
target_max_sat = np.max(target_sat)
target_min_val = np.min(target_val)
target_max_val = np.max(target_val)

print("Target saturation minimum " + str(target_min_sat) + ", maximum " + str(target_max_sat))
print("Target value minimum " + str(target_min_val) + ", maximum " + str(target_max_val))

##**TODO Are there any cases where the thresholding starts
# too high and we have to decrement?

# It's better to start with a busy image and then gradually
# increase the saturation to reduce the number of objects.
# So if the minimum saturation of the image is below the
# default then lower the default to create a busy image.
SATURATION_LOW_DEFAULT = 125
saturation_low = SATURATION_LOW_DEFAULT
VALUE_LOW = 125

if aperture_min_sat < SATURATION_LOW_DEFAULT:
    saturation_low = MIN_SAT_THRESHOLD

MIN_SAMPLE_AREA = 14000
MAX_SAMPLE_AREA = 21000

status, last_saturation_threshold = iterateThreshold(target_hsv, hsv_hue_low, hsv_hue_high, saturation_low, VALUE_LOW,
                                                     SaturationDirection.INITIAL,
                                                     MIN_SAMPLE_AREA, MAX_SAMPLE_AREA)

print("Final HSV inRange parameters: ")
print("Hue low " + str(hsv_hue_low) + ", hue high " + str(hsv_hue_high))
print("Saturation threshold low " + str(last_saturation_threshold))
print("Value threshold low " + str(VALUE_LOW))
final_thresholded_image = ImageUtils.threshold_hsv(target_hsv, hsv_hue_low, hsv_hue_high, last_saturation_threshold, VALUE_LOW)
if status:
    cv2.imshow("Final RotatedRect for sample at saturation threshold", final_thresholded_image)
    cv2.waitKey(0)
else:
    print("Unable to determine threshold levels")
    cv2.imshow("Sample at time of error at saturation threshold", final_thresholded_image)
    cv2.waitKey(0)


