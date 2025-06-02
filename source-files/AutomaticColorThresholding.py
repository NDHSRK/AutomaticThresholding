
# Based on https://pyimagesearch.com/2021/02/15/automatic-color-correction-with-opencv-and-python/

# In our adaptation of the pyimagesearch project we're using the
# Pantone color card code not for color correction but for the
# ArUco marker detection and perspective normalization.

# import the necessary packages
import numpy as np
import argparse
import cv2
import sys
import perspective
from ImageUtils import ImageUtils
import matplotlib.pyplot as plt

MIN_SAT_THRESHOLD = 25
MAX_SAT_THRESHOLD = 250

#########################################################
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
imageCard = perspective.find_color_card(pantone_image)

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

# Replace the color in the aperture with the color
# from the calibration image, which is also 85x60.
imageCard[aperture_y1: aperture_y2, aperture_x1: aperture_x2] = aperture_calibration_image

# Write and show the color matching card adjusted for perspective
# # with our selected color in the aperture.
cv2.imwrite(image_dir + "Pantone_01_aruco_.png", imageCard)
cv2.imshow("Pantone card with calibration aperture", imageCard)
cv2.waitKey(0)

##**TODO "in the wild" you would get an image of the card
# with the selected color in the aperture. So you would
# need a mask to isolate the aperture.

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

status, last_saturation_threshold = ImageUtils.iterateThreshold(lambda control_variable: ImageUtils.threshold_hsv(target_hsv, hsv_hue_low, hsv_hue_high, control_variable, VALUE_LOW), saturation_low, ImageUtils.ThresholdControlDirection.INITIAL,
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


