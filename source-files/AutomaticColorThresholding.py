
# Based on https://pyimagesearch.com/2021/02/15/automatic-color-correction-with-opencv-and-python/

# In our adaptation of the pyimagesearch project we're using the
# Pantone color card code not for color correction but for the
# ArUco marker detection and perspective normalization.

# import the necessary packages
import numpy as np
import argparse
import cv2
import sys
import aperture
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
# we're going to replace the color in the aperture with# one extracted from our calibration image.
pantone_image_filename = "Pantone_01.jpg"
pantone_image_path = image_dir + pantone_image_filename
pantone_image = cv2.imread(pantone_image_path, cv2.IMREAD_COLOR)
if pantone_image is None:
    print('Pantone image file not found')
    sys.exit(1)

# Read in a color image that is the same size as the aperture
# in the center of the Pantone card. For this proof-of-concept
# this image must created manually in Gimp from an image of a
# single sample taken at some earlier date and time.
aperture_calibration_path = image_dir + aperture_calibration_filename
aperture_calibration_image = cv2.imread(aperture_calibration_path, cv2.IMREAD_COLOR)
if aperture_calibration_image is None:
    print('Aperture calibration image file not found')
    sys.exit(1)

# Test a color image with a single sample image against
# our color calibration. Typically this will be the
# same full image of a single sample from which the
# aperture calibration image was extracted via Gimp.
# See the guidelines for more information.
target_image_path = image_dir + target_image_filename
target_image = cv2.imread(target_image_path, cv2.IMREAD_COLOR)
if target_image is None:
    print('Target image file not found')
    sys.exit(1)

# Now start processing.
# Get the minimum and maximum saturation and value  levels from the aperture.
aperture_hsv = cv2.cvtColor(aperture_calibration_image, cv2.COLOR_BGR2HSV)
aperture_h, aperture_sat, aperture_val = cv2.split(aperture_hsv)
aperture_min_sat = np.min(aperture_sat)
aperture_max_sat = np.max(aperture_sat)
aperture_min_val = np.min(aperture_val)
aperture_max_val = np.max(aperture_val)

print("Aperture saturation minimum " + str(aperture_min_sat) + ", maximum " + str(aperture_max_sat))
print("Aperture value minimum " + str(aperture_min_val) + ", maximum " + str(aperture_max_val))

# The second aperture_calibration_image argument is the same as the first
# because the aperture calibration image is already in the BGR format.
card_image, aperture_mask = aperture.prepare_aperture(pantone_image, aperture_calibration_image, aperture_calibration_image, image_dir)

# Get the histogram of the color behind the aperture.
card_hsv = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)
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
target_hsv = cv2.cvtColor(target_image, cv2.COLOR_BGR2HSV)

# For debugging split and get the minimum saturation for the full image.
target_h, target_sat, target_val = cv2.split(target_hsv)
target_min_sat = np.min(target_sat)
target_max_sat = np.max(target_sat)
target_min_val = np.min(target_val)
target_max_val = np.max(target_val)

print("Target saturation minimum " + str(target_min_sat) + ", maximum " + str(target_max_sat))
print("Target value minimum " + str(target_min_val) + ", maximum " + str(target_max_val))

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

status, last_saturation_threshold = ImageUtils.iterateThreshold(lambda control_variable: ImageUtils.threshold_hsv(target_hsv, hsv_hue_low, hsv_hue_high, control_variable, VALUE_LOW), saturation_low, ImageUtils.ThresholdControlDirection.INITIAL)

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


