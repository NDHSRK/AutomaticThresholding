# Based on https://pyimagesearch.com/2021/02/15/automatic-color-correction-with-opencv-and-python/

# In our adaptation of the pyimagesearch project we're using the
# Pantone color card code not for color correction but for the
# ArUco marker detection and perspective normalization.

# import the necessary packages
import numpy as np
import argparse
import cv2
import sys
from enum import Enum
import aperture
from ImageUtils import ImageUtils
import matplotlib.pyplot as plt

MIN_GRAYSCALE_THRESHOLD = 25
MAX_GRAYSCALE_THRESHOLD = 250

class GrayscaleSource(Enum):
    RED_CHANNEL = 0
    RED_CHANNEL_INVERTED = 1,
    GREEN_CHANNEL = 2
    GREEN_CHANNEL_INVERTED = 3,
    BLUE_CHANNEL = 4
    BLUE_CHANNEL_INVERTED = 5,
    COLOR_TO_GRAY = 6

def select_grayscale(bgr_image, source_selection):
    bgr_height, bgr_width, bgr_channels = bgr_image.shape
    xFF = np.ones((bgr_height, bgr_width, 1), dtype=np.uint8) * 255

    invert_selected_channel = False
    match source_selection:
        case GrayscaleSource.BLUE_CHANNEL:
            selected_channel = cv2.extractChannel(bgr_image, 0)  # B = 0, G = 1, R = 2
        case GrayscaleSource.BLUE_CHANNEL_INVERTED:
            invert_selected_channel = True
            selected_channel = cv2.extractChannel(bgr_image, 0)  # B = 0, G = 1, R = 2
        case GrayscaleSource.GREEN_CHANNEL:
            selected_channel = cv2.extractChannel(bgr_image, 1)  # B = 0, G = 1, R = 2
        case GrayscaleSource.GREEN_CHANNEL_INVERTED:
            invert_selected_channel = True
            selected_channel = cv2.extractChannel(bgr_image, 1)  # B = 0, G = 1, R = 2
        case GrayscaleSource.RED_CHANNEL:
            selected_channel = cv2.extractChannel(bgr_image, 2)  # B = 0, G = 1, R = 2
        case GrayscaleSource.RED_CHANNEL_INVERTED:
            invert_selected_channel = True
            selected_channel = cv2.extractChannel(bgr_image, 2)  # B = 0, G = 1, R = 2
        case GrayscaleSource.COLOR_TO_GRAY:
            selected_channel = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        case _:
            raise Exception("Unrecognized grayscale source")

    if invert_selected_channel:
        selected_channel = cv2.subtract(xFF, selected_channel)

    return selected_channel

def grayscale_threshold_wrapper(p_grayscale_image, grayscale_threshold_low):
  _, thresholded = cv2.threshold(p_grayscale_image, grayscale_threshold_low, 255, cv2.THRESH_BINARY)
  return thresholded

##########################################################
## main ##

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--image_dir", type=str)
ap.add_argument("--hsv", action="store_true")
ap.add_argument("--gray", action="store_true")
ap.add_argument("--source", type=str) # only with --gray
ap.add_argument("--aperture", type=str)
ap.add_argument("--target", type=str)
args = vars(ap.parse_args())

if not args["gray"] or args["source"] is None:
    print("Switch --gray requires switch --source")
    sys.exit(1)

image_dir = args["image_dir"]
aperture_calibration_filename = args["aperture"]
target_image_filename = args["target"]

# Read in any Pantone image from pyimagesearch because
# we're going to replace the color in the aperture with
# the selected grayscale of the calibration image.
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
aperture_replacement_path = image_dir + aperture_calibration_filename
aperture_replacement_image = cv2.imread(aperture_replacement_path, cv2.IMREAD_COLOR)
if aperture_replacement_image is None:
    print('Aperture calibration image file not found')
    sys.exit(1)

# Test a color image with a single sample image against
# our grayscale calibration. Typically this will be the
# same full image of a single sample from which the
# aperture calibration image was extracted via Gimp.
# See the guidelines for more information.
target_image_path = image_dir + target_image_filename
target_image = cv2.imread(target_image_path, cv2.IMREAD_COLOR)
if target_image is None:
    print('Target image file not found')
    sys.exit(1)

# Now start processing.
# According to the command line argument --source convert
# the color calibration image to grayscale.
try:
    selected_grayscale_source = GrayscaleSource[args["source"]]
except KeyError:
  print("No such grayscale source")
  sys.exit(1)

aperture_grayscale = select_grayscale(aperture_replacement_image, selected_grayscale_source)

# Get the minimum and maximum grayscale levels from the aperture.
aperture_min_grayscale = np.min(aperture_grayscale)
aperture_max_grayscale = np.max(aperture_grayscale)
print("Aperture minimum grayscale " + str(aperture_min_grayscale) + ", maximum " + str(aperture_max_grayscale))

aperture_replacement_bgr = cv2.cvtColor(aperture_grayscale, cv2.COLOR_GRAY2BGR) # need BGR to merge with card image
card_image, aperture_mask = aperture.prepare_aperture(pantone_image, aperture_replacement_bgr, image_dir)

# Using the returned aperture mask, call calcHist to get the
# histogram of the grayscale behind the aperture.
card_grayscale = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)

# Set up for calcHist.
histSize = 256 # one bin for each OpenCV grayscale value 0 - 255
ranges = [0, 256]

# [0] is the channel = grayscale; mask is the aperture; [256] is the number of bins; [0, 256] is the range
hist = cv2.calcHist([card_grayscale], [0], aperture_mask, [histSize], ranges)

# The bin index and the grayscale value are the same because
# we've allocated 256 bins, one for each grayscale value.
dominant_bin_index = np.argmax(hist)
print("Dominant grayscale/bin " + str(dominant_bin_index))

plt.plot(hist)
plt.title('Histogram of selected grayscale channel')
plt.xlabel('Pixel Intensity')
plt.ylabel('Pixel Count')
plt.show()

# According to the command line argument --source convert
# the color full image target to grayscale.
target_grayscale = select_grayscale(target_image, selected_grayscale_source)

# For debugging get the minimum grayscale for the full image.
target_min_grayscale = np.min(target_grayscale)
target_max_grayscale = np.max(target_grayscale)
print("Target minimum grayscale " + str(target_min_grayscale) + ", maximum " + str(target_max_grayscale))

# It's better to start with a busy image and then gradually
# increase the grayscale threshold to reduce the number of
# objects. So if the minimum grayscale of the aperture is
# below the default then lower the default to create a busy
# image.
GRAYSCALE_LOW_DEFAULT = 125
grayscale_low = GRAYSCALE_LOW_DEFAULT

if aperture_min_grayscale < GRAYSCALE_LOW_DEFAULT:
    grayscale_low = MIN_GRAYSCALE_THRESHOLD

status, last_grayscale_threshold = ImageUtils.iterateThreshold(lambda control_variable: grayscale_threshold_wrapper(target_grayscale, control_variable), grayscale_low, ImageUtils.ThresholdControlDirection.INITIAL)

print("Final grayscale threshold level: " + str(last_grayscale_threshold))
final_thresholded_image = grayscale_threshold_wrapper(target_grayscale, last_grayscale_threshold)
if status:
    cv2.imshow("Final RotatedRect for sample at grayscale threshold", final_thresholded_image)
    cv2.waitKey(0)
else:
    print("Unable to determine threshold levels")
    cv2.imshow("Sample at time of error at grayscale threshold", final_thresholded_image)
    cv2.waitKey(0)