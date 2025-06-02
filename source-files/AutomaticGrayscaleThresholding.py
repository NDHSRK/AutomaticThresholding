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
import perspective
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
# the selected grayscale source of calibration image.

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

# Test this image against our grayscale calibration.
target_image_path = image_dir + target_image_filename
target_image = cv2.imread(target_image_path, cv2.IMREAD_COLOR)
if target_image is None:
    print('Target image file not found')
    sys.exit(1)

# From the command line argument --source get a grayscale
# image from the selected source - for both the aperture
# and the full image.
try:
    selected_grayscale_source = GrayscaleSource[args["source"]]
except KeyError:
  print("No such grayscale source")
  sys.exit(1)

aperture_grayscale = select_grayscale(aperture_calibration_image, selected_grayscale_source)

##**TODO Some of this replicates the color path
# The Pantone card image is 3024x4032. Use the scaling factor
# from pyimagesearch.
SCALED_CARD_WIDTH = 600
SCALED_CARD_HEIGHT = 600
SCALED_CARD_SIZE = (SCALED_CARD_WIDTH, SCALED_CARD_HEIGHT)
pantone_image = cv2.resize(pantone_image, SCALED_CARD_SIZE, interpolation=cv2.INTER_AREA)

# Find the color matching card in the input image
print("Finding color matching card ...")
imageCard = perspective.find_color_card(pantone_image)

# If the color matching card is not found, just exit.
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

# Get the minimum and maximum grayscale levels from the aperture.
aperture_min_grayscale = np.min(aperture_grayscale)
aperture_max_grayscale = np.max(aperture_grayscale)
print("Aperture minimum grayscale " + str(aperture_min_grayscale) + ", maximum " + str(aperture_max_grayscale))

# Legacy images only: replace the color in the aperture with the grayscale
# from the calibration image, which is also 85x60.
aperture_bgr = cv2.cvtColor(aperture_grayscale, cv2.COLOR_GRAY2BGR)
imageCard[aperture_y1: aperture_y2, aperture_x1: aperture_x2] = aperture_bgr

# Write and show the color matching card adjusted for perspective.
cv2.imwrite(image_dir + "Pantone_01_aruco_.png", imageCard)
cv2.imshow("Pantone card with calibration aperture", imageCard)
cv2.waitKey(0)

# Create a mask for the histogram by drawing a filled rectangle
# over the aperture.
aperture_mask = np.zeros((height, width), dtype=np.uint8)
cv2.rectangle(aperture_mask, (aperture_x1, aperture_y1), (aperture_x2, aperture_y2), 255, cv2.FILLED)

# Get the histogram of the grayscale behind the aperture.
card_grayscale = cv2.cvtColor(imageCard, cv2.COLOR_BGR2GRAY)
histSize = 256 # one bin for each OpenCV grayscale value
ranges = [0, 256]

# [0] is the channel = hue; mask is the aperture; [180] is the number of bins; [0, 180] is the range
hist = cv2.calcHist([card_grayscale], [0], aperture_mask, [histSize], ranges)

# The bin index and the hue are the same because we've allocated 256 bins,
# one for each grayscale value.
dominant_bin_index = np.argmax(hist)
print("Dominant grayscale/bin " + str(dominant_bin_index))

plt.plot(hist)
plt.title('Histogram of selected grayscale channel')
plt.xlabel('Pixel Intensity')
plt.ylabel('Pixel Count')
plt.show()

target_grayscale = select_grayscale(target_image, selected_grayscale_source)

# For debugging get the minimum grayscale for the full image.
target_min_grayscale = np.min(target_grayscale)
target_max_grayscale = np.max(target_grayscale)
print("Target minimum grayscale " + str(target_min_grayscale) + ", maximum " + str(target_max_grayscale))

##**TODO Are there any cases where the thresholding starts
# too high and we have to decrement?

# It's better to start with a busy image and then gradually
# increase the saturation to reduce the number of objects.
# So if the minimum saturation of the image is below the
# default then lower the default to create a busy image.
GRAYSCALE_LOW_DEFAULT = 125
grayscale_low = GRAYSCALE_LOW_DEFAULT

if aperture_min_grayscale < GRAYSCALE_LOW_DEFAULT:
    grayscale_low = MIN_GRAYSCALE_THRESHOLD

MIN_SAMPLE_AREA = 14000
MAX_SAMPLE_AREA = 21000

status, last_grayscale_threshold = ImageUtils.iterateThreshold(lambda control_variable: grayscale_threshold_wrapper(target_grayscale, control_variable), grayscale_low, ImageUtils.ThresholdControlDirection.INITIAL,
                                                    MIN_SAMPLE_AREA, MAX_SAMPLE_AREA)

print("Final grayscale threshold level: " + str(last_grayscale_threshold))
final_thresholded_image = grayscale_threshold_wrapper(target_grayscale, last_grayscale_threshold)
if status:
    cv2.imshow("Final RotatedRect for sample at grayscale threshold", final_thresholded_image)
    cv2.waitKey(0)
else:
    print("Unable to determine threshold levels")
    cv2.imshow("Sample at time of error at grayscale threshold", final_thresholded_image)
    cv2.waitKey(0)