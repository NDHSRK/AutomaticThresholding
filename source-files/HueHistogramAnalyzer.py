
# Stand-alone hue range tester ...

# The assumption behind the Pantone aperture code is
# that we can determine the hue range of the entire
# sample from the portion of the sample visible
# through the aperture. This is true in many cases
# but not, for example, with the C920 image
# front_webcam_01131510_53687_IMG.png. The hue
# range in the aperture is 103 to 109 while the
# range of the full sample is more like 100 to 115.
# We need a separate tester the verifiy this.
# Note that the range of 110, 115 correctly detects
# the blue sample in front_webcam_01131443_18886_IMG.png,
# which contains samples of all three colors.

import numpy as np
import argparse
import cv2
import sys
from enum import Enum
from ImageUtils import ImageUtils
import matplotlib.pyplot as plt

# construct the argument parser and parse the arguments
## main ...
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--image_dir", type=str)
ap.add_argument("--image", type=str)
args = vars(ap.parse_args())

sample_image_filename = "front_webcam_01131510_53687_BLUE.png"

image_dir = args["image_dir"]
sample_image_path = image_dir + args["image"]
sample_image = cv2.imread(sample_image_path, cv2.IMREAD_COLOR)
if sample_image is None:
    print('Sample image file not found')
    sys.exit(1)

card_hsv = cv2.cvtColor(sample_image, cv2.COLOR_BGR2HSV)
histSize = 180 # one bin for each OpenCV HSV hue value
ranges = [0, 180]

# [0] is the channel = hue; null mask for the full image; [180] is the number of bins; [0, 180] is the range
hist = cv2.calcHist([card_hsv], [0], None, [histSize], ranges)

# The bin index and the hue are the same because we've allocated 180 bins,
# one for each hue.
dominant_bin_index = np.argmax(hist)
print("Dominant hue/bin " + str(dominant_bin_index))

plt.plot(hist)
plt.title('Histogram of selected channel')
plt.xlabel('Pixel Intensity')
plt.ylabel('Pixel Count')
plt.show()
