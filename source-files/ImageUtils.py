#################################################################
# ImageUtils.py
#################################################################

import cv2
import numpy as np
import sys
from enum import Enum

class ImageUtils:

    ## From FtcIntoTheDeepLimelight.ImageUtils; renamed from threshold_adjusted_hsv
    @staticmethod
    def apply_inRange(p_hsv_roi, hue_low, hue_high, sat_threshold_low, val_threshold_low):
        # Sanity check for hue.
        if not ((0 <= hue_low <= 180) and (0 <= hue_high <= 180)):
            raise Exception("Hue out of range")

        if hue_low < hue_high:  # Normal hue range.
            # Define lower and upper bounds in this way to avoid Python warnings.
            lower_bounds = np.array([hue_low, sat_threshold_low, val_threshold_low], dtype=np.uint8)
            upper_bounds = np.array([hue_high, 255, 255], dtype=np.uint8)
            thresholded = cv2.inRange(p_hsv_roi, lower_bounds, upper_bounds)
        else:
            # For a hue range from the XML file of low 170, high 10
            # the following yields two new ranges: 170 - 180 and 0 - 10.
            lower_bounds_1 = np.array([hue_low, sat_threshold_low, val_threshold_low])
            upper_bounds_1 = np.array([180, 255, 255])
            range1 = cv2.inRange(p_hsv_roi, lower_bounds_1, upper_bounds_1)

            lower_bounds_2 = np.array([0, sat_threshold_low, val_threshold_low])
            upper_bounds_2 = np.array([hue_high, 255, 255])
            range2 = cv2.inRange(p_hsv_roi, lower_bounds_2, upper_bounds_2)
            thresholded = cv2.bitwise_or(range1, range2)

        return thresholded

    @staticmethod
    def get_hue_range(p_hist, dominant_bin_index):
        # Log all non-zero histogram bins.
        min_pixel_count = np.min(p_hist)
        print("Minimum pixel count", min_pixel_count)
        for bin_index, count in enumerate(p_hist):
            if count[0] != min_pixel_count:
                print(f"Bin {bin_index}: {count[0]}")

        # Look at bins on each side of the dominant bin/hue
        # until you find one with the minimum pixel count,
        # typically 0. Be mindful of the wrap-around at 0/180.
        adjacent_bin = dominant_bin_index
        while True:
            adjacent_bin = adjacent_bin - 1
            if adjacent_bin == -1:
                adjacent_bin = 179  # crossed boundary at 0

            print("Bin " + str(adjacent_bin) + ", pixel count " + str(p_hist[adjacent_bin]))
            if p_hist[adjacent_bin] == min_pixel_count:
                print("Found minimum pixel count at bin " + str(adjacent_bin))
                hsv_hue_low = adjacent_bin
                break

        adjacent_bin = dominant_bin_index
        while True:
            adjacent_bin = adjacent_bin + 1
            if adjacent_bin == 180:
                adjacent_bin = 0  # crossed boundary at 179

            print("Bin " + str(adjacent_bin) + ", pixel count " + str(p_hist[adjacent_bin]))
            if p_hist[adjacent_bin] == min_pixel_count:
                print("Found minimum pixel count at bin " + str(adjacent_bin))
                hsv_hue_high = adjacent_bin
                break

        print("Hue low, high " + str(hsv_hue_low) + ", " + str(hsv_hue_high))
        return hsv_hue_low, hsv_hue_high

    # Based on - but not the same as - FtcIntoTheDeepLimelight.ImageUtils.
    @staticmethod
    def filter_contours(p_thresholded, image_height, image_width, min_area, max_area):
        contours, _ = cv2.findContours(p_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw on an all-black background; drawContours requires a BGR image.
        filtered_bgr = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        show_contour = np.zeros((image_height, image_width, 3), dtype=np.uint8)

        num_below_min_area = 0
        num_above_max_area = 0
        filtered_binary = [] # the goal is that this list should contain one element
        for i in range(len(contours)):
            contour_area = cv2.contourArea(contours[i])
            # oddly, some contours have zero area; test for closed contours
            # cv2.isContourConvex(contours[i]) missed a contour of 15686!
            if contour_area > 0.0:
                # Find the minimum area rectangle
                rect = cv2.minAreaRect(contours[i])
                center, dimensions, angle = rect
                rect_area = dimensions[0] * dimensions[1]
                print("Rotated rect area " + str(rect_area))
                if rect_area < min_area:
                    num_below_min_area = num_below_min_area + 1
                    continue

                if rect_area > max_area:
                    num_above_max_area = num_above_max_area + 1
                    continue

                # Got a rotated rectangle whose area is in range
                cv2.drawContours(show_contour, contours, i, (255, 255, 255), 2)
                #cv2.imshow("Found 1 contour ", show_contour)
                #cv2.waitKey(0)

                box = cv2.boxPoints(rect)
                box = np.int32(box)

                # Draw the rectangle
                cv2.drawContours(filtered_bgr, [box], 0, (255, 255, 255), cv2.FILLED)
                #cv2.imshow("One RotatedRect ", filtered_bgr)
                #cv2.waitKey(0)

                # Convert the BGR image to grayscale, which in our case should be binary.
                filtered_binary.append(cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2GRAY))

        return ImageUtils.FilteredContoursRecord(len(contours), num_below_min_area, num_above_max_area, filtered_binary)

    class FilteredContoursRecord:
        def __init__(self, num_unfiltered_contours, num_below_min_area, num_above_max_area, filtered_binary_output):
            self.numUnfilteredContours = num_unfiltered_contours
            self.numBelowMinArea = num_below_min_area
            self.numAboveMaxArea = num_above_max_area
            self.filtered_binary_output = filtered_binary_output

    # The direction in which the threshold variable will be modified.
    class ThresholdControlDirection(Enum):
        INITIAL = 0
        INCREMENT = 1
        DECREMENT = 2

    # Function that determines the correct thresholding for an image.
    # Since the body of the code is the same for HSV and grayscale,
    # the first parameter is a lambda that returns a thresholded
    # binary image: for HSV the lambda should call cv2.inRange(),
    # for grayscale it should call cv2.threshold(). For HSV the
    # caller should supply the saturation threshold as the control
    # variable; for grayscale the caller should supply the low
    # threshold value.

    ##**TODO Are there any cases where the thresholding starts
    # too high and we have to decrement?

    @staticmethod
    def iterateThreshold(threshold_image_function, control_variable, control_direction):
        CONTROL_VARIABLE_CHANGE = 5
        MIN_CONTROL_VARIABLE = 25
        MAX_CONTROL_VARIABLE = 250

        MIN_SAMPLE_AREA = 14000
        MAX_SAMPLE_AREA = 21000

        # Besides the target sample we'll allow a few contours below
        # the minimum area. These will be filtered out later.
        MAX_CONTOURS_BELOW_MIN_AREA = 10  # some may be zero length or not closed

        thresholded = threshold_image_function(control_variable)
        print("Current control variable " + str(control_variable))

        thr_non_zero_count = cv2.countNonZero(thresholded) # information
        print("Non-zero pixel count in thresholded image " + str(thr_non_zero_count))

        # cv2.imshow("Thresholded", thresholded)
        # cv2.waitKey(0)

        # Filter the contours and rotated rectangles.
        thr_height, thr_width = thresholded.shape
        filtered_contours = ImageUtils.filter_contours(thresholded, thr_height, thr_width, MIN_SAMPLE_AREA, MAX_SAMPLE_AREA)

        # The image is too sparse if:
        #   the total number of contours is 0
        #   the total number of rotated rectangles within the area range is 0
        # The image is too busy if:
        #   the total number of contours is above MAX_CONTOURS_BELOW_MIN_AREA
        #   the total number of rotated rectangles is > 1

        # Take the desired case first.
        if len(filtered_contours.filtered_binary_output) == 1 and filtered_contours.numBelowMinArea <= MAX_CONTOURS_BELOW_MIN_AREA and filtered_contours.numAboveMaxArea == 0:
            print("Final raw contour count " + str(filtered_contours.numUnfilteredContours))
            print("Final number of filtered contours with below minimum area " + str(filtered_contours.numBelowMinArea))
            return True, control_variable  # all good

        # If the thresholded image is too sparse then we need to
        # lower the control variable.
        if filtered_contours.numUnfilteredContours == 0:
            if control_direction == ImageUtils.ThresholdControlDirection.INCREMENT:
                print("Error: reversal of control direction from increment to decrement")
                return False, control_variable

            # INITIAL or DECREMENT
            next_control_variable = control_variable - CONTROL_VARIABLE_CHANGE
            if next_control_variable < MIN_CONTROL_VARIABLE:
                print("Error: below minimum control variable")
                return False, control_variable

            # call self
            next_control_direction = ImageUtils.ThresholdControlDirection.DECREMENT
            return ImageUtils.iterateThreshold(threshold_image_function, next_control_variable, next_control_direction)

        # If the thresholded image is too busy or we've found more than
        # one qualifying rectangle or an oversized blob, then we need to
        # raise the control variable.
        if filtered_contours.numBelowMinArea > MAX_CONTOURS_BELOW_MIN_AREA or len(
                filtered_contours.filtered_binary_output) > 1 or filtered_contours.numAboveMaxArea != 0:
            if control_direction == ImageUtils.ThresholdControlDirection.DECREMENT:
                print("Error: reversal of control direction from decrement to increment")
                return False, control_variable

            # INITIAL or INCREMENT
            next_control_variable = control_variable + CONTROL_VARIABLE_CHANGE
            if next_control_variable > MAX_CONTROL_VARIABLE:
                print("Error: above maximum control variable")
                return False, control_variable

            # call self
            next_control_direction = ImageUtils.ThresholdControlDirection.INCREMENT
            return ImageUtils.iterateThreshold(threshold_image_function, next_control_variable, next_control_direction)

        # failsafe
        print("Unhandled condition in iterateThreshold")
        sys.exit(1)  # OR raise exception
