#################################################################
# ImageUtils.py
#################################################################

import cv2
import numpy as np


# Python version of ImageUtils.java from the IntelliJ
# project IJIntoTheDeepVision.
class ImageUtils:

    ## From FtcIntoTheDeepLimelight.ImageUtils; renamed from threshold_adjusted_hsv
    @staticmethod
    def threshold_hsv(p_hsv_roi, hue_low, hue_high, sat_threshold_low, val_threshold_low):
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

        return FilteredContoursRecord(len(contours), num_below_min_area, num_above_max_area, filtered_binary)

class FilteredContoursRecord:
    def __init__(self, num_unfiltered_contours, num_below_min_area, num_above_max_area, filtered_binary_output):
        self.numUnfilteredContours = num_unfiltered_contours
        self.numBelowMinArea = num_below_min_area
        self.numAboveMaxArea = num_above_max_area
        self.filtered_binary_output = filtered_binary_output