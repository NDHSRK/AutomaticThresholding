import numpy as np
import cv2
import sys
import perspective

def prepare_aperture(p_pantone_image, p_aperture_calibration_image, p_aperture_image_replacement, p_image_dir):
    # The Pantone card image is 3024x4032. Use the scaling factor
    # from pyimagesearch.
    SCALED_CARD_WIDTH = 600
    SCALED_CARD_HEIGHT = 600
    SCALED_CARD_SIZE = (SCALED_CARD_WIDTH, SCALED_CARD_HEIGHT)
    pantone_image = cv2.resize(p_pantone_image, SCALED_CARD_SIZE, interpolation=cv2.INTER_AREA)

    # Find the color matching card in the input image
    print("Finding color matching card ...")
    card_image = perspective.find_color_card(pantone_image)

    # If the color matching card is not found, just exit.
    if card_image is None:
        print("Could not find color matching card")
        sys.exit(0)

    # After the Perspective Transform the ArUco markers are
    # aligned with the edges of the image. The center of the
    # aperture is at the center of the image.

    ##**TODO Will these numbers hold even if the camera is at a
    # different angle?

    # With the image resized to 600x600 the size of the transformed
    # image of the card only (in Gimp) is 339x314 and the size of
    # the aperture is width 85 x height 60
    APERTURE_WIDTH = 85
    APERTURE_HEIGHT = 60

    # Get the center point of the card and define the aperture.
    height, width, channels = card_image.shape
    card_center_x = width / 2.0
    card_center_y = height / 2.0
    aperture_x1 = int(card_center_x - (APERTURE_WIDTH / 2.0))
    aperture_y1 = int(card_center_y - (APERTURE_HEIGHT / 2.0)) # upper left
    aperture_x2 = int(card_center_x + (APERTURE_WIDTH / 2.0))
    aperture_y2 = int(card_center_y + (APERTURE_HEIGHT / 2.0)) # lower right

    # The aperture calibration image must have the same dimensions
    # as the aperture in the calibration card.
    ach, acw, _ = p_aperture_calibration_image.shape
    if ach != APERTURE_HEIGHT or acw != APERTURE_WIDTH:
        print("Aperture calibration image size does not match that of the card aperture")
        sys.exit(0)

    card_image[aperture_y1: aperture_y2, aperture_x1: aperture_x2] = p_aperture_image_replacement

    # Write and show the color matching card adjusted for perspective.
    cv2.imwrite(p_image_dir + "Pantone_01_aruco_.png", card_image)
    cv2.imshow("Pantone card with calibration aperture", card_image)
    cv2.waitKey(0)

    # Create a mask for the histogram by drawing a filled rectangle
    # over the aperture.
    aperture_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(aperture_mask, (aperture_x1, aperture_y1), (aperture_x2, aperture_y2), 255, cv2.FILLED)

    return card_image, aperture_mask