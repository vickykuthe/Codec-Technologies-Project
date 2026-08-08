import cv2
import numpy as np
from PIL import Image


def resize_image(image, width=28, height=28):
    """Resize digit image to 28 x 28."""
    
    pil_image = Image.fromarray(image)
    resized = pil_image.resize((width, height))
    
    return np.asarray(resized)


def preprocess_image(image):
    """Preprocess image for digit detection."""

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    # Same thresholding approach as original project
    _, threshold = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY
    )

    threshold = cv2.erode(
        threshold,
        kernel,
        iterations=1
    )

    threshold = cv2.dilate(
        threshold,
        kernel,
        iterations=1
    )

    threshold = cv2.erode(
        threshold,
        kernel,
        iterations=1
    )

    return gray, threshold


def find_digit_contours(threshold):
    """
    Detect only the outer digit contours.
    This prevents holes inside 6, 8, 9 and 0
    from being treated as separate digits.
    """

    contours, hierarchy = cv2.findContours(
        threshold,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]

    bounding_rectangles = [
        cv2.boundingRect(contour)
        for contour in contours
    ]

    # Find most common hierarchy level
    unique, indices = np.unique(
        hierarchy[:, -1],
        return_inverse=True
    )

    most_common_hierarchy = unique[
        np.argmax(
            np.bincount(indices)
        )
    ]

    final_rectangles = []

    for rectangle, h in zip(
        bounding_rectangles,
        hierarchy
    ):

        x, y, w, height = rectangle

        if (
            (w * height) > 250
            and 10 <= w <= 200
            and 10 <= height <= 200
            and h[3] == most_common_hierarchy
        ):
            final_rectangles.append(
                rectangle
            )

    return final_rectangles


def sort_rectangles(
    rectangles,
    image_width
):
    """
    Sort digits row-wise.
    """

    return sorted(
        rectangles,
        key=lambda r:
        r[1] * image_width + r[0]
    )


def extract_digit(
    gray,
    rectangle
):
    """
    Extract one digit and resize it.
    """

    x, y, w, h = rectangle

    digit = gray[
        y:y + h,
        x:x + w
    ]

    # Convert black digit to white
    digit = 255 - digit

    # Resize to same size used during training
    digit = resize_image(
        digit,
        28,
        28
    )

    return digit