import numpy as np
from skimage.feature import hog


def extract_hog_features(images):
    """
    Extract HOG (Histogram of Oriented Gradients)
    features from digit images.
    """

    features = []

    for image in images:

        # Make sure image is grayscale
        if len(image.shape) == 3:
            image = image[:, :, 0]

        hog_features = hog(
            image,
            orientations=10,
            pixels_per_cell=(5, 5),
            cells_per_block=(1, 1),
            block_norm="L2-Hys"
        )

        features.append(hog_features)

    return np.asarray(
        features,
        dtype=np.float32
    )