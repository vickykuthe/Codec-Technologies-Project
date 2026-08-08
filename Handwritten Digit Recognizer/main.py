import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from config import (
    TRAIN_IMAGE,
    TEST_IMAGE,
    TRAINING_OUTPUT,
    PREDICTION_OUTPUT,
    CLEAN_OUTPUT,
    KNN_K
)

from image_processing import (
    preprocess_image,
    find_digit_contours,
    sort_rectangles,
    extract_digit
)

from feature_extraction import (
    extract_hog_features
)

from model import KNNModel


# ==========================================
# LOAD TRAINING DATA
# ==========================================

def load_training_data():

    print("\nLoading training image...")

    image = cv2.imread(
        TRAIN_IMAGE
    )

    if image is None:
        raise FileNotFoundError(
            f"Training image not found: {TRAIN_IMAGE}"
        )

    gray, threshold = preprocess_image(
        image
    )

    rectangles = find_digit_contours(
        threshold
    )

    rectangles = sort_rectangles(
        rectangles,
        image.shape[1]
    )

    print(
        "Detected training digits:",
        len(rectangles)
    )

    training_digits = []
    labels = []

    # ======================================
    # IMPORTANT:
    # Training sheet is:
    # 1,2,3,4,5,6,7,8,9,0
    # ======================================

    for index, rectangle in enumerate(
        rectangles
    ):

        digit = extract_digit(
            gray,
            rectangle
        )

        training_digits.append(
            digit
        )

        # Correct label
        label = (index // 10 + 1) % 10

        labels.append(label)

    # Draw training rectangles
    training_preview = image.copy()

    for index, rectangle in enumerate(
        rectangles
    ):

        x, y, w, h = rectangle

        label = (index // 10 + 1) % 10

        cv2.rectangle(
            training_preview,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            training_preview,
            str(label),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

    cv2.imwrite(
        TRAINING_OUTPUT,
        training_preview
    )

    return (
        np.asarray(training_digits),
        np.asarray(labels)
    )


# ==========================================
# TRAIN MODEL
# ==========================================

def train_model():

    digits, labels = (
        load_training_data()
    )

    if len(digits) == 0:
        raise RuntimeError(
            "No training digits detected."
        )

    print(
        "\nTraining images:",
        len(digits)
    )

    print(
        "Labels:",
        len(labels)
    )

    print(
        "Classes:",
        np.unique(labels)
    )

    # Shuffle
    digits, labels = shuffle(
        digits,
        labels,
        random_state=42
    )

    # ======================================
    # HOG FEATURE EXTRACTION
    # ======================================

    print(
        "\nExtracting HOG features..."
    )

    features = (
        extract_hog_features(
            digits
        )
    )

    print(
        "Feature shape:",
        features.shape
    )

    # ======================================
    # TRAIN / TEST SPLIT
    # ======================================

    X_train, X_test, y_train, y_test = (
        train_test_split(
            features,
            labels,
            test_size=0.25,
            random_state=42,
            stratify=labels
        )
    )

    # ======================================
    # KNN
    # ======================================

    print(
        "\nTraining KNN model..."
    )

    model = KNNModel(
        k=3
    )

    model.train(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"\nValidation Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # ======================================
    # FINAL MODEL
    # ======================================

    print(
        "\nTraining final model using "
        "all training images..."
    )

    final_model = KNNModel(
        k=3
    )

    final_model.train(
        features,
        labels
    )

    return final_model


# ==========================================
# RECOGNIZE TEST IMAGE
# ==========================================

def recognize_digits(model):

    print(
        "\nLoading test image..."
    )

    image = cv2.imread(
        TEST_IMAGE
    )

    if image is None:
        raise FileNotFoundError(
            f"Test image not found: {TEST_IMAGE}"
        )

    original = image.copy()

    gray, threshold = (
        preprocess_image(
            image
        )
    )

    rectangles = (
        find_digit_contours(
            threshold
        )
    )

    rectangles = sort_rectangles(
        rectangles,
        image.shape[1]
    )

    print(
        "Detected test digits:",
        len(rectangles)
    )

    # White output image
    clean_image = np.ones_like(
        image
    ) * 255

    recognized_digits = []

    for rectangle in rectangles:

        x, y, w, h = rectangle

        # Extract digit
        digit = extract_digit(
            gray,
            rectangle
        )

        # HOG features
        features = (
            extract_hog_features(
                [digit]
            )
        )

        # Prediction
        prediction = model.predict(
            features
        )

        digit_value = int(
            prediction[0]
        )

        recognized_digits.append(
            digit_value
        )

        # Bounding box
        cv2.rectangle(
            original,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Prediction text
        cv2.putText(
            original,
            str(digit_value),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 0, 0),
            3
        )

        # Clean result
        cv2.putText(
            clean_image,
            str(digit_value),
            (x, y + h),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 0, 0),
            5
        )

    # Save
    cv2.imwrite(
        PREDICTION_OUTPUT,
        original
    )

    cv2.imwrite(
        CLEAN_OUTPUT,
        clean_image
    )

    print(
        "\n===================================="
    )

    print(
        "RECOGNIZED DIGITS:"
    )

    print(
        recognized_digits
    )

    print(
        "===================================="
    )

    print(
        "\nPrediction image:"
    )

    print(
        PREDICTION_OUTPUT
    )

    print(
        "\nClean result:"
    )

    print(
        CLEAN_OUTPUT
    )


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 50)

    print(
        "CUSTOM HANDWRITTEN DIGIT RECOGNITION"
    )

    print("=" * 50)

    os.makedirs(
        "output",
        exist_ok=True
    )

    # Train
    model = train_model()

    # Predict
    recognize_digits(
        model
    )

    print(
        "\nPROJECT COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()