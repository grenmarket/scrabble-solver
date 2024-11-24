import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pytesseract


def extract_board_image(path):
    raw_img = cv.imread(path)
    img = cv.resize(raw_img, None, fx=0.25, fy=0.25)
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    blurred = cv.GaussianBlur(gray_image, (5, 5), 0)

    # Step 2: Edge Detection using Canny
    edges = cv.Canny(blurred, 50, 150)  # Tune thresholds if needed

    # Step 3: Find Contours
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Step 4: Find the largest contour that resembles a rectangle
    scrabble_board_contour = None
    max_area = 0
    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)

        # Check if it has 4 points and a large enough area
        if len(approx) == 4 and cv.contourArea(approx) > max_area:
            scrabble_board_contour = approx
            max_area = cv.contourArea(approx)

    # Step 5: If a contour is found, apply a perspective transform
    if scrabble_board_contour is not None:
        # Sort points to ensure correct order (top-left, top-right, bottom-right, bottom-left)
        def order_points(pts):
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]  # Top-left
            rect[2] = pts[np.argmax(s)]  # Bottom-right
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]  # Top-right
            rect[3] = pts[np.argmax(diff)]  # Bottom-left
            return rect

        points = scrabble_board_contour.reshape(4, 2)
        rect = order_points(points)

        # Compute the width and height of the new board
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        # Destination points for the top-down view
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        # Compute the perspective transform matrix and apply it
        M = cv.getPerspectiveTransform(rect, dst)
        warped = cv.warpPerspective(img, M, (maxWidth, maxHeight))
        return warped
    else:
        print("No Scrabble board detected.")

def apply_mask(image):
    # Define a color range for the white-ish tiles (tweak these values if needed)
    lower_white = np.array([210, 210, 210])  # Lower bound for white in HSV
    upper_white = np.array([255, 255, 255])  # Upper bound for white in HSV

    # Create a mask for the white-ish areas
    mask = cv.inRange(image, lower_white, upper_white)

    # Bitwise AND the mask with the original image to isolate white areas
    isolated = cv.bitwise_and(image, image, mask=mask)
    return isolated


def extract_letters_from_board(image):
    # Resize to ensure consistency (e.g., 750x750 pixels for a 15x15 grid)
    board_size = 750
    resized = cv.resize(image, (board_size, board_size))

    # Divide into 15x15 grid
    grid_size = board_size // 15  # Size of one tile (e.g., 50x50 pixels)
    letter_matrix = []

    for row in range(15):
        row_letters = []
        for col in range(15):
            # Extract the tile region
            x_start, y_start = col * grid_size, row * grid_size
            tile = resized[y_start:y_start + grid_size, x_start:x_start + grid_size]

            # Preprocess tile (binarization to improve OCR accuracy)
            # _, tile_thresh = cv.threshold(tile, 150, 255, cv.THRESH_BINARY_INV)

            # Use OCR with confidence scores
            ocr_data = pytesseract.image_to_data(
                tile,
                lang='pol',
                config='--psm 10 -c tessedit_char_whitelist=AĄBCĆDEĘFGHIJKLŁMNŃOÓPRSŚTUVWXYZŻŹ',
                output_type=pytesseract.Output.DICT
            )

            # Extract the letter with the highest confidence
            text = None
            max_confidence = 0

            for i, conf in enumerate(ocr_data['conf']):
                if conf != '-1':  # Exclude empty results
                    conf = int(conf)
                    if conf > max_confidence:
                        max_confidence = conf
                        text = ocr_data['text'][i]

            # Only accept the letter if confidence is above a threshold (e.g., 70)
            if text and max_confidence > 20:
                row_letters.append(text.strip()[0])  # Take the first character if not empty
            else:
                row_letters.append('.')  # No valid letter detected for this tile

            # Optional: Debugging visualization
            # print(f"Row: {row}, Col: {col}, Letter: {text}, Confidence: {max_confidence}")
            # plt.imshow(tile, cmap='gray')
            # plt.title(f"Row: {row}, Col: {col}, Letter: {text}, Confidence: {max_confidence}")
            # plt.show()

        letter_matrix.append(row_letters)

    return letter_matrix

image_path = "/home/s/BW.png"
image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

# Step 1: Enhance Contrast
image_contrast = cv.convertScaleAbs(image, alpha=1.5, beta=0)

# Step 2: Remove Grid Lines
# Use morphological operations to remove thin lines
kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
no_grid = cv.morphologyEx(image_contrast, cv.MORPH_OPEN, kernel)

# Step 3: Thresholding
_, binary_image = cv.threshold(no_grid, 120, 255, cv.THRESH_BINARY)

# Step 4: Refine Characters with Dilation
# Dilate the characters to make them bolder
kernel_dilate = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
dilated = cv.dilate(binary_image, kernel_dilate, iterations=1)

plt.imshow(dilated)
plt.show()

# matrix = extract_letters_from_board(dilated)
# for row in matrix:
#     print(row)