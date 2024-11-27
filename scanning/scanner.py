import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import easyocr
from PIL import Image

reader = easyocr.Reader(['pl'])

def extract_board_image(path):
    img = cv.imread(path)
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    blurred = cv.GaussianBlur(gray_image, (7, 7), 0)

    median_val = np.median(blurred)
    lower = int(max(0, 0.66 * median_val))
    upper = int(min(255, 1.33 * median_val))
    # Step 2: Edge Detection using Canny
    edges = cv.Canny(blurred, lower, upper)  # Tune thresholds if needed

    kernel = np.ones((5, 5), np.uint8)
    edges = cv.dilate(edges, kernel, iterations=1)

    # Step 3: Find Contours
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Step 4: Find the largest contour that resembles a rectangle
    scrabble_board_contour = None
    max_area = 0
    for contour in contours:
        # Approximate the contour
        epsilon = 0.04 * cv.arcLength(contour, True)
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
    # return image
    # Define a color range for the white-ish tiles (tweak these values if needed)
    lower_white = np.array([150, 150, 150])  # Lower bound for white in HSV
    upper_white = np.array([255, 255, 255])  # Upper bound for white in HSV

    # Create a mask for the white-ish areas
    mask = cv.inRange(image, lower_white, upper_white)

    # Bitwise AND the mask with the original image to isolate white areas
    isolated = cv.bitwise_and(image, image, mask=mask)
    return isolated


def extract_letters_from_board(image):
    height, width, channels = image.shape

    # Divide into 15x15 grid
    grid_h = height // 15
    grid_w = width // 15
    letter_matrix = []

    for row in range(15):
        row_letters = []
        for col in range(15):
            # Extract the tile region
            x_start, y_start = col * grid_w, row * grid_h
            tile_raw = image[y_start:y_start + grid_h, x_start:x_start + grid_w]
            tile = enhance_tile(tile_raw)

            # Preprocess tile (binarization to improve OCR accuracy)
            # _, tile_thresh = cv.threshold(tile, 150, 255, cv.THRESH_BINARY_INV)

            text = e_easyocr(tile)
            row_letters.append(text)



            # Optional: Debugging visualization
            # plt.imshow(tile, cmap='gray')
            # plt.title(f"Row: {row}, Col: {col}, Letter: {result}")
            # plt.axis('off')
            # plt.show()

        letter_matrix.append(row_letters)

    return letter_matrix

def enhance_tile(tile):
    gray_image = cv.cvtColor(tile, cv.COLOR_BGR2GRAY)
    gray_np = np.array(gray_image)

    _, binary_image = cv.threshold(gray_np, 150, 255, cv.THRESH_BINARY)

    resized_image = cv.resize(binary_image, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    sharpened_image = cv.filter2D(resized_image, -1, kernel)

    return sharpened_image


def e_easyocr(tile):
    result = reader.readtext(tile, allowlist='AĄBCĆDEĘFGHIJKLŁMNŃOÓPRSŚTUVWXYZŻŹ')
    if len(result) > 0:
        first = result[0]
        text, confidence = first[1], first[2]
        if text and len(text) == 1 and confidence > 0.8:
            return text[0]
    else:
        return None


def scan(path):
    matrix = extract_letters_from_board(apply_mask(extract_board_image(path)))
    for row in matrix:
        print(row)
    return matrix

