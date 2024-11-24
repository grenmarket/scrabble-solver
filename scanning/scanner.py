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
        return cv.warpPerspective(gray_image, M, (maxWidth, maxHeight))
    else:
        print("No Scrabble board detected.")


def extract_letters_from_board(warped_image):
    # Resize to ensure consistency (e.g., 750x750 pixels for a 15x15 grid)
    board_size = 750
    resized = cv.resize(warped_image, (board_size, board_size))

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
            _, tile_thresh = cv.threshold(tile, 150, 255, cv.THRESH_BINARY_INV)

            # Use OCR to recognize the letter
            letter = pytesseract.image_to_string(
                tile_thresh,
                lang='pol',
                config='--psm 10 -c tessedit_char_whitelist=AĄBCĆDEĘFGHIJKLŁMNŃOÓPRSŚTUVWXYZŻŹ'
            ).strip()

            # Append the detected letter (or None if no letter detected)
            row_letters.append(letter[0] if letter else None)

            # Optional: Visualize each tile being processed
            # plt.imshow(tile_thresh, cmap='gray')
            # plt.title(f'Row: {row}, Col: {col}, Letter: {letter}')
            # plt.show()

        letter_matrix.append(row_letters)

    return letter_matrix


img = extract_board_image('1.jpeg')
print(extract_letters_from_board(img))
