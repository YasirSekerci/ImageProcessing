import cv2
import numpy as np

def to_grayscale(image):
    # Validate the input image
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel BGR image")

    # Extract the blue, green, and red channels
    B = image[:, :, 0].astype(np.float32)
    G = image[:, :, 1].astype(np.float32)
    R = image[:, :, 2].astype(np.float32)

    # Apply the luminosity method to calculate grayscale
    # Grayscale = 0.2989 * R + 0.5870 * G + 0.1140 * B
    gray = 0.2989 * R + 0.5870 * G + 0.1140 * B

    # Clip the values to 0-255 and convert to uint8
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray


def resize_image(image, new_width, new_height):
    # Validate input image
    if len(image.shape) != 2:
        raise ValueError("Input image must be a 2D grayscale image")

    # Get the original dimensions
    orig_height, orig_width = image.shape

    # Check for empty images
    if orig_height == 0 or orig_width == 0:
        raise ValueError("Input image is empty or has invalid dimensions")

    # Create an empty output image with the new dimensions
    resized_image = np.zeros((new_height, new_width), dtype=np.uint8)

    # Calculate the scaling factors
    scale_x = orig_width / new_width
    scale_y = orig_height / new_height

    # Iterate over each pixel in the output image
    for y in range(new_height):
        for x in range(new_width):
            # Map the coordinates back to the original image
            orig_x = min(int(x * scale_x), orig_width - 1)  # Ensure within bounds
            orig_y = min(int(y * scale_y), orig_height - 1)  # Ensure within bounds

            # Assign the nearest pixel value
            resized_image[y, x] = image[orig_y, orig_x]

    return resized_image
    

def calculate_histogram(image, bins=256, range_min=0, range_max=256):
    # Validate the input
    if len(image.shape) != 2:
        raise ValueError("Input image must be a 2D grayscale image.")

    # Initialize histogram with zeros
    histogram = np.zeros(bins, dtype=np.int32)

    # Calculate the bin width based on the range
    bin_width = (range_max - range_min) / bins

    # Iterate through each pixel and increment the corresponding bin
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            pixel_value = image[y, x]
            if range_min <= pixel_value < range_max:
                bin_index = int((pixel_value - range_min) / bin_width)
                histogram[bin_index] += 1

    # Normalize the histogram to make it comparable (convert to float32)
    histogram = histogram.astype(np.float32)
    histogram /= (histogram.sum() + 1e-7)  # Avoid division by zero

    return histogram


def compare_histograms(hist1, hist2):
    # Validate input histograms
    if hist1.shape != hist2.shape:
        raise ValueError("Histograms must have the same shape")

    # Compute the mean of each histogram
    mean1 = np.mean(hist1)
    mean2 = np.mean(hist2)

    # Subtract the means to center the histograms
    centered_hist1 = hist1 - mean1
    centered_hist2 = hist2 - mean2

    # Compute the numerator: sum of element-wise product of centered histograms
    numerator = np.sum(centered_hist1 * centered_hist2)

    # Compute the denominator: product of the Euclidean norms of centered histograms
    denominator = np.sqrt(np.sum(centered_hist1**2) * np.sum(centered_hist2**2))

    # Avoid division by zero
    if denominator == 0:
        return 0.0

    # Return the correlation score
    return numerator / denominator


def multi_scale_template_matching(frame_gray, template, scales):
    best_match_val = -1
    best_match_loc = None
    best_match_scale = None

    for scale in scales:
        # Calculate the new width and height based on the scale
        new_width = int(template.shape[1] * scale)  # scale * original width
        new_height = int(template.shape[0] * scale)  # scale * original height

        # Use the custom resize function
        resized_template = resize_image(template, new_width, new_height)

        if resized_template.shape[0] > frame_gray.shape[0] or resized_template.shape[1] > frame_gray.shape[1]:
            continue

        # Perform template matching
        result = cv2.matchTemplate(frame_gray, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_match_val:
            best_match_val = max_val
            best_match_loc = max_loc
            best_match_scale = scale
            best_template_size = resized_template.shape

    return best_match_val, best_match_loc, best_match_scale, best_template_size


def filter_eye_matches(left_eye, right_eye, min_distance=30):
    if left_eye and right_eye:
        left_x = left_eye[0][0] + left_eye[2][1] // 2
        right_x = right_eye[0][0] + right_eye[2][1] // 2

        # Check horizontal distance
        if abs(right_x - left_x) < min_distance:
            # If eyes are too close, invalidate one of them
            if left_eye[0][0] < right_eye[0][0]:
                right_eye = None  # Remove the right eye if too close
            else:
                left_eye = None  # Remove the left eye if too close

    return left_eye, right_eye


def face_detection(image, templates, threshold=0.6):
    # Scaling factors to account for different object sizes
    scales = [0.75, 0.5, 0.25]
    face_boxes = []

    # Dictionary to hold best matches for each template
    best_matches = {}

    # Perform template matching for each body part
    for part_name in ['left_eye', 'right_eye', 'mouth']:
        template = templates[part_name]
        match_val, match_loc, match_scale, match_size = multi_scale_template_matching(image, template, scales)
        if match_val > threshold:
            best_matches[part_name] = (match_loc, match_scale, match_size)

    # Filter eye matches to ensure they are apart
    left_eye = best_matches.get('left_eye')
    right_eye = best_matches.get('right_eye')
    left_eye, right_eye = filter_eye_matches(left_eye, right_eye, min_distance=30)

    # Calculate face bounding box if at least two features are detected
    if left_eye and right_eye:
        left_eye_center = (left_eye[0][0] + left_eye[2][1] // 2, left_eye[0][1] + left_eye[2][0] // 2)
        right_eye_center = (right_eye[0][0] + right_eye[2][1] // 2, right_eye[0][1] + right_eye[2][0] // 2)
        eye_width = abs(right_eye_center[0] - left_eye_center[0])
        face_width = int(eye_width * 2.2)
        face_height = int(eye_width * 2.7)
        x_min = int(min(left_eye_center[0], right_eye_center[0]) - eye_width // 2)
        y_min = int(min(left_eye_center[1], right_eye_center[1]) - eye_width)
        face_boxes.append((x_min, y_min, face_width, face_height))

    elif len(best_matches) == 2:  # Use mouth and one eye to approximate
        keypoints = list(best_matches.values())
        x_coords = [kp[0][0] for kp in keypoints]
        y_coords = [kp[0][1] for kp in keypoints]
        w = max(kp[2][1] for kp in keypoints) * 2
        h = w * 1.5
        x_min = int(min(x_coords))
        y_min = int(min(y_coords))
        face_boxes.append((x_min, y_min, int(w), int(h)))

    return face_boxes

