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


def gaussian_filter(image, kernel_size = 3, sigma = 1):
    sigma_squared = sigma ** 2
    pi = math.pi
    e_value = math.e
    center = kernel_size // 2
    gaussian_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    for y in range(-center, center + 1):
        for x in range(-center, center + 1):
            Gxy = (1 / (2 * pi * sigma_squared)) * (e_value ** ( -((x ** 2 + y ** 2) / (2 * sigma_squared))))
            gaussian_kernel[y + center][x + center] = Gxy

    
    gaussian_kernel /= gaussian_kernel.sum()

    height, width = image.shape
    blurred_image = np.zeros((height, width), dtype=np.uint8)
    
    padded_image = np.zeros((height + center * 2, width + center * 2), dtype=np.uint8)

    padded_image = np.pad(image, pad_width=center, mode='edge')
#    for row in range(0, height + 2, 1):
#        for column in range(0, width + 2, 1):
#            if column == 0 and row == 0:
#                padded_image[0][0] = image[0][0]
#            elif (column == width + 1) and (row == height + 1):
#                padded_image[row][column] = image[height - 1][width - 1]
#            elif (column == width + 1) and row == 0:
#                padded_image[0][column] = image[0][width - 1]
#            elif column == 0 and (row == height + 1):
#                padded_image[row][0] = image[height - 1][0]
#            elif row == 0:
#                padded_image[0][column] = image[0][column - 1]
#            elif column == 0:
#                padded_image[row][0] = image[row - 1][0]
#            elif row == height + 1:
#                padded_image[row][column] = image[row - 2][column - 1]
#            elif column == width + 1:
#                padded_image[row][column] = image[row - 1][column - 2]
#            else:
#                padded_image[row][column] = image[row - 1][column - 1]    

    for row in range(1, height + 1, 1):
        for column in range(1, width + 1, 1):
            total = 0
            for y in range(kernel_size):
                for x in range(kernel_size):
                    total += gaussian_kernel[y][x] * padded_image[row + y - 1][column + x - 1]

            blurred_image[row - 1][column - 1] = np.clip(round(total), 0, 255)

    print(blurred_image)
    return blurred_image


def sobel_filter(image):
    sobel_filter_x = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
    sobel_filter_y = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]])
    kernel_size = 3
    
    height, width = image.shape
    filtered_image = np.zeros((height, width), dtype=np.uint8)
    
    padded_image = np.zeros((height + 2, width + 2), dtype=np.uint8)
    magnitudes = np.zeros((height, width), dtype=np.float32)

    padded_image = np.pad(image, pad_width=1, mode='edge')
#    for row in range(0, height + 2, 1):
#        for column in range(0, width + 2, 1):
#            if column == 0 and row == 0:
#                padded_image[0][0] = image[0][0]
#            elif (column == width + 1) and (row == height + 1):
#                padded_image[row][column] = image[height - 1][width - 1]
#            elif (column == width + 1) and row == 0:
#                padded_image[0][column] = image[0][width - 1]
#            elif column == 0 and (row == height + 1):
#               padded_image[row][0] = image[height - 1][0]
#            elif row == 0:
#                padded_image[0][column] = image[0][column - 1]
#            elif column == 0:
#                padded_image[row][0] = image[row - 1][0]
#            elif row == height + 1:
#                padded_image[row][column] = image[row - 2][column - 1]
#            elif column == width + 1:
#                padded_image[row][column] = image[row - 1][column - 2]
#            else:
#                padded_image[row][column] = image[row - 1][column - 1]    

    for row in range(1, height + 1, 1):
        for column in range(1, width + 1, 1):
            total_x = 0
            total_y = 0
            for y in range(kernel_size):
                for x in range(kernel_size):
                    total_x += sobel_filter_x[y][x] * padded_image[row + y - 1][column + x - 1]
                    total_y += sobel_filter_y[y][x] * padded_image[row + y - 1][column + x - 1]
            magnitude = np.sqrt(total_x**2 + total_y**2)
            magnitudes[row - 1][column - 1] = magnitude

    threshold = 40   
    min_magnitude = magnitudes.min()
    max_magnitude = magnitudes.max()

    if max_magnitude - min_magnitude > 0:
        normalized_magnitudes = (magnitudes - min_magnitude) / (max_magnitude - min_magnitude) * 255
        filtered_image = np.where(normalized_magnitudes > threshold, normalized_magnitudes, 0).astype(np.uint8)
    else:
        filtered_image = np.zeros_like(magnitudes, dtype=np.uint8)

    return filtered_image


def resize_image(image, new_width, new_height):
    # Validate input image
    if len(image.shape) != 2:
        raise ValueError("Input image must be a 2D grayscale image")

    # Get the original dimensions
    orig_height, orig_width = image.shape

    # Create an empty output image with the new dimensions
    resized_image = np.zeros((new_height, new_width), dtype=np.uint8)

    # Calculate the scaling factors
    scale_x = orig_width / new_width
    scale_y = orig_height / new_height

    # Iterate over each pixel in the output image
    for y in range(new_height):
        for x in range(new_width):
            # Map the coordinates back to the original image
            orig_x = int(x * scale_x)
            orig_y = int(y * scale_y)

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


def face_detection(image, threshold = 0.5):
    return 
