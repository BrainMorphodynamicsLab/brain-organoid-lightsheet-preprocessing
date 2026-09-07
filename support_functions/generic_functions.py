import numpy
from skimage import filters


def get_steps(range, step):
    step_positive = numpy.arange(0, range/2 + step, step)
    step_negative = - numpy.flip(numpy.delete(step_positive, 0))
    steps = numpy.append(step_negative, step_positive)
    return list(steps)


def sigmoid (x, center, slope, amplitude, baseline):
    return 1 / (1 + numpy.exp ((x - center) / slope)) * amplitude + baseline


def moving_average(vector, window_size):
    """
    Apply a moving average of a given window size to a vector.

    Parameters
    ----------
    vector : 1D array
        Vector to which to apply the moving average.
    window_size : int
        Window size of the moving average.

    Returns
    -------
    moving average : list of float
        Moving average of the vector.
    """
    i = 0
    moving_average = []
    while i < len(vector) - window_size + 1:
        window = numpy.copy(vector[i : i + window_size])
        window_average = numpy.sum(window) / window_size
        moving_average.append(window_average)	
        i += 1
    return moving_average


def moving_median(vector, window_size):
    """
    Apply a moving median of a given window size to a vector.

    Parameters
    ----------
    vector : 1D array
        Vector to which to apply the moving median.
    window_size : int
        Window size of the moving median.

    Returns
    -------
    moving median : list of float
        Moving median of the vector.
    """
    i = 0
    moving_median = []
    while i < len(vector) - window_size + 1:
        window = numpy.copy(vector[i : i + window_size])
        window_median = numpy.median(window)
        moving_median.append(window_median)	
        i += 1
    return moving_median


def crop_image(image, crop_x, crop_y, center_x=None, center_y=None):
    """
    Crop an image around a point.

    Parameters
    ----------
    image : 2D array
        Image to crop.
    crop_x, crop_y : int
        Size (in pixels) of the crop along the X and Y direction.
    center_x, center_y : int
        Coordinate (in pixels) of the point. If undefined: center of the image

    Returns
    -------
    image : 2D array
        Cropped image.
    """
    y,x = image.shape
    if center_x is None:
        center_x = x//2
    if center_y is None:
        center_y = y//2
    start_x = center_x-(crop_x//2)
    start_y = center_y-(crop_y//2)
    crop = numpy.copy(image[start_y:start_y+crop_y,start_x:start_x+crop_x])    
    
    return crop


def fit_line(x,y):
    """
    Fit a line to points.

    Parameters
    ----------
    x,y : 1-D array
        Independent and dependent coordinates of points.

    Returns
    -------
    slope, intercept, r_squared : float
        Slope and intercept of the fitted line.
    """
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    mean_x = numpy.average(x)
    mean_y = numpy.average(y)
    x_centered = x - mean_x
    y_centered = y - mean_y
    slope = numpy.dot(x_centered,y_centered) / numpy.dot(x_centered,x_centered)
    intercept = mean_y - slope * mean_x
    total_sum_of_squares = numpy.sum(numpy.square(y_centered))
    residual_sum_of_squares = numpy.sum(numpy.square(y - (slope * x + intercept) ))
    r_squared = 1 - residual_sum_of_squares/total_sum_of_squares

    return slope, intercept, r_squared


def remove_background(image, std_factor):
    error = False
    background = int(numpy.median(image))
    image = numpy.copy(image)
    image[image == 2**16 - 1] = background # remove hot pixels
    values_below_background = background - numpy.ndarray.flatten(image[image <= background])
    std_values_below_background = numpy.sqrt(numpy.sum(numpy.square(values_below_background)) / len(values_below_background))
    background_threshold = background + std_factor * std_values_below_background
    image = image.astype('float64') - background_threshold
    image[image < 0] = 0
    if numpy.count_nonzero(image) == 0:
        print('\nERROR: Beam signal is too low or too far from the center of the image. Please, center the beam around the approximate center or increase laser power or camera exposure.')
        error = True

    return image, error


def interpolate_curve_with_missing_values(vector):
    error = False
    valid_indeces = numpy.argwhere(numpy.isfinite(vector))
    invalid_indeces = numpy.argwhere(numpy.isnan(vector))
    if valid_indeces.size <= 0.5 * vector.size:
        print('\nERROR: Beam signal is too low. Please, increase the laser power or camera exposure.')
        error = True
        return vector, error
    
    for index in invalid_indeces:
        if not any(valid_indeces < index):
            following = min(valid_indeces[valid_indeces > index])
            value = vector[following]
        elif not any(valid_indeces > index):
            previous = max(valid_indeces[valid_indeces < index])
            value = vector[previous]
        else:
            previous = max(valid_indeces[valid_indeces < index])
            following = min(valid_indeces[valid_indeces > index])
            value = (vector[previous] * (following-index) + vector[following] * (index-previous)) / (following-previous)
        vector[index] = value

    return vector, error