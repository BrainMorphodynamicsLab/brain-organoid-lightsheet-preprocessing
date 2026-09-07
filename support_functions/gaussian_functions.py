import numpy
import math
import random
from scipy.optimize import curve_fit


def gaussian_fit(y, x):
    # initial guess
    mean = sum(x*y)/sum(y)
    sigma = numpy.sqrt(sum(y*(x-mean)**2)/sum(y))
    try:
        popt, _ = curve_fit(gaussian, x, y, p0=[1,mean,sigma], maxfev=100)
    except RuntimeError:
        return [1, mean, sigma]
        
    return popt


def gaussian_fit_p_value(y, x, gaussian_coef):
    # get the root mean square deviation value between scores of stack and gaussian fit
    y_fitted = gaussian(x, *gaussian_coef)
    rmsd_fitted = _rmsd(y, y_fitted)

    # run permutation test
    higher_than_rmsd_fitted = 0
    test_number = 500
    y_perm = y.copy()

    for _ in range(test_number):
        random.shuffle(y_perm)                                       # randomized data point
        gaussian_coef_perm = gaussian_fit(y_perm, x)                 # new gaussian fit on the rdm data points
        rmsd_perm = _rmsd(y_perm, gaussian(x, *gaussian_coef_perm))  # rmsd on random data
        if rmsd_perm > rmsd_fitted: higher_than_rmsd_fitted += 1

    return higher_than_rmsd_fitted/test_number 


def gaussian(x, a, mean, sigma):
        return [a*(math.exp(-1*(i-mean)**2/(2*sigma**2))) for i in x]


def _rmsd(a, b):
        return math.sqrt(numpy.square(numpy.subtract(a,b)).mean())
