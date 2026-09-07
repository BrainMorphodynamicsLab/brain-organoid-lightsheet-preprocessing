import numpy
from scipy import fftpack

cache_mask = None
cache_width = None
cache_height = None
cache_r_psf_pxl = None

def dct_generate_mask(width, heigth, r_psf_pxl):
    global cache_width
    global cache_height
    global cache_mask
    global cache_r_psf_pxl

    if (cache_width == width and cache_height == heigth and cache_r_psf_pxl == r_psf_pxl):
        return cache_mask

    mask = numpy.zeros(shape=(width,heigth))
    for x in range(width):
        for y in range(width):
            if x+y < int(width/r_psf_pxl):
                mask[y][x] = True
            else: mask[y][x] = False

    cache_width = width
    cache_height = heigth
    cache_r_psf_pxl = r_psf_pxl
    cache_mask = mask

    return mask


def dct_fast(image, mask, r_psf_pxl, bin_factor):
    image = _rebin(image, bin_factor)
    width,_ = numpy.shape(image)
    mask = _rebin(mask, bin_factor)

    Fc = fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho')
    Fc_masked = numpy.multiply(Fc**2, mask)
    L2 = numpy.sum(Fc_masked)
    L2 = numpy.sqrt(L2)
    FL = Fc/L2
    DCTentr = -(2/(width/r_psf_pxl)**2)*numpy.sum(numpy.abs(FL[FL!=0])*(numpy.log2(numpy.abs(FL[FL!=0]))))
    
    return DCTentr


def _rebin(array, binning_factor):
    # get dimensions of original array
    shape = numpy.shape(array)
    binning_factor = int(binning_factor)
    
    # create square image if necessary
    if shape[0] != shape[1]:
        min_dim = numpy.amin(shape)
        array = array[0:min_dim, 0:min_dim]

    # get new dimensions of array
    shape = numpy.shape(array)
    
    # get new dimensions and potential left over
    new_shape = (int(shape[0]//binning_factor), int(shape[1]//binning_factor))
    leftover = (int(shape[0]%binning_factor), int(shape[1]%binning_factor))

    # remove leftover(s) at the right end of the array
    if leftover[0] != 0 or leftover[1] != 0:
        array = array[0 : -leftover[0], 0 : -leftover[1]]

    # tupple for reshapping
    shape = (new_shape[0], binning_factor, new_shape[1], binning_factor)
    
    return array.reshape(shape).mean(-1).mean(1)
