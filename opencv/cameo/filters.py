import cv2 as cv
import numpy
import utils


def recolourRC(src,dst):
    b, g, r = cv.split(src)
    cv2.addWeighted(b, 0.5, g, 0.5, 0, b)
    cv2.merge((b,b,r), dst)


def recolourRGV(src,dst):
    b,g,r  = cv.split(src)
    cv2.min( b,g,b )
    cv.min( b,r,b )
    cv.merge((b,g,r), dst)


def recolourCMV(src,dst):
    b,g,r = cv.split(src)
    cv.max(b,g,b)
    cv.max(b,r,b)
    cv.merge((b,g,r),dst)


class VFuncFilter(object):
    def __init__(self, vFunc=None, dtype=numpy.uint8):
        length = numpy.iinfo(dtype).max + 1
        self._vLookupArray = utils.createLookupArray(vFunc, length)

    def apply(self, src, dst):
        srcFlatView = utils.flatview(src)
        dstFlatView = utils.flatview(dst)
        utils.applyLookupArray(self._vLookupArray, srcFlatView, dstFlatView)


class VCurveFilter(VFuncFilter):
    def __init__(self, vPoints, dtype=numpy.uint8):
        VFuncFilter.__init__(self, utils.createCurveFunc(vPoints), dtype)


class BGRFuncFilter(object):
    def __init__(self, vFunc=None, bFunc=None, gFunc=None, rFunc=None, dtype=numpy.uint8):
        length = numpy.iinfo(dtype).max + 1
        self._bLookupArray = utils.createLookupArray(utils.compositeFunc(bFunc,vFunc), length)
        self._gLookupArray = utils.createLookupArray(utils.compositeFunc(gFunc,vFunc), length)
        self._rLookupArray = utils.createLookupArray(utils.compositeFunc(rFunc,vFunc), length)

    def apply(self, src, dst):
        b, g, r = cv.split(src)
        utils.applyLookupArray(self._bLookupArray, b, b)
        utils.applyLookupArray(self._gLookupArray, g ,g )
        utils.applyLookupArray(self._rLookupArray, r, r )
        cv.merge([b,g,r], dst)


class BGRCurveFilter(BGRFuncFilter):
    def __init__(self, vPoints=None, bPoints=None, gPoints=None, rPoints=None, dtype=numpy.uint8):
        BGRFuncFilter.__init__(self, utils.createCurveFunc(vPoints), utils.createCurveFunc(bPoints), utils.createCurveFunc(gPoints), utils.createCurveFunc(rPoints), dtype)


def strokeEdges(src, dst, kBlurSize=7, kEdgeSide=5 ):
    if kBlurSize >= 3:
        blurredSrc = cv.medianBlur(src, kBlurSize)
        graySrc = cv.cvtColor(blurredSrc, cv.COLOR_BGR2GRAY)
    else:
        graySrc = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    cv.Laplacian(graySrc, cv.CV_8U, graySrc, ksize=kEdgeSide)
    normalizedInverseAlpha = (1.0/255) * (255 - graySrc)
    channels = cv.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv.merge(channels, dst)


class BGRPortraCurveFilter(BGRCurveFilter):
    """A filter that applies Portra-like curves to BGR."""
    def __init__(self, dtype = numpy.uint8):
        BGRCurveFilter.__init__(
               self,
               vPoints = [(0,0),(23,20),(157,173),(255,255)],
               bPoints = [(0,0),(41,46),(231,228),(255,255)],
               gPoints = [(0,0),(52,47),(189,196),(255,255)],
               rPoints = [(0,0),(69,69),(213,218),(255,255)],
               dtype = dtype)


class VConvolutionFilter(object):
    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        cv.filters2D(src, -1, self._kernel, dst)


class SharpenFilter(VConvolutionFilter):
    def __init(self):
        kernel = numpy.array([[-1,-1,-1],
                              [-1, 9, -1],
                              [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class FindEdgesFilter(VConvolutionFilter):
    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                              [-1, 8, -1 ],
                              [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)
