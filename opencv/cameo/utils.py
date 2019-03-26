import cv2
import numpy
import scipy.interpolate


def createCurveFunc(points):
    if points is None:
        return None

    numpoints = len(points)
    if numpoints < 2 :
        return None
    xs,ys = zip(*points)
    if numpoints < 4:
        kind = 'linear'
        print("Error unimplemented")
    else:
        kind = 'cubic'

    return scipy.interpolate.interp1d(xs,ys,kind,bounds_error=False)


def createLookupArray(func, length=256):
    if func is None:
        return None
    lookupArray = numpy.empty(length)
    i = 0
    while i < length:
        func_i = func(i)
        lookupArray[i] = min(max(0,func_i), length-1)
        i += 1
    return lookupArray

def applyLookupArray(lookupArray, src, dst):
    if lookupArray is None:
        return
    dst[:] = lookupArray[src]

def compositeFunc( func1, func2 ):
    if func1 is None:
        return func2
    if func2 is None:
        return func1
    return lambda x: func1( func2(x) )

def createFlatView(array):
    flatview = array.view()
    flatview.shape = array.size
    return flatview
