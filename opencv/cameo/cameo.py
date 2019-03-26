import cv2 as cv
import filters
from managers import WindowManager, CaptureManager


class Cameo(object):
    def __init__(self):
        self._windowManager = WindowManager("Cameo", self.onkeypress)
        self._captureManager = CaptureManager( cv.VideoCapture(0), self._windowManager, True )
        self._curveFilter = filters.BGRPortraCurveFilter()

    def run(self):
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            copyFrame = frame
            filters.strokeEdges(frame, copyFrame,9,7)
            self._windowManager.show(copyFrame)
            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onkeypress(self, keycode):
        if keycode == 32: #space somerberry
            self._captureManager.writeImage("screenshot.png")
        elif keycode == 9: #tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo("screencast.avi")
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27: #esc
            self._windowManager.destroyWindow()




if __name__ == "__main__":
    Cameo().run()
