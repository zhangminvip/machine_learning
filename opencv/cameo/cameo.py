import cv2
import filters
from managers import WindowManager,  CaptureManager

class Cameo(object):
    def __init__(self):
        self._windowManager = WindowManager('Cameo',self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager,True)
        self._curveFilter = filters.SharpenFilter()

    def run(self):
        '''run the main loop'''
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            filters.strokeEdges(frame, frame)
            self._curveFilter.apply(frame,frame)
            self._captureManager.exitFrame()
            self._windowManager.processEvent()

    def onKeypress(self,keyCode):
        '''handle a key press

        space --> Task a screenshot
        tab  --> start/stop a screen a screencast
        escape --> quit
        '''
        print(keyCode)
        if keyCode == 32: #space
            self._captureManager.writeImage('screenshot.png')
        elif keyCode == 9:#tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keyCode == 27:
            self._captureManager._capture.release()
            self._windowManager.destoryWindow()


if __name__ == "__main__":
    Cameo().run()

