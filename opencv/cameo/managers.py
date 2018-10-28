import cv2
import numpy
import time


class CaptureManager(object):

    def __init(self, capture, previewWindowsManager=None,
               shouldMirrorPreview=False):
        self.previeWindowManager = previewWindowsManager
        self.shouldMirrorPreview = shouldMirrorPreview

        self._capture = capture
        self._channel = 0
        self._enteredFrame = False
        self._frame = None
        self._imageFilename = None
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None
        self._startTime = None
        self._framesElapsed = long(0)
        self._fpsestimate = None


    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve()
        return self._frame

    @property
    def isWritingImage(self):

        return self._imageFilename is not None

    @property
    def isWritingVideo(self):
        return self._videoFilename is not None

    def enterFrame(self):
        'Capture the next frame ,if any'

        #But first, check that any previous frame was exited
        assert not self._enteredFrame,\
        'previous enterFrame() had no matching exitFrame()'
        if self._capture is not None:
            self._enteredFrame = self._capture.grab()

    def exitFrame(self):
        '''
        Draw to the window, write to the files, Release the frame
        '''
        #Check weather any grabbed frame is retrievable
        #The getter may retrieve and  cache the frame

        if self.frame is None:
            self._enteredFrame = False
            return

        #Update the FPS estimate and related variables
        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time() -self._startTime
            self._fpsestimate = self._framesElapsed/timeElapsed
        self._framesElapsed += 1

        #Drop the window , if any
        if self.previeWindowManager is not None:
            if self.shouldMirrorPreview:
                mirroredFrame = numpy.fliplr(self._frame).copy()
                self.previeWindowManager.show(mirroredFrame)
            else:
                self.previeWindowManager.show(self._frame)










