import cv2
import numpy
import time


class CaptureManager(object):

    def __init__(self, capture, previewWindowsManager=None,
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
        self._framesElapsed = int(0)
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

        #Write to the image file, if any.
        if self.isWritingImage:
            cv2.imwrite(self._imageFilename,self._frame)
            self._imageFilename = None


        #Write to the video file, if any
        self._writeVideoFrame()


        # Release the frame
        self._frame = None
        self._enteredFrame = False

    def writeImage(self, filename):
        '''Write the next exited frame to an image file'''
        self._imageFilename = filename

    def startWritingVideo(self, filename, encoding = cv2.VideoWriter_fourcc('I','4','2','0')):
        '''start writing exited frames to a video file'''
        self._videoFilename = filename
        self._videoEncoding = encoding

    def stopWritingVideo(self):
        '''Stop Writing exited frames to a video file'''
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

    def _writeVideoFrame(self):
        if not self.isWritingVideo:
            return

        if self._videoWriter is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps == 0.0:
                # The capture's FPS is unknown so use an estimate.
                if self._framesElapsed < 20:
                    # Wait until more frames elapse so that
                    # the estimate is more stable
                    return
                else:
                    fps = self._fpsestimate
            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            self._videoWriter = cv2.VideoWriter(self._videoFilename,self._videoEncoding,fps,size)
        self._videoWriter.write(self._frame)


class WindowManager(object):
    def __init__(self, windowName, keypressCallback = None):
        self.keypressCallback = keypressCallback
        self._windowName = windowName
        self._isWindowCreated = False

    @property
    def isWindowCreated(self):
        return self._isWindowCreated

    def createWindow(self):
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self,frame):
        cv2.imshow(self._windowName,frame)

    def destoryWindow(self):
        print(type(self._windowName))
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvent(self):
        keyCode = cv2.waitKey(1)
        if self.keypressCallback is not None and keyCode != -1:
            keyCode &= 0xFF
            self.keypressCallback(keyCode)











