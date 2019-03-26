import cv2


def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

#videoCapture = cv2.VideoCapture("space.mp4") #Using a video file
#Using a live feed from a camera using the camera index
clicked = False
videoCapture = cv2.VideoCapture(0)
fps = 30 #videoCapture.get( cv2.CAP_PROP_FPS )
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter("my_output.avi", cv2.VideoWriter_fourcc('P','I','M','1'), fps, size)

cv2.namedWindow("MyWindow")
cv2.setMouseCallback("MyWindow", onMouse)


print("Using FPS {:}, and size of {:}".format(fps,size))
print("Showing Camera Feed, press any key or click window to stop")

success, frame = videoCapture.read()
numFrames = 10 * fps - 1
while success and cv2.waitKey(1) == -1 and not clicked:
    print("Number of Frames left {:}".format(numFrames))
    print("Success status {:}".format(success))
    videoWriter.write(frame)
    success, frame = videoCapture.read()
    cv2.imshow("MyWindow", frame)
    numFrames -= 1

cv2.destroyWindow("MyWindow")
print("Complete.... xoxoxoxox")
