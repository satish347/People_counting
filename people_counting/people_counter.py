from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import cv2
import dlib

processed = -1

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
net = cv2.dnn.readNetFromCaffe('mobilenet_ssd/MobileNetSSD_deploy.prototxt',
                               'mobilenet_ssd/MobileNetSSD_deploy.caffemodel')
# "/home/satish/Downloads/people-counting-opencv/videos/example_01.mp4"
vs = cv2.VideoCapture("/home/satish/Downloads/people-counting-opencv/videos/example_01.mp4")
# vs = cv2.VideoCapture("rtsp://naresh:naresh1234@sidvr1.ddns.net:554/Streaming/Channels/101")
# vs=cv2.VideoCapture("/home/satish/Videos/train.mp4")

# width=vs.get(3)
# height=vs.get(4)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 24.0, (int(width),int(height)),True)


W = None
H = None

fps = FPS().start()

ct = CentroidTracker(maxDisappeared=50, maxDistance=50)
trackers = []
trackableObjects = {}
objectCentroids = {}
presentObjects = []

totalFrames = 0


status1 = False
x1 = 0
y1 = 0
x2 = 0
y2 = 0

while True:
    ret, frame = vs.read()

    if frame is None:
        break

    frame = imutils.resize(frame, width=400)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    status = "waiting"
    rects = []

    if totalFrames % 30 == 0:
        # if True:

        status = "Detecting"
        trackers = []

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > 0.2:

                idx = int(detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                trackers.append(tracker)

    else:
        # if True:

        for tracker in trackers:
            status = "Tracking"

            tracker.update(rgb)
            pos = tracker.get_position()

            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            rects.append((startX, startY, endX, endY))

    height, width, channels = frame.shape
    upper_left = (int(width / 4), int(height / 4))
    bottom_right = (int(width * 3 / 4), int(height * 3 / 4))

    (x1, y1) = upper_left
    (x2, y2) = bottom_right
    # print(x2,y2)
    # print(x2,y2)
    # print(upper_left)
    # print(bottom_right)

    cv2.rectangle(frame, upper_left, bottom_right, (0, 255, 0), 2)

    objects = ct.update(rects)
    presentObjects = []

    for (objectID, centroid) in objects.items():

        to = trackableObjects.get(objectID, None)

        presentObjects.append(objectID)

        # print(trackableObjects.get(objectID))

        if to is None:
            to = TrackableObject(objectID, centroid)


        else:
            y = [c[1] for c in to.centroids]
            x = [c[0] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            direction2 = centroid[0] - np.mean(x)
            to.centroids.append(centroid)
            if not to.counted:
                to.counted = True
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # construct a tuple of information we will be displaying on the
    # frame
    # loop over the info tuples and draw them on our frame
    # for (i, (k, v)) in enumerate(info):
    #     text = "{}: {}".format(k, v)
    #     cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # # check to see if we should write the frame to disk
    # if writer is not None:
    #	writer.write(frame)
    # show the output frame
    cv2.imshow("Frame", frame)
    # out.write(frame)
    key = cv2.waitKey(5) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # increment the total number of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1
    fps.update()
# stop the timer and display FPS information
for i in ct.gone:
    length = len(trackableObjects[i].centroids)
    a, b = trackableObjects[i].centroids[0]
    c, d = trackableObjects[i].centroids[length - 1]
    x_diff = c - a
    y_diff = d - b
    if y_diff > x_diff:
        if y_diff - x_diff > 0:
            for centroid1 in trackableObjects[i].centroids:
                (x, y) = centroid1
                if y > y1 and not status1 and x1 < x < x2:
                    print(i)
                    print("object enter in Y direction")
                    print("enter time")
                    print(ct.enterTime.get(i))
                    status1 = True
                if y > y2 and x1 < x < x2:
                    print(i)
                    print("object exited in Y direction")
                    print("exit time")
                    print(ct.exitTime.get(i))
                    break
    elif abs(y_diff) > x_diff:
        if y_diff - x_diff < 0:
            for centroid1 in trackableObjects[i].centroids:
                (x, y) = centroid1
                if y < y2 and x1 < x < x2 and not status1:
                    print(i)
                    print("object enter in -Y direction")
                    print("enter time")
                    print(ct.enterTime.get(i))
                    status1 = True
                if y < y2 and x1 < x < x2:
                    print(i)
                    print("object exited in -Y direction")
                    print("exit time")
                    print(ct.exitTime.get(i))
                    break
    elif x_diff > y_diff:
        if x_diff - y_diff > 0:
            for centroid1 in trackableObjects[i].centroids:
                (x, y) = centroid1
                if x > x1 and y1 < y < y2 and not status1:
                    print(i)
                    print("ojbect enter in X direction")
                    print("enter time")
                    print(ct.enterTime.get(i))
                    status1 = True
                if x > x2 and y1 < y < y2:
                    print(i)
                    print("object exited in X direction")
                    print("exit time")
                    print(ct.exitTime.get(i))
                    break
    elif abs(x_diff) > y_diff:
        if x_diff - y_diff < 0:
            for centroid1 in trackableObjects[i].centroids:
                (x, y) = centroid1
                if x < x2 and y1 < y < y2 and not status1:
                    print(i)
                    print("object enter in -X direction")
                    print("enter time")
                    print(ct.enterTime.get(i))
                    status1 = True
                if x < x1 and y1 < y < y2:
                    print(i)
                    print("object exited in -X direction")
                    print("exit time")
                    print(ct.exitTime.get(i))
                    break
    status1 = False

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

vs.release()
cv2.destroyAllWindows()
