# --------------------------------------
# Shows single "super-centroid" of all robots in the video feed
# --------------------------------------
import numpy as np
import cv2

print("initializing camera")
cam = cv2.VideoCapture(2) # Open the z30 camera (for my computer, 0 is front camera, 1 is back camera)

# Get z30 width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('super-centroid.mp4', fourcc, 60.0, (frame_width, frame_height))

print("Finished Initializing")

while True:
    ret, frame = cam.read() # ret = true if frame is read correctly, frame = the actual image
    if ret == False:
        print("Error: Failed to capture frame")
        break

    # image processing to create contrast between robot and background
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)[1]
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # computing centroids of robots 
    centroids = []
    for c in cnts:
        M = cv2.moments(c) # moments are an average of the pixel intensities in the contour,
        # m00: Area of the shape, m10, m01: Used to calculate the centroid (the geometric center).
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))

    if centroids:
        points = np.array(centroids, dtype=np.int32)
        # Find convex hull
        hull = cv2.convexHull(points)
        # Draw the hull (outline)
        cv2.polylines(frame, [hull], isClosed=True, color=(0,255,0), thickness=2)


        # find the mean "super-centroid"
        avg_cX = int(sum([c[0] for c in centroids]) / len(centroids))
        avg_cY = int(sum([c[1] for c in centroids]) / len(centroids))
        cv2.circle(frame, (avg_cX, avg_cY), 10, (0, 0, 255), -1)
        cv2.putText(frame, "super-centroid", (avg_cX - 50, avg_cY - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    out.write(frame) # saves to video file
    cv2.imshow("Frame", frame) # shows live video feed

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()