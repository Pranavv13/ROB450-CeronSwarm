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
        # find convex hull of the centroids (the smallest convex shape that can enclose all the centroids)
        hull = cv2.convexHull(points)
        # draw the outline (hull)
        cv2.polylines(frame, [hull], isClosed=True, color=(0,255,0), thickness=2)


        # find the mean "super-centroid"
        avg_cX = int(sum([c[0] for c in centroids]) / len(centroids))
        avg_cY = int(sum([c[1] for c in centroids]) / len(centroids))
        super_centroid = np.array([avg_cX, avg_cY])

        cv2.circle(frame, (avg_cX, avg_cY), 10, (0, 0, 255), -1)
        cv2.putText(frame, "super-centroid", (avg_cX - 50, avg_cY - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # find distances from outer centroids to super-centroid
        distances = np.linalg.norm(points-super_centroid, axis=1)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        # define outlier distance threshold (adjust 2.5)
        outlier_thresh = mean_dist + 2.5 * std_dist

        # identify outliers
        for i, (cX, cY) in enumerate(points):
            d = distances[i]

            if d > outlier_thresh:
                # draw outlier in blue
                cv2.circle(frame, (cX, cY), 8, (255, 0, 0), -1)
                cv2.putText(frame, "OUTLIER", (cX-20,cY-15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 2)


    out.write(frame) # saves to video file
    cv2.imshow("Frame", frame) # shows live video feed

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()