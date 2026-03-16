import numpy as np
import cv2

# setup live video capture
print("initializing camera")
cam = cv2.VideoCapture(1) # Open the z30 camera (for my computer, 0 is front camera, 1 is back camera)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Finished Initializing")

# define the list of boundaries (HSV)
red = ([136, 87, 111], [180, 255, 255])
green = ([25, 52, 72], [102, 255, 255])
blue = ([94, 80, 2], [126, 255, 255])


# load the video stream
while True:
    ret, frame = cam.read()

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ----- STEP 1: CREATING MASKS FOR EACH COLOR -----
    # red mask
    lower_red = np.array(red[0], dtype="uint8")
    upper_red = np.array(red[1], dtype="uint8")
    red_mask = cv2.inRange(hsv_frame, lower_red, upper_red)

    # green mask
    lower_green = np.array(green[0], dtype="uint8")
    upper_green = np.array(green[1], dtype="uint8")
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    # blue mask
    lower_blue = np.array(blue[0], dtype="uint8")
    upper_blue = np.array(blue[1], dtype="uint8")
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    # ----- STEP 2: PROCESSING MASKS -----
    kernal = np.ones((5, 5), "uint8")

    # red
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(frame, frame, mask=red_mask)

    # green
    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(frame, frame, mask=green_mask)

    # blue
    blue_mask = cv2.dilate(blue_mask, kernal)
    res_blue = cv2.bitwise_and(frame, frame, mask=blue_mask)


    # ----- STEP 3: CREATING CONTOURS FOR EACH COLOR -----
    # red contours
    contours, _ = cv2.findContours(red_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 600:
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(frame, "Red", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, "Location: (" + str(x) + ", " + str(y) + ")", (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # green contours
    contours, _ = cv2.findContours(green_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 600:
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(frame, "Green", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, "Location: (" + str(x) + ", " + str(y) + ")", (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # blue contours
    contours, _ = cv2.findContours(blue_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 600:
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(frame, "Blue", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, "Location: (" + str(x) + ", " + str(y) + ")", (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("image", frame)
    cv2.waitKey(0)

    if cv2.waitKey(1) == ord('q'):
        # Release the capture and writer objects
        cam.release()
        cv2.destroyAllWindows()
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()