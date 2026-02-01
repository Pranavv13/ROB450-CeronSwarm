import cv2
import easyocr as ocr
import matplotlib.pyplot as plt

# read image
img_path = 'E.jpeg'

img = cv2.imread(img_path)

# create ocr
reader = ocr.Reader(['en'])

# detect bounding box, text, confidence
text = reader.readtext(img_path)
bbox, text, score = text[0]

# create new image with results
cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 5)
cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

plt.imshow(img)
plt.show()