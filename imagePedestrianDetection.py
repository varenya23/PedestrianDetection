import cv2
import imutils

# Initializing the HOG
hog_pd = cv2.HOGDescriptor() #creates an instance of the HOG descriptor
hog_pd.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) #sets the SVM detector for the HOG descriptor to use a pre-trained model specifically for detecting people

img = cv2.imread('path_to_image')
img = imutils.resize(img, width=min(500, img.shape[1])) #resizes image to width of 500 pixels. Improves HOG detector's effeciency.

# Detecting regions in Image having pedestrians 
(rect, _) = hog_pd.detectMultiScale(img,winStride=(4, 4),padding=(1, 1),scale=1.05)

# Drawing a white rectangle encompassing the pedestrians
for (x, y, w, h) in rect:
	cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()