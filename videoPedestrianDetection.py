import cv2
import imutils 

hog_pd = cv2.HOGDescriptor()
hog_pd.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
video = cv2.VideoCapture('path_to_video')

while video.isOpened():
	read_success, image = video.read()
	if read_success:
		image = imutils.resize(image, width=min(400, image.shape[1]))

		# Detecting regions in Image having pedestrians 
		(rect, _) = hog_pd.detectMultiScale(image, winStride=(4, 4), padding=(1, 1), scale=1.05)

		# Drawing the rectangle within the Image
		for (x, y, w, h) in rect:
			cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 1)

		cv2.imshow("Image", image)

		# Waits for 50 miliseconds for 'a' key to be pressed (loop breaks, video processing stops)
		# & 0xFF part ensures compatibility across different platforms.
		if cv2.waitKey(50) & 0xFF == ord('a'):
			break
	else:
		break

video.release()
cv2.destroyAllWindows()