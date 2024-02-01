import cv2
import numpy as np     # Numpy module will be used for horizontal stacking of two frames

video=cv2.VideoCapture(0)
a=0
while True:
    a=a+1
    check, frame= video.read()
    if check is False:
        print('shit')
        break  # terminate the loop if the frame is not read successfully

    # Converting the input frame to grayscale
    #gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   

    # Fliping the image as said in question
    gray_flip = cv2.flip(frame,-1)
    #proceed_frame = cv2.cvtColor(gray_flip, cv2.COLOR_GRAY2BGR)

    # Combining the two different image frames in one window
    #combined_window = np.hstack([gray,gray_flip])

    # Displaying the single window
    cv2.imshow("Combined videos ",gray_flip)
    key=cv2.waitKey(1)

    if key==ord('q'):
        break
print(a)

video.release()
cv2.destroyAllWindows