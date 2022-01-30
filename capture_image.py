import cv2

video_path=r"N:\Projects\yolor_tracking\videos\ion_alarm.mkv"


video=cv2.VideoCapture(video_path)
video_height=720
video_width=int(720*1.77)
while True:
    ret,image=video.read()

    if not ret :
        break
    image = cv2.resize(image,(video_width,video_height))
    cv2.imshow("image",image)

    key=cv2.waitKey(0) & 0xFF
    
    if key==ord("q"):
        break
    elif key==ord("w"):
        cv2.imwrite("image2.jpg",image)


video.release()
cv2.destroyAllWindows()