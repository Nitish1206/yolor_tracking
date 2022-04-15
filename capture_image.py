import cv2

video_name="test3"
video_path=r"N:\Projects\yolor_tracking\videos\\"+video_name+".mp4"


video=cv2.VideoCapture(video_path)
# frame_size=(1080,720)
# video_height=720
# video_width=1080
while True:
    ret,image=video.read()

    if not ret :
        break
    # image = cv2.resize(image,(video_width,video_height))
    cv2.imshow("image",image)

    key=cv2.waitKey(0) & 0xFF
    
    if key==ord("q"):
        break
    elif key==ord("w"):
        cv2.imwrite(video_name+".jpg",image)


video.release()
cv2.destroyAllWindows()