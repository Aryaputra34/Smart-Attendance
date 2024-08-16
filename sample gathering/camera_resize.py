import cv2
import os

vid = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Menggunakan DSHOW untuk mengatasi delay inisialisasi camera
vid.set(cv2.CAP_PROP_FRAME_WIDTH,640)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
# name = '2107421030_Aryaputra Maheswara'
# name = '2107421008_William Nison Manurung'
# name = '2107421011_Joko Prasetyo'
# name = '2107421011_Ilman Fuada Asror'
name = 'image'
success, frame = vid.read()


img_counter = 0
path = os.getcwd()
while success:
    #inside infinity loop
    success, frame = vid.read()

    # resize = cv2.resize(frame, (640,480))
    cv2.imshow('frame', frame)
    # print(success)
    if cv2.waitKey(1) & 0xFF == ord("q"):
    # if cv2.waitKey(1) & 0xFF == 27: UNTUK ESC button
    # if cv2.waitKey(1) & 0xFF == 32: UNTUK SPACE BAR
    # & 0xFF untuk huruf kecil dan huruf besar, yang dibaca cuma huruf kecil
        
        break
    elif cv2.waitKey(1) & 0xFF == 32:
        # img_name = "./dataset/"+ name +"/image_{}.jpg".format(img_counter) di linux
        img_name = os.path.join(path, "dataset", name, "image_{}.jpg".format(img_counter)) # khusus windows
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1


vid.release()
# Destroy all the windows
cv2.destroyAllWindows() 