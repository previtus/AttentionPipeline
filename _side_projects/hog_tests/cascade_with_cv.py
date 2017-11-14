import numpy as np
import cv2
from timeit import default_timer as timer
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

folder = 'haarcascades/'
face_cascade = cv2.CascadeClassifier(folder+'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(folder+'haarcascade_eye.xml')
body_cascade = cv2.CascadeClassifier(folder+'haarcascade_fullbody.xml')
upperbody_cascade = cv2.CascadeClassifier(folder+'haarcascade_upperbody.xml')

IMG_PATH = '/home/ekmek/intership_project/video_parser/_videos_to_test/PL_Pizza sample/input/frames/0001.jpg'
IMG_PATH = '/home/ekmek/intership_project/video_parser/_videos_to_test/small_dataset/input/frames/s0216.jpg'
#img = cv2.imread('/home/ekmek/intership_project/hog_tests/s0216/0022.jpg')
#img = cv2.imread('/home/ekmek/intership_project/hog_tests/face_example.jpg')
img = cv2.imread(IMG_PATH)
print(img.shape)
#scale = 1.0/2.0
scale = 0.7

img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

print(img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


start = timer()
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
end = timer()
print("faces:", (end - start))
start = timer()

eyes = eye_cascade.detectMultiScale(gray)
end = timer()
print("eyes:", (end - start))
start = timer()

body = body_cascade.detectMultiScale(gray)
end = timer()
print("bodies:", (end - start))
start = timer()

upperbody = upperbody_cascade.detectMultiScale(gray)
end = timer()
print("upperbodies:", (end - start))



img = Image.open(IMG_PATH)
print(img.size)
img = img.resize((int(img.size[0]*scale),int(img.size[1]*scale)), Image.ANTIALIAS)
width, height = img.size

bool_generate_plot = True
fig, ax = plt.subplots()

plt.imshow(img)
plt.xlim(-1*(width/10.0), width+1*(width/10.0))
plt.ylim(-1*(height/10.0), height+1*(height/10.0))
plt.gca().invert_yaxis()


for (x,y,w,h) in faces:
    print("face", x,y,w,h)
    ax.add_patch(
        patches.Rectangle(  # xy, width, height
            (x, y),
            w,
            h, fill=False, linewidth=2.0, color="red"  # color=cmap(i)
        )
    )

for (x,y,w,h) in eyes:
    print("eye", x,y,w,h)
    ax.add_patch(
        patches.Rectangle(  # xy, width, height
            (x, y),
            w,
            h, fill=False, linewidth=2.0, color="yellow"  # color=cmap(i)
        )
    )

for (x, y, w, h) in body:
    print("body", x, y, w, h)
    ax.add_patch(
        patches.Rectangle(  # xy, width, height
            (x, y),
            w,
            h, fill=False, linewidth=2.0, color="blue"  # color=cmap(i)
        )
    )

for (x, y, w, h) in upperbody:
    print("upperbody", x, y, w, h)
    ax.add_patch(
        patches.Rectangle(  # xy, width, height
            (x, y),
            w,
            h, fill=False, linewidth=2.0, color="green"  # color=cmap(i)
        )
    )

#cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#cv2.imshow('img', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

plt.show()