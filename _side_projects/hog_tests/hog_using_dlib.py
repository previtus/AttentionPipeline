import sys
import dlib
from PIL import Image
import numpy as np
from timeit import default_timer as timer

# (563, 511, 3)   total time: 0.433351890999802
#file_name = "/home/ekmek/intership_project/hog_tests/face_example.jpg"
file_name = '/home/ekmek/intership_project/video_parser/_videos_to_test/small_dataset/input/frames/s0216.jpg'

show = True


if show:
    win = dlib.image_window()

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()

# Load the image into an array
#image = io.imread(file_name)
image = Image.open(file_name)

image = np.array(image)
print(image.shape)

# Run the HOG face detector on the image data.
# The result will be the bounding boxes of the faces in our image.

start = timer()
detected_faces = face_detector(image, 1)
end = timer()
t = (end - start)
print("total time:", t)

print("I found {} faces in the file {}".format(len(detected_faces), file_name))

# Open a window on the desktop showing the image
if show:
    win.set_image(image)

# Loop through each face we found in the image
for i, face_rect in enumerate(detected_faces):

    # Detected faces are returned as an object with the coordinates
    # of the top, left, right and bottom edges
    print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(),
                                                                             face_rect.right(), face_rect.bottom()))

    # Draw a box around each face we found
    if show:
        win.add_overlay(face_rect)

# Wait until the user hits <enter> to close the window
if show:
    dlib.hit_enter_to_continue()

