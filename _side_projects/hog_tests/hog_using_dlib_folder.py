import os
import dlib
from PIL import Image
import numpy as np
from timeit import default_timer as timer

images_folder = "/home/ekmek/intership_project/hog_tests/s0216/"
images_folder = "/home/ekmek/intership_project/video_parser_v1/_videos_to_test/small_dataset/temporary_small_test4_diffModelSize_544_384x384/crops/s0216/"
show = False

image_files = os.listdir(images_folder)

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()

faces_found = 0

#for i in range(0,3):
for i in range(0,len(image_files)):
    img_path = images_folder+image_files[i]
    image = Image.open(img_path)
    image = np.array(image)
    print(image.shape)

    start = timer()
    detected_faces = face_detector(image, 1)
    end = timer()
    t = (end - start)
    print("total time:", t)

    print("I found {} faces in the file {}".format(len(detected_faces), img_path))
    faces_found += len(detected_faces)

    if show:
        win = dlib.image_window()
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

print("Faces found =",faces_found)