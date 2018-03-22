#pip install jpegtran-cffi

from jpegtran import JPEGImage
img = JPEGImage('/home/ekmek/intership_project/video_parser_v1/_videos_to_test/RuzickaDataset/input/five/0001.jpg')


import requests

# JPEGImage can also be initialized from a bytestring
#blob = requests.get("http://example.com/image.jpg").content
#from_blob = JPEGImage(blob=blob)

# Reading various image parameters
print(img.width, img.height)  # "640 480"
print(img.exif_orientation)  # "1" (= "normal")

"""
# If present, the JFIF thumbnail can be obtained as a bytestring
thumb = img.exif_thumbnail

# Transforming the image
img.scale(320, 240).save('scaled.jpg')
img.rotate(90).save('rotated.jpg')
img.crop(0, 0, 100, 100).save('cropped.jpg')

# Transformations can be chained
data = (img.scale(320, 240)
            .rotate(90)
            .flip('horizontal')
            .as_blob())

# jpegtran can transform the image automatically according to the EXIF
# orientation tag
photo = JPEGImage(blob=requests.get("http://example.com/photo.jpg").content)
print(photo.exif_orientation)  # "6" (= 270°)
print(photo.width, photo.height) # "4320 3240"
corrected = photo.exif_autotransform()
print(corrected.exif_orientation)  # "1" (= "normal")
print(corrected.width, corrected.height)  # "3240 4320"
"""