import cv2
image = cv2.imread("/home/ekmek/intership_project/hog_tests/face_example.jpg",0)

winSize = (64,64)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
#compute(img[, winStride[, padding[, locations]]]) -> descriptors
winStride = (8,8)
padding = (8,8)
locations = ((10,20),)
#### zjevne ne hist, hog_img = hog.compute(image,winStride,padding,locations, visualise=True)
hist = hog.compute(image,winStride,padding,locations)

print(len(hist),hist)