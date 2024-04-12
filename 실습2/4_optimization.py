import numpy as np
import cv2 as cv
import skimage.data
import skimage.segmentation
import time

img = skimage.data.coffee()

start = time.time()

slic = skimage.segmentation.slic(img, compactness=20, n_segments=600, start_label=1)

marking = skimage.segmentation.mark_boundaries(img, slic)
slic_coffee = np.uint8(marking * 255.0)

print(img.shape, "image를 분할하는 데", time.time()-start, "초 소요")

cv.imshow("SLIC", slic_coffee)
cv.waitKey(0)
cv.destroyAllWindows()
