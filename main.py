import matplotlib.pyplot as plt
imagePath="/content/image.png"# the image is of the document from which the operations are done
#read image
image = cv2.imread(imagePath,cv2.COLOR_BGR2RGB)
#Convert to greyscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # grayscale
#Apply threshold
ret,thresh1 = cv2.threshold(gray, 0, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
plt.imshow(thresh1,cmap = 'gray')
#preprocessing
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
plt.imshow(dilation,cmap = 'gray')
#Detect contours
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours[0]
height, width, _ = image.shape
min_x, min_y = width, height
max_x = max_y = 0   
for contour, hier in zip(contours, hierarchy):
    (x,y,w,h) = cv2.boundingRect(contour)
    min_x, max_x = min(x, min_x), max(x+w, max_x)
    min_y, max_y = min(y, min_y), max(y+h, max_y)
    if w > 80 and h > 80:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
if max_x - min_x > 0 and max_y - min_y > 0:
    fin=cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
#plt.imshow(fin)
final=cv2.drawContours(image, contours,-1,(0,0,255),6)
#plt.imshow(final,cmap = 'gray')
plt.imsave('/content/new.png',final)
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
# Load image and HSV color threshold
image = cv2.imread('/content/new.png')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0, 0, 80])
upper = np.array([255, 255, 255])
mask = cv2.inRange(hsv, lower, upper)
#cv2_imshow( hsv)
result = cv2.bitwise_and(image, image, mask=mask)
result[mask==0] = (255, 255, 255)

# Find contours on extracted mask, combine boxes, and extract ROI
cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = np.concatenate(cnts)
x,y,w,h = cv2.boundingRect(cnts)
cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
ROI = result[y:y+h, x:x+w]
plt.imsave('/content/new1.png',mask)
#fingerprint
import cv2
from PIL import ImageOps
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import skimage
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops
from PIL import Image, ImageDraw, ImageOps
img = cv2.imread('/content/new1.png', 0)
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] 
# Extract Blobs
blobs = img > img.mean()
blobs_labels = measure.label(blobs, background=1)
image_label_overlay = label2rgb(blobs_labels, image=img,bg_label=0)
total_area = 0
counter = 0
average = 0.0
for region in regionprops(blobs_labels):
    if region.area >70:
        total_area = total_area + region.area
        counter = counter + 1
# Threshold
average = (total_area/counter)
a4_constant = ((average/100.0)*250.0)+100
b = morphology.remove_small_objects(blobs_labels, a4_constant)
plt.imsave('pre_version.png', b)

# read the pre-version
img2 = cv2.imread('pre_version.png',cv2.IMREAD_GRAYSCALE)
img2 = cv2.threshold(img2, 50, 250, cv2.THRESH_BINARY_INV)[1]
cv2_imshow(img2)
plt.imsave('/content/fingerprints.png',img2) 
#signature
import cv2
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import skimage
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops
img = cv2.imread('/content/new1.png', 0)
#cv2_imshow(img)
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] 
# Extract Blobs
blobs = img > img.mean()
blobs_labels = measure.label(blobs, background=1)
image_label_overlay = label2rgb(blobs_labels, image=img)
b = morphology.remove_small_objects(blobs_labels, 230)
plt.imsave('pre_version.png', b)
# read the pre-version
img2 = cv2.imread('pre_version.png', 0)
hsv = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
lower = np.array([0, 0, 80])
upper = np.array([255, 255, 255])
mask = cv2.inRange(hsv, lower, upper)
mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)[1] 
cv2_imshow(mask)
plt.imsave('/content/signature.png',mask)
