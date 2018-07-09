import glob
import cv2
import os
import time
import xml.etree.ElementTree as ET
import lxml.etree as etree

image_list = []
for filename in sorted(glob.glob('results/session16/*.png')): #assuming gif
    im=cv2.imread(filename)
    print filename
    image_list.append(im)

height, width, layers = image_list[0].shape

fourcc = cv2.cv.CV_FOURCC(*'DIVX')
out = cv2.VideoWriter("session.avi", fourcc, 10.0, (width,height))

tree = ET.parse("experiment_session.adrfp")
body = tree.getroot()[1] 
i = 0
for frame in body:
	delta_time = float(frame[-1].text)
	num_of_frames = delta_time/0.1
	for j in range(int(num_of_frames)):
		out.write(image_list[i])
		#cv2.imshow("image",image_list[i])
		#if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
		#	break
		#time.sleep(0.1)
	print str(i)+"th frame"
	i += 1


out.release()
cv2.destroyAllWindows()