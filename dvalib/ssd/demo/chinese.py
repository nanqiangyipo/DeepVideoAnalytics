# # -*- coding:utf-8 -*-
from PIL import Image,ImageDraw,ImageFont
# import cv2
import numpy as np
img=Image.new("RGB", (200,200),(120,20,20))
d=ImageDraw.Draw(img)
d.text((0,0),'asdf',(254,254,0))
# img.save(r't.jpg')
#
#
# writer=cv2.VideoWriter(r'aczm.avi',fourcc=cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'),fps=25,frameSize=(200,200))
# if writer.isOpened:
#     print 'yes'
# else:
#     print 'no'
#
# for  _ in range(4000):
#     writer.write(np.asanyarray(img))
#
# # writer.release()

from skvideo.io import FFmpegWriter
import numpy as np

outputdata = np.random.random(size=(1000, 480, 680, 3)) * 255
outputdata = outputdata.astype(np.uint8)

writer = FFmpegWriter("outputvideo.mp4" ,outputdict={  '-vcodec': 'libx264'})

for i in range(outputdata.shape[0]):
    print(i,np.asanyarray(img))
    writer.writeFrame(np.asanyarray(img))

writer.close()