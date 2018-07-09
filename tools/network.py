import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
from networks.factory import get_network
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import datetime
from trackingUtils import detector_utils as detector_utils
from kmeans import kmeans
import xml.etree.ElementTree as ET

CLASSES = ('__background__',  # always index 0
           'plate', 'piece of yarn', 'hand', 'car toy',
           'utensils', 'textured block', 'dump truck', 'board book', 'multiple pop-up',
           'cow', 'toy telephone', 'baby doll', 'jack-in-the-box',
           'head', 'child', 'cup',
           'medium-sized ball', 'small ball', '8 letter block', 'music box')



#CLASSES = ('__background__','person','bike','motorbike','car','bus')
class frcnnImageDetections:

    def vis_detections(self, im, class_name, dets, inds, im3d, adrfpFrame, handPoints):
        """Draw detected bounding boxes."""
        
        image = Image.fromarray(im.astype('uint8'), 'RGB')
        #image3d = Image.fromarray(im3d.astype('uint8'), 'RGB')
        draw = ImageDraw.Draw(image)
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            depth = self.getDepht(im3d, bbox)
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            oim = image.crop((x1,y1,x2,y2))
            oim.save("results/objects/"+str(class_name)+".jpg")
            detectedObject = ET.SubElement(adrfpFrame, 'object')
            adrfpLabel = ET.SubElement(detectedObject, 'label')
            adrfpLabel.text = class_name
            adrfpScore = ET.SubElement(detectedObject, 'score')
            adrfpScore.text = str(score)
            adrfpDepth = ET.SubElement(detectedObject, 'depth')
            adrfpDepth.text = str(depth)

            adrfpIsInHand = ET.SubElement(detectedObject, 'isInHand')
            
            if handPoints!=None:
                xc = (x1+x2)/2
                yc = (y1+y2)/2
                xh1 = handPoints[0][0]
                yh1 = handPoints[0][1]
                xh2 = handPoints[1][0]
                yh2 = handPoints[1][1]

                print x1,y1,x2,y2
                xp1 = max(xh1,x1)
                yp1 = max(yh1,y1)
                xp2 = min(xh2,x2)
                yp2 = min(yh2,y2)

                print xp1,yp1,xp2,yp2
                xp1 = min(xp1,xh2)
                yp1 = min(yp1,yh2)
                xp2 = max(xp2,xh1)
                yp2 = max(yp2,yh1)

                print xp1,yp1,xp2,yp2
                print xh1,yh1,xh2,yh2
                koef = ((xp2-xp1)*(yp2-yp1)*1.0)/((xh2-xh1)*(yh2-yh1)*1.0)
                print "IoU:"+str(koef)

                if koef>0.3:
                    print "Predmet je u ruci."
                    adrfpIsInHand.text = "1"
                else:
                    print "Predmet nije u ruci."
                    adrfpIsInHand.text = "0"

                #if (xc>=xh1 and xc<=xh2) and (yc>=yh1 and yc<=yh2):
                #    adrfpIsInHand.text = "1"
                #else:
                #    adrfpIsInHand.text = "0"
            else:
                adrfpIsInHand.text = "0"

            adrfpId = ET.SubElement(detectedObject, 'id')
            adrfpId.text = str(i)
            #oim3d = image3d.crop((x1,y1,x2,y2))
            #oim3d.save("results/objects/"+str(class_name)+"3d"+".jpg")
            print("label: "+str(class_name)+", score: "+str(score)+", depth: "+str(depth))
            draw.rectangle(((bbox[0],bbox[1]),(bbox[2],bbox[3])),fill=None, outline=(255, 0, 0))
            draw.text((bbox[0], bbox[1]), str(class_name), fill="red")
        #name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        #image.save("results/"+str(self.iterPic)+".jpg")
        #image3d.save("results/"+name+"3d"+".jpg")
        return np.array(image)

    def getDepht(self, im3d, bbox):
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        average = 0.0
        strIm = ""
        num = 0
        filteredImage = {}
        for y in range(y1,y2+1):
            strIm += "|"
            for x in range(x1,x2+1):
                d = im3d[y][x][0]
                strIm += "{:>3}|".format(d)
                #if d<=7: continue
                filteredImage[(y,x)] = d
                average += 1.0*d
                num += 1
            strIm += "\n"
        #print strIm
        c1,c2 = kmeans(filteredImage, 2, 1000, 320, 240)
        depth1 = average/num
        depth2 = min([c1, c2])
        depth1 = (depth1 + 18.7579)/0.5181 
        depth2 = (depth2 + 18.7579)/0.5181

        print depth1
        print depth2

        if abs(depth1-depth2)>10:
            depth = depth2
        else:
            depth = depth1
        
        return depth


    def detect(self, im, im3d, detection_graph, trackingSess, adrfpFrame):
        """Detect object classes in an image using pre-computed object proposals."""

        # Load the demo image
        #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
        #im_file = os.path.join('/home/corgi/Lab/label/pos_frame/ACCV/training/000001/',image_name)
        #im = cv2.imread(im_file)

        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(self.sess, self.net, im)
        hboxes, hscores = detector_utils.detect_objects(im, detection_graph, trackingSess)
        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0])
        
        num_hands_detect = 1
        score_thresh = 0.5
        points = detector_utils.draw_box_on_image(num_hands_detect, score_thresh, hscores, hboxes, 320, 240, im, adrfpFrame)
        
        # Visualize detections for each class
        im = im[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')

        CONF_THRESH = 0.8
        NMS_THRESH = 0.3
        for cls_ind, cls in enumerate(CLASSES[1:]):
            if cls=="hand":continue
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]

            inds = np.where(dets[:, -1] >= 0.7)[0]
            if len(inds) == 0:
                #print "preskacem "+str(dets[:, -1])
                continue

            im = self.vis_detections(im, cls, dets, inds, im3d, adrfpFrame, points)
        img = im[:,:,:].copy()    
        img[:,:,2] = im[:,:,0].copy()
        img[:,:,0] = im[:,:,2].copy()

        name = None
        if self.iterPic<10:
            name="0"+str(self.iterPic)
        else:
            name=str(self.iterPic)
        cv2.imwrite(name+'.png',img)
        self.iterPic += 1
        return im

    def __init__(self, args):    
        cfg.TEST.HAS_RPN = True  # Use RPN for proposals

        if args.model == ' ':
            raise IOError(('Error: Model not found.\n'))
            
        self.iterPic = 0
        # init session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        # load network
        self.net = get_network(args.demo_net)
        # load model
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
        self.saver.restore(self.sess, args.model)
       
        #sess.run(tf.initialize_all_variables())

        print '\n\nLoaded network {:s}'.format(args.model)

        # Warmup on a dummy image
        im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
        for i in xrange(2):
            _, _= im_detect(self.sess, self.net, im)

        #im_names = ['01.jpg','02.jpg','03.jpg','04.jpg','05.jpg','06.jpg','07.jpg','08.jpg','09.jpg','10.jpg']


        #for im_name in im_names:
        #    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        #    print 'Demo for data/demo/{}'.format(im_name)
        #    demo(sess, net, im_name)

        #plt.show()