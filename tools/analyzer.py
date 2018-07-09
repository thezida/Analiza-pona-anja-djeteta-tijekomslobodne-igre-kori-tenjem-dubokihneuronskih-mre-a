import xml.etree.ElementTree as ET
import lxml.etree as etree
import argparse

CLASSES = ('__background__',  # always index 0
           'plate', 'piece of yarn', 'hand', 'car toy',
           'utensils', 'textured block', 'dump truck', 'board book', 'multiple pop-up',
           'cow', 'toy telephone', 'baby doll', 'jack-in-the-box',
           'head', 'child', 'cup',
           'medium-sized ball', 'small ball', '8 letter block', 'music box')

def smoothing(body, i, label):
	before = False
	after = False
	if i!=0:
		frame = body[i-1]
		for obj in frame:
			if label == "hand": continue
			if label == "__background__": continue
			if len(obj)==0:continue
			if obj[0].text==label:
				if int(obj[3].text)==1:
					before = True
	if i!=len(body)-1:
		frame = body[i+1]
		for obj in frame:
			if label == "hand": continue
			if label == "__background__": continue
			if len(obj)==0:continue
			if obj[0].text==label:
				if int(obj[3].text)==1:
					after = True
	return before, after

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='FRCNN pepper analyzer')
    parser.add_argument('--source', dest='source', help='Experiment data', default=None, type=int)

    args = parser.parse_args()

if __name__ == "__main__":
	#args = parse_args()

	#if args.source == ' ':
	#	raise IOError(('Error: Source not found.\n'))

	tree = ET.parse("results/session16/experiment_session.adrfp")
	body = tree.getroot()[1]
	t = 0
	objectsInfo = {}

	for objClass in CLASSES:
		if objClass == "hand": continue
		if objClass == "__background__": continue
		objectsInfo[objClass] = ([], 0) 

	for i in range(len(body)):
		frame = body[i]
		objects = frame[0:len(frame)-1]
		delta_time = float(frame[-1].text)
		
		if i!=len(body)-1:
			next_delta_time = float(body[i+1][-1].text)
		else:
			next_delta_time = 0
		print delta_time, next_delta_time
		delta_time = (delta_time + next_delta_time)/2
		print delta_time
		t += delta_time
		for obj in objects:
			label = obj[0].text

			if label=="hand":continue

			score = float(obj[1].text)
			depth = float(obj[2].text)
			isInHand = int(obj[3].text)

			beforeInHand, afterInHand = smoothing(body, i, label)
			if beforeInHand and afterInHand:
				isInHand = 1

			detphList = objectsInfo[label][0]
			detphList.append(depth)
			totalTime = objectsInfo[label][1]
			print i, label
			if isInHand==1:
				print "yes"
				totalTime = totalTime + delta_time
			objectsInfo[label] = (detphList, totalTime)

	print "total time :"+str(t)
	for objClass in CLASSES:
		if objClass == "hand": continue
		if objClass == "__background__": continue
		print str(objClass)+" - time in hand: "+str(objectsInfo[objClass][1]) 

