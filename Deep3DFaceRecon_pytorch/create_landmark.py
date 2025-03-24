import os
import re
import math
#os.add_dll_directory(os.getcwd())

import argparse
from mtcnn import MTCNN
import cv2

if not os.environ['PATH'].__contains__(os.getcwd()):
  os.environ['PATH'] = os.getcwd() + os.pathsep + os.environ['PATH']


def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def main(args):
    #img = cv2.cvtColor(cv2.imread("ivan.jpg"), cv2.COLOR_BGR2RGB)
    detector = MTCNN()


    directory = os.path.abspath(args.image_folder)
    totals = len(os.listdir(directory))
    j = 0
    for file in os.listdir(directory):
         filename = os.path.join(directory, file)
         if filename.endswith(".jpg"):
             img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
             detect = detector.detect_faces(img)
             confidence = detect[0]['confidence']
             bbox =  detect[0]['box']
             #print(totals)
             if j < 1: print("Detection progress: " + str(round((j/totals)*100,2)) + "%")
             if float(confidence) > 0.95 or j < 1:
                 Leye = find_between(str(detect), '\'left_eye\': (', ')').replace(', ', '\t')
                 Reye = find_between(str(detect), '\'right_eye\': (', ')').replace(', ', '\t')
                 nose = find_between(str(detect), '\'nose\': (', ')').replace(', ', '\t')
                 Lmouth = find_between(str(detect), '\'mouth_left\': (', ')').replace(', ', '\t')
                 Rmouth = find_between(str(detect), '\'mouth_right\': (', ')').replace(', ', '\t')
                 centerX = (float(bbox[0])+(float(bbox[2])/2.0)) #center X
                 centerY = (float(bbox[1])+(float(bbox[3])/2.0)) #center Y
                 LeyeB = Leye
                 ReyeB = Reye
                 noseB = nose
                 LmouthB = Lmouth
                 RmouthB = Rmouth
                 centerXB = centerX
                 centerYB = centerY
             else:
                 print('low confidence')
                 Leye = LeyeB
                 Reye = ReyeB
                 nose = noseB
                 Lmouth = LmouthB
                 Rmouth = RmouthB
                 centerX = centerXB
                 centerY = centerYB
             #print(Leye + '\n' + Reye + '\n' + nose + '\n' + Lmouth + '\n' + Rmouth)
             f = open(os.path.join(directory, 'detections\\', file.replace('.jpg', '.txt')), "w")
             f.write(Leye + '\n' + Reye + '\n' + nose + '\n' + Lmouth + '\n' + Rmouth)
             f.close()
             f2 = open(os.path.join(directory, 'detections\\', file.replace('.jpg', '.center')), "w")
             f2.write(str(centerX) + '\n' + str(centerY))
             f2.close()
             j+=1
             print("Detection progress: " + str(round((j/totals)*100,2)) + "%")



             continue
             #break
         else:
             continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect Faces')
    parser.add_argument('-i', '--image_folder', type=str, default='configs/')


    args = parser.parse_args()
    main(args)


