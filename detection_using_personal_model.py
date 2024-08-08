"""
 a program to use keras and a customised model for birds species detection in a local library
 this program is part of a set of smal porgrams to learn hpw to use  keras and tensorflow from a customised pretrained models , on mac
 features are very limited and usage is also highly dedicated ot the developer  lacal file structure

 Author Alain Gueguen
"""
from imageai.Detection.Custom import CustomObjectDetection
import pandas as pd
import os
from keras import backend as K
#K.clear_session()
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
#detector.setModelPath("detection_model-ex-001--loss-0031.789.h5")
#detector.setModelPath('detection_model-ex-002--loss-0013.237.h5')
#detector.setModelPath('./detection_model-ex-006--loss-0009.708.h5')
#detector.setModelPath('./detection_model-ex-007--loss-0009.177.h5')
#detector.setModelPath('./detection_model-ex-008--loss-0009.084.h5')
detector.setModelPath("detection_model-ex-011--loss-0008.395.h5")

detector.setJsonPath("json\detection_config.json")
detector.loadModel()
resultatofdetection=open('detectedherons_SSD_epoch11.txt','w')
resultatofNOdetection=open('detectedherons_SSD_epoch11_empty.txt','w')
zefile='res\listimage_SSD_D_WITH_pattern_1583608850.csv'
mainidentifications=pd.read_csv(zefile, sep='|', delimiter=None, header=None, names=None)
coll_one=mainidentifications[0]
smalcolonnefilenames=set(coll_one)
indexofimg=0
currentlocation= os.getcwd()
print (currentlocation)
for i in smalcolonnefilenames:
    indexofimg += 1

    if os.path.isfile(i):
        print("open %s"%i)
        #if  os.path.isfile('ima-detected.jpg'):
        #    os.remove('ima-detected.jpg')
        strimagebox=os.path.join(currentlocation,'boxes_epoch10','boxednewmod_%s'%(os.path.basename(i)) )
        detections = detector.detectObjectsFromImage(input_image=i, output_image_path=strimagebox, minimum_percentage_probability=50)
        if len(detections) == 0:
            strmp="%s | no Detection  \n"%(os.path.basename(i))
            resultatofNOdetection.write(strmp)
            if os.path.isfile(strimagebox):
                os.remove(strimagebox)

        for j in range(len(detections)):
            print (j)
            if detections[j]['name']!='':
                strtmp="%s | %s | %s | %s \n"%(os.path.basename(i),detections[j]['name'] ,detections[j]['percentage_probability'],detections[j]['box_points'] )
                resultatofdetection.write(strtmp)
                if os.path.isfile(strimagebox):
                    try:
                        probanum = float(detections[j]['percentage_probability'])
                        if probanum >80.:   #detections[j]['percentage_probability'])>0.8:
                            tmpnewname="%s_proba_%s.jpg"%(strimagebox.split('.jpg')[0],detections[j]['percentage_probability'])
                            os.rename(strimagebox,tmpnewname)
                        else:
                            os.remove(strimagebox)
                    except:
                        os.remove(strimagebox)
            else:
                strmp="%s | empty name | %s  \n"%(os.path.basename(i),detections[j]['box_points'])
                resultatofNOdetection.write(strmp)
                if os.path.isfile(strimagebox):
                    os.remove(strimagebox)
    else:
        print("%s  IS NOT A FILE"%i)

resultatofdetection.close()
resultatofNOdetection.close()
print (indexofimg)
print (indexofimg)
print (indexofimg)
print (indexofimg)
K.clear_session()
print (indexofimg)
        #...:     print (detections[j]['percentage_probability'])
        #os.path.basename(zefile)
        #Out[15]: 'IMG_4226_DxO11august.jpg'
        #for j in range(len()
        #print (i,detections)
#    print (i)
#detections = detector.detectObjectsFromImage(input_image="../photos/2018/aout18_aber400mm/dxo/IMG_2149_DxO11.jpg", output_image_path="ima-detected.jpg", minimum_percentage_probability=50)
#detections = detector.detectObjectsFromImage(input_image="/Users/agueguen/Pictures/2019/chili2019/dernierpostbog/signed/heron_strie_6.jpg", output_image_path="ima-detected.jpg", minimum_percentage_probability=50)
#../photos/2018/aout18_aber400mm/dxo/IMG_2149_DxO11.jpg

#res/dataset/heroncendre/dataset/validation/images/aug5_images_289.jpg
#print (detections)
