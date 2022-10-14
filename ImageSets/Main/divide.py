# from _typeshed import NoneType
import os
import shutil
import xml.etree.ElementTree as ET
import random

def get_groundtruth(file):
    anno = ET.parse(file).getroot()
    pth = anno.find("object")
    if not pth == None:
        pth = pth.find("name").text
    else:
        pth = None
    return pth, file.split('/')[-1].split('.xml')[0]

annpth_ = '/home/xliu423/datasets/polyps/PolypsSet/GLRC/Annotations'

pths = []
for root, dirs, files in os.walk(annpth_, topdown=False):
    for name in files:
        if name.endswith('.xml'):
            pths.append(os.path.join(root, name))

xmls = {"adenomatous": [], "hyperplastic": []}
for f in pths:
    pth, filename = get_groundtruth(f)
    if pth != None:
        xmls[pth].append(filename)

CLINET_NUM = 4
TRAIN_PERC = 0.7

cls1 = xmls['adenomatous']
count_cls1 = len(cls1)
per_client_count_cls1 = int(count_cls1/CLINET_NUM)
random.shuffle(cls1)
for i in range(CLINET_NUM):
    ci_cls1 = cls1[(per_client_count_cls1 * i):(per_client_count_cls1 * (i+1))]
    ci_cls1train = ci_cls1[:int(len(ci_cls1)*TRAIN_PERC)]
    ci_cls1test = ci_cls1[int(len(ci_cls1)*TRAIN_PERC):]
    with open("c"+str(i)+"train.txt","w") as f:
        for fn in ci_cls1train:
            f.write(fn+'\n')
    with open("c"+str(i)+"test.txt","w") as f:
        for fn in ci_cls1test:
            f.write(fn+'\n')

cls1 = xmls['hyperplastic']
count_cls1 = len(cls1)
per_client_count_cls1 = int(count_cls1/CLINET_NUM)
random.shuffle(cls1)
for i in range(CLINET_NUM):
    ci_cls1 = cls1[(per_client_count_cls1 * i):(per_client_count_cls1 * (i+1))]
    ci_cls1train = ci_cls1[:int(len(ci_cls1)*TRAIN_PERC)]
    ci_cls1test = ci_cls1[int(len(ci_cls1)*TRAIN_PERC):]
    with open("c"+str(i)+"train.txt","a") as f:
        for fn in ci_cls1train:
            f.write(fn+'\n')
    with open("c"+str(i)+"test.txt","a") as f:
        for fn in ci_cls1test:
            f.write(fn+'\n')


print("Finish!")

# cls2 = xmls['hyperplastic']
# count_cls2 = len(cls2)
# per_client_count_cls2 = int(count_cls2/CLINET_NUM)
# random.shuffle(cls2)


    