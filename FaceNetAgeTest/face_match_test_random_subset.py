import tensorflow as tf
import numpy as np
import facenet
from align import detect_face
import cv2
import os
import shutil
import random

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import matplotlib.pyplot as plt





config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# some constants kept as default from facenet
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 0  # 44
input_image_size = 160

sess = tf.Session()

# read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
pnet, rnet, onet = detect_face.create_mtcnn(sess, 'align')

# read 20170512-110547 model file downloaded from https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk
facenet.load_model("20170512-110547/20170512-110547.pb")

# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]




dir = "dataset/test/"

not_detected = 0


def getFace(img):
    faces = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            if face[4] > 0.50:
                det = np.squeeze(face[0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                resized = cv2.resize(cropped, (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)
                prewhitened = facenet.prewhiten(resized)
                faces.append(
                    {'face': resized, 'rect': [bb[0], bb[1], bb[2], bb[3]], 'embedding': getEmbedding(prewhitened)})
    return faces


def getEmbedding(resized):
    reshaped = resized.reshape(-1, input_image_size, input_image_size, 3)
    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    return embedding


def compare2face(person1, person1_image, person2, person2_image):
    global not_detected
    img1 = cv2.imread(dir+person1+"/"+person1_image)
    img2 = cv2.imread(dir+person2+"/"+person2_image)
    face1 = getFace(img1)
    face2 = getFace(img2)
    if face1 and face2:
        # calculate Euclidean distance
        dist = np.sqrt(np.sum(np.square(np.subtract(face1[0]['embedding'], face2[0]['embedding']))))
        return dist
    else:
        if face1 == []:
            shutil.copyfile(dir+person2+"/"+person2_image, "export/not_detected_faces/"+person2+" - "+person2_image)
            not_detected += 1
        if face2 == []:
            shutil.copyfile(dir+person2+"/"+person2_image, "export/not_detected_faces/"+person2+" - "+person2_image)
            not_detected += 1
    return -1


def choosePersonImages(person1, person2, exportFile):
    distance = -1
    while distance == -1:
        person1_dir = dir+person1+"/"
        person1_images = os.listdir(person1_dir)
        person1_image = random.choice(person1_images)

        person2_dir = dir+person2+"/"
        person2_images = os.listdir(person2_dir)
        person2_image = random.choice(person2_images)

        while str(person1_image) == str(person2_image):
            person2_image = random.choice(person2_images)

        distance = compare2face(person1, person1_image, person2, person2_image)

    exportFile.write(person1+";"+person1_image+";"+person2+";"+person2_image+";"+str(distance)+"\n")
    print(person1+";"+person1_image+";"+person2+";"+person2_image+";"+str(distance))

    return distance



num_samples = 10000

threshold = 1.1


persons = os.listdir(dir)

print("Same Person:")
samePerson = []
samePerson_Filename = "export/same_person.csv"
if os.path.exists(samePerson_Filename):
    samePerson_importFile = open(samePerson_Filename, "r")
    samePerson_importFile.readline()
    for line in samePerson_importFile.readlines():
        elements = line.strip().split(";")
        person1, person1_image, person2, person2_image = elements[:4]
        distance = float(elements[4])
        samePerson.append(distance)
    samePerson_importFile.close()
else:
    samePerson_exportFile = open(samePerson_Filename, "w")
    samePerson_exportFile.write("person1;person1_image;person2;person2_image;distance\n")

    for count in range(num_samples):
        person = random.choice(persons)

        samePerson.append(choosePersonImages(person, person, samePerson_exportFile))
    samePerson_exportFile.close()
#print(samePerson)



print("Different Person:")
differentPerson = []
differentPerson_Filename = "export/different_person.csv"
if os.path.exists(differentPerson_Filename):
    differentPerson_importFile = open(differentPerson_Filename, "r")
    differentPerson_importFile.readline()
    for line in differentPerson_importFile.readlines():
        elements = line.strip().split(";")
        person1, person1_image, person2, person2_image = elements[:4]
        distance = float(elements[4])
        differentPerson.append(distance)
    differentPerson_importFile.close()
else:
    differentPerson_exportFile = open(differentPerson_Filename, "w")
    differentPerson_exportFile.write("person1;person1_image;person2;person2_image;distance\n")

    for count in range(num_samples):
        person1 = random.choice(persons)
        person2 = random.choice(persons)

        while str(person1) == str(person2):
            person2 = random.choice(persons)

        differentPerson.append(choosePersonImages(person1, person2, differentPerson_exportFile))
    differentPerson_exportFile.close()
#print(differentPerson)


print("Not detected: "+str(not_detected)+" percent: "+str(not_detected/(2.0*num_samples)))


colors = ['#E69F00', '#56B4E9']
names = ['Compared images of same person (Sample size: '+str(num_samples)+')', 'Compared image of different persons (Sample size: '+str(num_samples)+')']
plt.hist([samePerson, differentPerson], bins = int(100), normed=True,
         color = colors, label=names)
plt.legend()
plt.title("FaceNet evaluation based on test set of VGGFace2 dataset (500 people, ~170.000 images)")
plt.xlabel('Difference Metric - Smaller Values means more similar')
plt.ylabel('Normalized amount of compared images with difference metric value')

plt.show()


'''
persons = os.listdir(dir)
for person in persons:
    personDir = dir + person + "/"
    imagesPerson = os.listdir(personDir)
    for i1, imageName1 in enumerate(imagesPerson):
        print(personDir+imageName1)
        img1 = cv2.imread(personDir+imageName1)
        for i2, imageName2 in enumerate(imagesPerson):
            if i2 <= i1:
                continue
            img2 = cv2.imread(personDir+imageName2)
            distance = compare2face(img1, img2)

            print(str(i1)+" + "+str(i2)+": "+str(distance))
    break
'''

exit()
