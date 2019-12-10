import tensorflow as tf
import numpy as np
import facenet
from align import detect_face
import cv2
import argparse
import os, json, tqdm
from models import create_model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import cv2

# Parameters

partition = "withoutbenchmarkpartition"

facetraining_exp = "0.1"
facematch_exp = "default0912Presentation0.1"
facematch_data_save = False
facematch_images_save = True

# End


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# some constants kept as default from facenet
minsize = 30
#threshold = [0.6, 0.7, 0.7] Default, changed because many false positives
threshold = [0.9, 0.9, 0.9]
factor = 0.709
input_image_size = 224

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


def getFace(img, name, set):
    faces = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    i_face = 1
    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            if face[4] > 0.50:
                # x = row, y = col
                # det = [y_left_border, x_top_border, y_right_border, x_bottom_border]
                det = np.squeeze(face[0:4])

                width = abs(det[2] - det[0])
                height = abs(det[3] - det[1])
                max_length = max(height, width)
                center_x = (det[3] + det[1]) / 2
                center_y = (det[2] + det[0]) / 2

                # img_size = (height, width)
                rel_x = center_x / max(img_size[0], img_size[1])
                rel_y = center_y / max(img_size[0], img_size[1])

                bb = np.zeros(4, dtype=np.int32)
                '''
                bb[0] = np.maximum(det[0], 0)
                bb[1] = np.maximum(det[1], 0)
                bb[2] = np.minimum(det[2], img_size[1])
                bb[3] = np.minimum(det[3], img_size[0])
                '''
                bb[0] = np.maximum(round(center_y - max_length/2), 0)
                bb[1] = np.maximum(round(center_x - max_length/2), 0)
                bb[2] = np.minimum(round(center_y + max_length/2), img_size[1])
                bb[3] = np.minimum(round(center_x + max_length/2), img_size[0])

                '''
                # OLD, kept for security
                height = abs(det[2] - det[0])
                width = abs(det[3] - det[1])
                max_length = max(height, width)
                center_x = (det[2] + det[0]) / 2
                center_y = (det[3] + det[1]) / 2

                rel_x = center_x / img_size[0]
                rel_y = center_y / img_size[1]

                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(round(center_x - max_length/2), 0)
                bb[1] = np.maximum(round(center_y - max_length/2), 0)
                bb[2] = np.minimum(round(center_x + max_length/2), img_size[1])
                bb[3] = np.minimum(round(center_y + max_length/2), img_size[0])
                '''

                size = min(bb[2]-bb[0], bb[3]-bb[1])

                if size < minsize:
                    continue

                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                '''
                if facematch_images_save:
                    cv2.imwrite('../data-facematch/'+partition+'/'+facematch_exp+'/'+set+'/face-'+str(name)+'-'+str(i_face)+'.jpg', cropped)
                '''
                resized = cv2.resize(cropped, (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)
                prewhitened = facenet.prewhiten(resized)
                '''
                kernel = np.ones((13, 13), np.float32)/169
                blurred = cv2.filter2D(resized, -1, kernel)
                '''
                blurred = cv2.GaussianBlur(resized,(9,9),cv2.BORDER_DEFAULT)

                blurred_prewhitened = facenet.prewhiten(blurred)

                gray_face = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                blur_face = variance_of_laplacian(gray_face)

                faces.append(
                    {'face': resized, 'rect': [bb[0], bb[1], bb[2], bb[3]], 'size': size, 'embedding': getEmbedding(prewhitened), 'blurred_embedding': getEmbedding(blurred_prewhitened), 'name': name, 'x': rel_x, 'y': rel_y, 'blur': blur_face})
                i_face += 1
    return faces


def getEmbedding(resized):
    reshaped = resized.reshape(-1, input_image_size, input_image_size, 3)
    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    return embedding


def predictWithImageRescale(img1, img2, FLAGS):
    # print("predict")

    for name in FLAGS.images:
        if not os.path.exists(name):
            raise Exception("<" + name + "> not found")
        if os.path.isdir(name):
            for image in os.listdir(name):
                if image.lower().endswith(".jpg"):
                    FLAGS.images.append(name + "/" + image)
    # FLAGS.images = filter(lambda e : not os.path.isdir(e), FLAGS.images)

    X = []
    for i, image in enumerate([img1, img2]):
        width, height = image.shape[:2]
        if width < height:
            width = int(224. * width / height)
            height = 224
        else:
            height = int(224. * height / width)
            width = 224
        image = cv2.resize(image, (height, width)).astype(np.float32)
        image = np.pad(image, ((0, 224 - width), (0, 224 - height), (0, 0)), mode="constant", constant_values=0)
        image = np.expand_dims(image, axis=0)
        X.append(image / 255.)

    score = model_facetraining.predict(X, batch_size=1)[0] * 100

    if score[0] > score[1]:
        # print("<" + FLAGS.images[0] + ">", "is better than", "<" + FLAGS.images[1] + ">", "with {:.1f}% confidence".format(score[0]))
        return 0, score[0]
    else:
        # print("<" + FLAGS.images[1] + ">", "is better than", "<" + FLAGS.images[0] + ">", "with {:.1f}% confidence".format(score[1]))
        return 1, score[1]

    # scores = np.array(scores) / (len(list(FLAGS.images)) - 1)
    # indices = np.argsort(-scores)
    # for i in indices.tolist():
    #    print("<" + list(FLAGS.images)[i] + ">", "scores", "{:.1f}".format(scores[i]))


def predictFaceTraining(X):
    score = model_facetraining.predict(X, batch_size=1)[0] * 100

    if score[0] > score[1]:
        return 1, score[0], score[1]
    else:
        return 2, score[0], score[1]


def predictAutoTriage(img1, img2, name1, name2):
    X = []
    for i, image in enumerate([img1, img2]):
        width, height = image.shape[:2]
        if width < height:
            width = int(224. * width / height)
            height = 224
        else:
            height = int(224. * height / width)
            width = 224
        image = cv2.resize(image, (height, width)).astype(np.float32)
        image = np.pad(image, ((0, 224 - width), (0, 224 - height), (0, 0)), mode="constant", constant_values=0)
        image = np.expand_dims(image, axis=0)
        X.append(image / 255.)

    score = model_autotriage.predict(X, batch_size=1)[0] * 100

    if score[0] > score[1]:
        # print("<" + name1 + ">", "is better than", "<" + name2 + ">", "with {:.1f}% confidence".format(score[0]))
        return 1, score[0], score[1]
    else:
        # print("<" + name2 + ">", "is better than", "<" + name1 + ">", "with {:.1f}% confidence".format(score[1]))
        return 2, score[0], score[1]


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def dataFacematch(sets):
    X, Y = {}, {}
    for set in sets:
        if os.path.exists("../data-facematch/" + partition + "/" + facematch_exp + "/" + set + ".npz") and False:
            print("Loading ", set, ".npz")
            files = np.load(open("../data-facematch/" + partition + "/" + facematch_exp + "/" + set + ".npz", "rb"))
            X[set], Y[set] = [files["X0"], files["X1"]], files["Y"]
        else:
            X[set], Y[set] = [[], []], []

            if not os.path.exists("../data-facematch/" + partition + "/"+facematch_exp+"/"+set):
                os.makedirs("../data-facematch/" + partition + "/"+facematch_exp+"/"+set)

            fileEvaluate = open("../data-facematch/" + partition + "/" + facematch_exp + "/" + set + ".csv", "w")
            fileEvaluate.write('Series;Image 1;Image 2;Metric;ground truth;AutoTriageGoodImage;AutoTriageFirstImageScore;AutoTriageSecondImageScore;Image1Blur;Image2Blur')
            for i in range(0, 10):
                fileEvaluate.write(';iFace1;iFace2;distance;position_difference;Face1Blur;Face2Blur;rect0Face1;rect1Face1;rect2Face1;rect3Face1;rect0Face2;rect1Face2;rect2Face2;rect3Face2;FaceTrainingGoodImage;FaceTrainingFirstImageScore;FaceTrainingSecondImageScore')

            with open("../data/" + partition + "/" + set + ".txt", "r") as file:
                for line in tqdm.tqdm(file.readlines(), desc=set):
                    elements = line.strip().split()
                    prefix, a, b = map(int, elements[:3])
                    # print("Loading image pair {:06d}-{:02d}.JPG {:06d}-{:02d}.JPG".format(prefix, a, prefix, b))
                    name1 = "{:06d}-{:02d}.JPG".format(prefix, a)
                    name2 = "{:06d}-{:02d}.JPG".format(prefix, b)
                    img1 = cv2.imread("../data/images/" + name1)
                    img2 = cv2.imread("../data/images/" + name2)

                    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                    blur_img1 = variance_of_laplacian(gray_img1)
                    blur_img2 = variance_of_laplacian(gray_img2)

                    groundTruth = -1
                    groundTruthAppend = ''
                    groundTruthWrite = ''
                    #if set in ["train", "valid"]:
                    if set in ["train", "valid", "test"]:
                        score = float(elements[3])
                        groundTruth = score
                        groundTruthAppend = ' ground-truth ' + str(score)
                        groundTruthWrite = str(score)

                    goodImage, firstImageScore, secondImageScore = predictAutoTriage(img1, img2, name1, name2)

                    description = 'ifFirstBetter1Else0'

                    autotriagePredictAppend = 'autotriage-predict ' + str(goodImage) + ' confidence {:.1f}%'.format(max(firstImageScore, secondImageScore))

                    facetrainingPredictAppendConcat = ''

                    facetrainingPredictSumScores = 0
                    facetrainingPredictNumFaces = 0

                    fileEvaluate.write('\n' + str(prefix) + ';' + str(a) + ';' + str(b) + ';ifFirstBetter1Else2;' + groundTruthWrite + ';' + str(goodImage) + ';' + str(firstImageScore) + ';' + str(secondImageScore) + ';' + str(blur_img1) + ';' + str(blur_img2))

                    face1 = getFace(img1, "{:06d}-{:02d}".format(prefix, a), set)
                    face2 = getFace(img2, "{:06d}-{:02d}".format(prefix, b), set)
                    if face1 and face2:
                        for i1, f1 in enumerate(face1):
                            for i2, f2 in enumerate(face2):
                                # calculate Euclidean distance
                                distanceNN = np.sqrt(np.sum(np.square(np.subtract(f1['embedding'], f2['embedding']))))
                                distanceNB = np.sqrt(np.sum(np.square(np.subtract(f1['embedding'], f2['blurred_embedding']))))
                                distanceBN = np.sqrt(np.sum(np.square(np.subtract(f1['blurred_embedding'], f2['embedding']))))
                                distance = min(distanceNN, distanceNB, distanceBN)
                                #distance = np.sqrt(np.sum(np.square(np.subtract(f1['embedding'], f2['embedding']))))

                                pos_difference = np.sqrt(np.square(np.subtract(f1['x'], f2['x']))+np.square(np.subtract(f1['y'], f2['y'])))

                                '''
                                # Score detection:
                                threshold = 1.00    # set yourself to meet your requirement
                                if distance <= threshold:
                                    saveNamePair = 'scoreThreshold1.00/' + str(distance) + 'faces' + os.path.basename(FLAGS.images[0].rstrip(os.sep)) + '-' + os.path.basename(FLAGS.images[1].rstrip(os.sep)) + '-' + str(i1) + '-' + str(i2) + '.jpg'
                                    pair = np.concatenate((f1['face'], f2['face']), axis=0)
                                    cv2.imwrite(saveNamePair, pair)
                                '''

                                threshold = 0.9  # set yourself to meet your requirement
                                # print("Result = " + ("same person" if distance <= threshold else "not same person") + "   distance = "+str(distance))
                                if distance <= threshold or True:

                                    facesResized = []
                                    for i, image in enumerate([f1['face'], f2['face']]):
                                        width, height = image.shape[:2]
                                        if width < height:
                                            width = int(224. * width / height)
                                            height = 224
                                        else:
                                            height = int(224. * height / width)
                                            width = 224
                                        image = cv2.resize(image, (height, width)).astype(np.float32)
                                        image = np.pad(image, ((0, 224 - width), (0, 224 - height), (0, 0)), mode="constant", constant_values=0)

                                        if facematch_data_save:
                                            X[set][i].append(image / 255.)

                                        image = np.expand_dims(image, axis=0)
                                        facesResized.append(image / 255.)

                                    if facematch_data_save:
                                        if set in ["train", "valid"]:
                                            score = float(elements[3])
                                            if score >= 0.5:
                                                Y[set].append([1, 0])
                                            else:
                                                Y[set].append([0, 1])

                                    goodFace, face1Score, face2Score = predictFaceTraining(facesResized)

                                    fileEvaluate.write(';' + str(i1) + ';' + str(i2) + ';' + str(distance) + ';' + str(pos_difference) + ';' + str(f1['blur']) + ';' + str(f2['blur']) + ';' + str(f1['rect'][0]) + ';' + str(f1['rect'][1]) + ';' + str(f1['rect'][2]) + ';' + str(f1['rect'][3]) + ';' + str(f2['rect'][0]) + ';' + str(f2['rect'][1]) + ';' + str(f2['rect'][2]) + ';' + str(f2['rect'][3]) + ';' + str(goodFace) + ';' + str(face1Score) + ';' + str(face2Score))

                                    facetrainingPredictSumScores += goodFace
                                    facetrainingPredictNumFaces += 1

                                    facetrainingPredictAppend = ' facetraining-predict ' + str(goodFace) + ' confidence {:.1f}%'.format(max(face1Score, face2Score))
                                    facetrainingPredictAppendConcat += facetrainingPredictAppend


                                    if facematch_images_save:
                                        '''
                                        saveNamePair = "../data-facematch/"+facematch_exp+"/"+set+"/" + str(distance) + ' ' + str(pos_difference) + ' faces' + str(f1['name']) + '-' + str(f2['name']) + '-' + str(i1) + '-' + str(i2) + '.jpg'
                                        pair = np.concatenate((f1['face'], f2['face']), axis=0)
                                        cv2.imwrite(saveNamePair, pair)
                                        '''

                                        folder = ''
                                        if groundTruth != -1 and goodFace == 2-round(groundTruth):
                                            folder = 'facetraining Predict Correct/'
                                        if groundTruth != -1 and goodFace != 2-round(groundTruth):
                                            folder = 'facetraining Predict False/'

                                        if not os.path.exists('../data-facematch/' + partition + '/' + facematch_exp + '/' + set + '/' + folder):
                                            os.makedirs('../data-facematch/' + partition + '/' + facematch_exp + '/' + set + '/' + folder)

                                        saveNamePair = '../data-facematch/' + partition + '/' + facematch_exp + '/' + set + '/' + folder + 'facematch distance ' + str(distance) + ' top-image-is-ground-truth ' + name1 + '-' + name2 + ' ' + str(i1) + '-' + str(i2) + ' ' + description + ' ' + groundTruthAppend + ' ' + autotriagePredictAppend + facetrainingPredictAppend + '.jpg'

                                        pair = np.concatenate((f1['face'], f2['face']), axis=0)
                                        if round(groundTruth) == 0:
                                            pair = np.concatenate((f2['face'], f1['face']), axis=0)

                                        cv2.imwrite(saveNamePair, pair)

                                        if goodFace == 0:
                                            cv2.rectangle(img1, (f1['rect'][0], f1['rect'][1]), (f1['rect'][2], f1['rect'][3]), (0, 255, 0), 2)
                                            cv2.rectangle(img2, (f2['rect'][0], f2['rect'][1]), (f2['rect'][2], f2['rect'][3]), (0, 0, 255), 2)
                                        else:
                                            cv2.rectangle(img2, (f2['rect'][0], f2['rect'][1]), (f2['rect'][2], f2['rect'][3]), (0, 255, 0), 2)
                                            cv2.rectangle(img1, (f1['rect'][0], f1['rect'][1]), (f1['rect'][2], f1['rect'][3]), (0, 0, 255), 2)

                    averagefacetrainingPredictAppend = ''
                    if facetrainingPredictNumFaces > 0:
                        averagefacetrainingPredict = facetrainingPredictSumScores / facetrainingPredictNumFaces
                        averagefacetrainingPredictAppend = ' facetraining-predict-average {:.1f}'.format(averagefacetrainingPredict)

                    '''
                    vis = np.concatenate((img1, img2), axis=0)
                    imageSaveName = '../data-facematch/' + partition + '/' + set + '/' + 'imagematchpair ' + name1 + '-' + name2 + ' ' + description + ' ' + autotriagePredictAppend + averagefacetrainingPredictAppend + facetrainingPredictAppendConcat + '.jpg'
                    cv2.imwrite(imageSaveName, vis)
                    '''
                    if facematch_images_save and face1 and face2:
                        imageSingle1SaveName = '../data-facematch/' + partition + '/' + facematch_exp + '/' + set + '/' + 'imagematchsingle ' + name1 + '-' + name2 + ' 1 ' + description + ' ' + autotriagePredictAppend + averagefacetrainingPredictAppend + facetrainingPredictAppendConcat + '.jpg'
                        cv2.imwrite(imageSingle1SaveName, img1)
                        imageSingle2SaveName = '../data-facematch/' + partition + '/' + facematch_exp + '/' + set + '/' + 'imagematchsingle ' + name1 + '-' + name2 + ' 2 ' + description + ' ' + autotriagePredictAppend + averagefacetrainingPredictAppend + facetrainingPredictAppendConcat + '.jpg'
                        cv2.imwrite(imageSingle2SaveName, img2)

            if facematch_data_save:
                print("map(np.array)")
                X[set][0], X[set][1], Y[set] = map(np.array, [X[set][0], X[set][1], Y[set]])

                # print("Number of Facematches: " + str(len(X[set][0])))
                print("Savez")
                np.savez(open("../data-facematch/" + partition + "/" + facematch_exp + "/" + set + ".npz", "wb"), X0 = X[set][0], X1 = X[set][1], Y = Y[set])
    return X, Y


def loadModelFaceTraining(FLAGS):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    arguments = json.load(open("../exp-facetraining/" + partition + "/" + facetraining_exp + "/arguments.json", "r"))
    for key, value in FLAGS.__dict__.items():
        arguments[key] = value
    for key, value in arguments.items():
        setattr(FLAGS, key, value)

    if not hasattr(FLAGS, "optimizer"):
        setattr(FLAGS, "optimizer", "sgdm")
    if not hasattr(FLAGS, "model"):
        setattr(FLAGS, "model", "vgg16")
    if not hasattr(FLAGS, "siamese"):
        setattr(FLAGS, "siamese", "share")
    if not hasattr(FLAGS, "weights"):
        setattr(FLAGS, "weights", "imagenet")
    if not hasattr(FLAGS, "module"):
        setattr(FLAGS, "module", "subtract")
    if not hasattr(FLAGS, "activation"):
        setattr(FLAGS, "activation", "tanh")
    if not hasattr(FLAGS, "regularizer"):
        setattr(FLAGS, "regularizer", "l2")

    model_facetraining = create_model(FLAGS)
    model_facetraining.load_weights("../exp-facetraining/" + partition + "/" + facetraining_exp + "/weights.hdf5")

    if FLAGS.optimizer == "sgd":
        from keras.optimizers import SGD
        optimizer = SGD(lr=0.001)
    elif FLAGS.optimizer == "sgdm":
        from keras.optimizers import SGD
        optimizer = SGD(lr=0.001, momentum=0.9)
    elif FLAGS.optimizer == "adam":
        from keras.optimizers import Adam
        optimizer = Adam(lr=0.01)
    else:
        raise NotImplementedError

    model_facetraining.compile(optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    return model_facetraining


def loadModelAutoTriage(FLAGS):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    arguments = json.load(open("../exp/" + partition + "/" + FLAGS.exp + "/arguments.json", "r"))
    for key, value in FLAGS.__dict__.items():
        arguments[key] = value
    for key, value in arguments.items():
        setattr(FLAGS, key, value)

    if not hasattr(FLAGS, "optimizer"):
        setattr(FLAGS, "optimizer", "sgdm")
    if not hasattr(FLAGS, "model"):
        setattr(FLAGS, "model", "vgg16")
    if not hasattr(FLAGS, "siamese"):
        setattr(FLAGS, "siamese", "share")
    if not hasattr(FLAGS, "weights"):
        setattr(FLAGS, "weights", "imagenet")
    if not hasattr(FLAGS, "module"):
        setattr(FLAGS, "module", "subtract")
    if not hasattr(FLAGS, "activation"):
        setattr(FLAGS, "activation", "tanh")
    if not hasattr(FLAGS, "regularizer"):
        setattr(FLAGS, "regularizer", "l2")

    model_autotriage = create_model(FLAGS)
    model_autotriage.load_weights("../exp/" + partition + "/" + FLAGS.exp + "/weights.hdf5")

    if FLAGS.optimizer == "sgd":
        from keras.optimizers import SGD
        optimizer = SGD(lr=0.001)
    elif FLAGS.optimizer == "sgdm":
        from keras.optimizers import SGD
        optimizer = SGD(lr=0.001, momentum=0.9)
    elif FLAGS.optimizer == "adam":
        from keras.optimizers import Adam
        optimizer = Adam(lr=0.01)
    else:
        raise NotImplementedError

    model_autotriage.compile(optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    return model_autotriage


parser = argparse.ArgumentParser()
parser.add_argument("--exp", default="default")
parser.add_argument("--gpu", default="0")

FLAGS = parser.parse_args()

model_autotriage = loadModelAutoTriage(FLAGS)

model_facetraining = loadModelFaceTraining(FLAGS)

#sets = ["valid", "train", "test"]
sets = ["test"]

'''
X, Y = dataFacematch(sets)

if facematch_data_save:
    results = {}
    try:
        sets.remove("test")
    except:
        print("test no in sets")

    for set in sets:
        loss, accuracy = model_facetraining.evaluate(X[set], Y[set], batch_size=FLAGS.batch, verbose=1)
        results[set] = accuracy
        print(set, "accuracy:", accuracy)
    json.dump(results, open("../exp-facematch/" + partition + "/" + FLAGS.exp + "/results.json", "w"))
'''
exit()
