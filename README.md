# CVSP_Mobile_Face_Recognition

# How to install:
Go to https://play.google.com/store/apps/details?id=at.ac.tuwien.pointner.cvspmobilefacerecognition&hl=de or search for "CVSP" in the Play Store and 
download my "CVSP Mobile face recognition / Biometric match" app.

# Requirements:
Minimum Android 8.0 (=minSdkVersion: 26, targetSdkVersion: 29, architecture: armeabi-v7a or arm64-v8a)

# How to run the FaceNetAgeTest:
- Enter FaceNetAgeTest
- Go to http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/data_infor.html and download the "Test Data_v1." an extract it into the a new subfolder ./dataset/test
- Download the model https://gitlab.fit.cvut.cz/pitakma1/mvi-sp/blob/eb9c9db755077bd6fe0a61c1bbb1cced5f20d6d1/data/20170512-110547/20170512-110547.pb and put it into folder named 20170512-110547
- Create and export folder
- Install tensorflow (I used tensorflow-gpu:1.14.0) and all packages under requirements.txt
- Run face_match_test_random_subset.py

# Used tutorials and resources for implementation:
- https://developer.android.com/training/camera/photobasics
- https://codelabs.developers.google.com/codelabs/face-detection/#4
- https://medium.com/analytics-vidhya/facenet-on-mobile-cb6aebe38505
- https://github.com/pillarpond/face-recognizer-android
