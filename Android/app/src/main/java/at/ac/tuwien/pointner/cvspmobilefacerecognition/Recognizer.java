/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package at.ac.tuwien.pointner.cvspmobilefacerecognition;

import android.content.ContentResolver;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.RectF;
import android.net.Uri;
import android.os.ParcelFileDescriptor;

import java.io.FileDescriptor;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import at.ac.tuwien.pointner.cvspmobilefacerecognition.env.FileUtils;
import at.ac.tuwien.pointner.cvspmobilefacerecognition.ml.BlazeFace;
import at.ac.tuwien.pointner.cvspmobilefacerecognition.ml.FaceNet;
import at.ac.tuwien.pointner.cvspmobilefacerecognition.ml.LibSVM;

/**
 * Generic interface for interacting with different recognition engines.
 */
public class Recognizer {
    /**
     * An immutable result returned by a Classifier describing what was recognized.
     */
    public class Recognition {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        private final String id;

        /**
         * Display name for the recognition.
         */
        private final String title;

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        private final Float confidence;

        /** Optional location within the source image for the location of the recognized object. */
        private RectF location;

        private FloatBuffer embedding;

        Recognition(
                final String id, final String title, final Float confidence, final RectF location, final FloatBuffer embedding) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
            this.embedding = embedding;
        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }

        public RectF getLocation() {
            return new RectF(location);
        }

        public FloatBuffer getEmbedding() {
            return embedding;
        }

        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
            }

            if (location != null) {
                resultString += location + " ";
            }

            return resultString.trim();
        }
    }

    private static Recognizer recognizer;

    private static FloatBuffer IDCardEmbedding;

    private BlazeFace blazeFace;
    private FaceNet faceNet;
    private LibSVM svm;

    private List<String> classNames;

    private Recognizer() {}

    static Recognizer getInstance (AssetManager assetManager) throws Exception {
        if (recognizer != null) return recognizer;

        recognizer = new Recognizer();
        recognizer.blazeFace = BlazeFace.create(assetManager);
        recognizer.faceNet = FaceNet.getInstance(assetManager);
        recognizer.svm = LibSVM.getInstance();
        recognizer.classNames = FileUtils.readLabel(FileUtils.LABEL_FILE);

        return recognizer;
    }

    CharSequence[] getClassNames() {
        CharSequence[] cs = new CharSequence[classNames.size() + 1];
        int idx = 1;

        cs[0] = "+ add new person";
        for (String name : classNames) {
            cs[idx++] = name;
        }

        return cs;
    }

    public void addIDImage(Bitmap bitmap, RectF rectF) {
        Rect rect = new Rect();
        rectF.round(rect);
        IDCardEmbedding = faceNet.getEmbeddings(bitmap, rect);
        System.out.println("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& IDCardEmbedding added " + IDCardEmbedding);
    }

    List<Recognition> recognizeImage(Bitmap bitmap, Matrix matrix) {
        synchronized (this) {
            /*
            Bitmap bitmapFlipped;
            Matrix matrixFlipped = new Matrix();
            matrixFlipped.preScale(-1.0f, 1.0f);
            bitmapFlipped = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrixFlipped, false);
            */
            List<RectF> faces = blazeFace.detect(bitmap);
            final List<Recognition> mappedRecognitions = new LinkedList<>();

            System.out.println("############################ Faces found: " + faces.size());

            for (RectF rectF : faces) {
                Rect rect = new Rect();
                rectF.round(rect);

                FloatBuffer buffer = faceNet.getEmbeddings(bitmap, rect);
                /*
                System.out.println("Embeddings: "+buffer);
                int size = FaceNet.EMBEDDING_SIZE;
                System.out.println("Embeddings size: "+size);
                for(int i=0; i<size; i++) {
                    System.out.print(buffer.get(i));
                }
                */

                //LibSVM.Prediction prediction = svm.predict(buffer);

                matrix.mapRect(rectF);
                /*
                int index = prediction.getIndex();
                index = 1;

                String name = classNames.get(index);

                System.out.println("############################ Prediction: "+index+" "+name+" "+prediction.getProb()+" "+rectF);
                 */
                /*
                Recognition result =
                        new Recognition("" + index, name, prediction.getProb(), rectF);
                 */
                Recognition result =
                        new Recognition("" + 1, "Donald Trump", 0.4f, rectF, buffer);
                mappedRecognitions.add(result);
            }
            return mappedRecognitions;
        }

    }

    void updateData(int label, ContentResolver contentResolver, ArrayList<Uri> uris) throws Exception {
        synchronized (this) {
            ArrayList<float[]> list = new ArrayList<>();

            for (Uri uri : uris) {
                Bitmap bitmap = getBitmapFromUri(contentResolver, uri);
                List<RectF> faces = blazeFace.detect(bitmap);

                Rect rect = new Rect();
                if (!faces.isEmpty()) {
                    faces.get(0).round(rect);
                }

                float[] emb_array = new float[FaceNet.EMBEDDING_SIZE];
                faceNet.getEmbeddings(bitmap, rect).get(emb_array);
                System.out.println("############################ UpdateData");
                list.add(emb_array);
            }

            svm.train(label, list);
        }
    }

    int addPerson(String name) {
        FileUtils.appendText(name, FileUtils.LABEL_FILE);
        classNames.add(name);

        return classNames.size();
    }

    private Bitmap getBitmapFromUri(ContentResolver contentResolver, Uri uri) throws Exception {
        ParcelFileDescriptor parcelFileDescriptor =
                contentResolver.openFileDescriptor(uri, "r");
        FileDescriptor fileDescriptor = parcelFileDescriptor.getFileDescriptor();
        Bitmap bitmap = BitmapFactory.decodeFileDescriptor(fileDescriptor);
        parcelFileDescriptor.close();

        return bitmap;
    }

    void enableStatLogging(final boolean debug){
    }

    void close() {
        blazeFace.close();
        faceNet.close();
    }
}
