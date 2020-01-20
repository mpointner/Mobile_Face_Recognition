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
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.RectF;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.util.SparseArray;

import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.face.Face;
import com.google.android.gms.vision.face.FaceDetector;

import java.io.FileDescriptor;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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

        private final int color;

        Recognition(
                final String id, final String title, final Float confidence, final RectF location, final FloatBuffer embedding, int color) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
            this.embedding = embedding;
            this.color = color;
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

        public int getColor() {
            return color;
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
        IDCardEmbedding = ByteBuffer.allocateDirect(FaceNet.EMBEDDING_SIZE * FaceNet.BYTE_SIZE_OF_FLOAT)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        FloatBuffer embedding = faceNet.getEmbeddings(bitmap, rect);
        IDCardEmbedding.rewind();
        IDCardEmbedding.put(embedding);
        IDCardEmbedding.flip();
        System.out.println("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& IDCardEmbedding added " + IDCardEmbedding);
    }

    List<Recognition> recognizeImage(Bitmap bitmap, Matrix matrix, Context context) {
        synchronized (this) {
            /*
            Bitmap bitmapFlipped;
            Matrix matrixFlipped = new Matrix();
            matrixFlipped.preScale(-1.0f, 1.0f);
            bitmapFlipped = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrixFlipped, false);
            */

            final List<Recognition> mappedRecognitions = new LinkedList<>();


            FaceDetector faceDetector = new
                    FaceDetector.Builder(context)
                    .setTrackingEnabled(true)
                    .setMode(FaceDetector.FAST_MODE)
                    .setProminentFaceOnly(true)
                    .build();
            if (!faceDetector.isOperational()) {
                System.out.println("Could not set up the face detector!");
            }

            Frame frame = new Frame.Builder().setBitmap(bitmap).build();
            SparseArray<Face> facesMobileVision = faceDetector.detect(frame);

            System.out.println("############################ Faces found: " + facesMobileVision.size());

            for (int i = 0; i < facesMobileVision.size(); i++) {
                Face thisFace = facesMobileVision.valueAt(i);
                float x1 = thisFace.getPosition().x;
                float y1 = thisFace.getPosition().y;
                float x2 = x1 + thisFace.getWidth();
                float y2 = y1 + thisFace.getHeight();

                System.out.println("MobileVisionFace: x1=" + x1 + ", y1=" + y1 + ", x2=" + x2 + ", y2=" + y2);

                Rect rect = new Rect();
                RectF rectF = new RectF(x1, y1, x2, y2);
                rectF.round(rect);

                FloatBuffer buffer = faceNet.getEmbeddings(bitmap, rect);

                if (IDCardEmbedding != null) {
                    System.out.println("Embeddings: ");
                    int size = FaceNet.EMBEDDING_SIZE;
                    float sum = 0f;
                    for (int j = 0; j < size; j++) {
                        sum += Math.pow(buffer.get(j) - IDCardEmbedding.get(j), 2);
                    }
                    float dist = (float) Math.sqrt(sum);

                    System.out.println("Embedding Distance: " + dist);

                    matrix.mapRect(rectF);


                    Recognition result;
                    if (dist <= 1.1) {
                        result = new Recognition("" + 1, "FaceMatch confirmed", dist, rectF, buffer, Color.GREEN);
                    } else {
                        result = new Recognition("" + 1, "FaceMatch denied", dist, rectF, buffer, Color.RED);
                    }

                    mappedRecognitions.add(result);
                }

                //tempCanvas.drawRoundRect(new RectF(x1, y1, x2, y2), 2, 2, myRectPaint);
            }


            /*
            List<RectF> faces = blazeFace.detect(bitmap);

            System.out.println("############################ Faces found: " + faces.size());

            for (RectF rectF : faces) {
                Rect rect = new Rect();
                rectF.round(rect);

                FloatBuffer buffer = faceNet.getEmbeddings(bitmap, rect);

                if (IDCardEmbedding != null) {
                    System.out.println("Embeddings: ");
                    int size = FaceNet.EMBEDDING_SIZE;
                    float sum = 0f;
                    for (int i = 0; i < size; i++) {
                        sum += Math.pow(buffer.get(i) - IDCardEmbedding.get(i), 2);
                    }
                    float dist = (float) Math.sqrt(sum);

                    System.out.println("Embedding Distance: " + dist);
                }

                //LibSVM.Prediction prediction = svm.predict(buffer);

                matrix.mapRect(rectF);

                //Recognition result = new Recognition("" + index, name, prediction.getProb(), rectF);
                Recognition result =
                        new Recognition("" + 1, "Donald Trump", 0.4f, rectF, buffer);
                mappedRecognitions.add(result);
            }
            */
            return mappedRecognitions;
        }

    }

    void updateData(int label, ContentResolver contentResolver, ArrayList<Uri> uris) throws Exception {
        synchronized (this) {
            System.out.println("UPDATE UPDATE UPDATE UPDATE UPDATE UPDATE UPDATE UPDATE UPDATE UPDATE UPDATE UPDATE UPDATE UPDATE UPDATE UPDATE UPDATE UPDATE UPDATE");
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
