package at.ac.tuwien.pointner.cvspmobilefacerecognition;

import android.Manifest;
import android.app.AlertDialog;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.drawable.BitmapDrawable;
import android.hardware.Camera;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.SparseArray;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.face.Face;
import com.google.android.gms.vision.face.FaceDetector;
import com.google.android.material.floatingactionbutton.FloatingActionButton;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

import at.ac.tuwien.pointner.cvspmobilefacerecognition.env.FileUtils;
import at.ac.tuwien.pointner.cvspmobilefacerecognition.env.Logger;

public class MainActivity extends AppCompatActivity {
    private static final Logger LOGGER = new Logger();

    public ImageView imageView;

    static final int REQUEST_TAKE_PHOTO = 1;
    static final int REQUEST_CAMERA = 2;

    String currentPhotoPath;

    private FloatingActionButton nextButton;
    private Recognizer recognizer;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        System.out.println("&&&&&&&&&&&&&& START &&&&&&&&&&&");
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = findViewById(R.id.imageView);
        nextButton = findViewById(R.id.next);

        // Here, thisActivity is the current activity
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {

            // Permission is not granted
            // Should we show an explanation?
            if (ActivityCompat.shouldShowRequestPermissionRationale(this,
                    Manifest.permission.CAMERA)) {
                // Show an explanation to the user *asynchronously* -- don't block
                // this thread waiting for the user's response! After the user
                // sees the explanation, try again to request the permission.
                System.out.println("Show an explanation to the user *asynchronously*");
            } else {
                // No explanation needed; request the permission
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.CAMERA},
                        REQUEST_CAMERA);

                // MY_PERMISSIONS_REQUEST_READ_CONTACTS is an
                // app-defined int constant. The callback method gets the
                // result of the request.
            }
        } else {
            // Permission has already been granted
        }

        try {
            File dir = new File(FileUtils.ROOT);

            if (!dir.isDirectory()) {
                if (dir.exists()) dir.delete();
                dir.mkdirs();

                AssetManager mgr = getAssets();
                FileUtils.copyAsset(mgr, FileUtils.DATA_FILE);
                FileUtils.copyAsset(mgr, FileUtils.MODEL_FILE);
                FileUtils.copyAsset(mgr, FileUtils.LABEL_FILE);
            }

            recognizer = Recognizer.getInstance(getAssets(), getApplicationContext());
        } catch (Exception e) {
            LOGGER.e("Exception initializing classifier!", e);
            finish();
        }

        AlertDialog alertDialog = new AlertDialog.Builder(MainActivity.this).create();
        alertDialog.setTitle("Welcome to this Mobile face recognition / Biometric match demo");
        alertDialog.setMessage("Please take a landscape photo of your ID Card.");
        alertDialog.setButton(AlertDialog.BUTTON_NEUTRAL, "OK",
                (dialog, which) -> {
                    dialog.dismiss();
                    takeIDPhoto();
                });
        alertDialog.show();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String[] permissions, int[] grantResults) {
        switch (requestCode) {
            case REQUEST_CAMERA: {
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    // permission was granted, yay! Do the
                    // contacts-related task you need to do.
                    System.out.println("Permission granted");
                } else {
                    // permission denied, boo! Disable the
                    // functionality that depends on this permission.
                    System.out.println("Permission denied");
                }
                return;
            }

            // other 'case' lines to check for other
            // permissions this app might request.
        }
    }

    public void next(View view) {
        Intent intent = new Intent(this, CameraActivity.class);
        startActivity(intent);
    }

    public void action(View view) {
        takeIDPhoto();
    }

    private void showVideo() {
        boolean hasCameraHardware = checkCameraHardware(this);
        System.out.println("hasCameraHardware: " + hasCameraHardware);
        System.out.println("NumberOfCameras: " + Camera.getNumberOfCameras());
        Camera camera = getCameraInstance();
        System.out.println("Camera: " + camera);
    }

    /**
     * Check if this device has a camera
     */
    private boolean checkCameraHardware(Context context) {
        // this device has a camera
        // no camera on this device
        return context.getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA);
    }

    /**
     * A safe way to get an instance of the Camera object.
     */
    public static Camera getCameraInstance() {
        Camera c = null;
        try {
            c = Camera.open(); // attempt to get a Camera instance
        } catch (Exception e) {
            // Camera is not available (in use or does not exist)
            System.err.println("Camera is not available (in use or does not exist)");
        }
        return c; // returns null if camera is unavailable
    }

    private void takeIDPhoto() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        // Ensure that there's a camera activity to handle the intent
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            // Create the File where the photo should go
            File photoFile = null;
            try {
                photoFile = createImageFile();
            } catch (IOException ex) {
                // Error occurred while creating the File
                System.out.println("IOException");
            }
            // Continue only if the File was successfully created
            if (photoFile != null) {
                Uri photoURI = FileProvider.getUriForFile(this,
                        "at.ac.tuwien.pointner.cvspmobilefacerecognition.fileprovider",
                        photoFile);
                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                startActivityForResult(takePictureIntent, REQUEST_TAKE_PHOTO);
            }
        }
    }

    private File createImageFile() throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(
                imageFileName,  /* prefix */
                ".jpg",         /* suffix */
                storageDir      /* directory */
        );

        // Save a file: path for use with ACTION_VIEW intents
        currentPhotoPath = image.getAbsolutePath();
        return image;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_TAKE_PHOTO && resultCode == RESULT_OK) {
            System.out.println("onActivityResult");
            /*
            Bundle extras = data.getExtras();
            Bitmap imageBitmap = (Bitmap) extras.get("data");
            imageView = (ImageView) findViewById(R.id.imageView);
            imageView.setImageBitmap(imageBitmap);
             */
            galleryAddPic();
            setPic();
        }
    }

    private void galleryAddPic() {
        Intent mediaScanIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
        File f = new File(currentPhotoPath);
        Uri contentUri = Uri.fromFile(f);
        mediaScanIntent.setData(contentUri);
        this.sendBroadcast(mediaScanIntent);
    }

    private void setPic() {
        // Get the dimensions of the View
        int targetW = imageView.getWidth();
        int targetH = imageView.getHeight();

        // Get the dimensions of the bitmap
        BitmapFactory.Options bmOptions = new BitmapFactory.Options();
        bmOptions.inJustDecodeBounds = true;

        int photoW = bmOptions.outWidth;
        int photoH = bmOptions.outHeight;

        // Determine how much to scale down the image
        int scaleFactor = Math.min(photoW / targetW, photoH / targetH);

        // Decode the image file into a Bitmap sized to fill the View
        bmOptions.inJustDecodeBounds = false;
        bmOptions.inSampleSize = scaleFactor;
        bmOptions.inPurgeable = true;

        Bitmap bitmap = BitmapFactory.decodeFile(currentPhotoPath, bmOptions);


        imageView.setImageDrawable(new BitmapDrawable(getResources(), bitmap));

        Toast.makeText(getApplicationContext(), "", Toast.LENGTH_LONG).show();

        AlertDialog patientDialog = new AlertDialog.Builder(MainActivity.this).create();
        patientDialog.setTitle("FaceDetection Running");
        patientDialog.setMessage("Searching for Faces... This might take a few seconds, please be patient.");
        patientDialog.setButton(AlertDialog.BUTTON_NEUTRAL, "OK",
                (dialog, which) -> {
                    dialog.dismiss();
                });
        patientDialog.show();

        AsyncTask.execute(() -> {
                    FaceDetector faceDetector = new
                            FaceDetector.Builder(getApplicationContext()).setTrackingEnabled(false)
                            .build();
                    if (!faceDetector.isOperational()) {
                        System.out.println("Could not set up the face detector!");
                        return;
                    }

                    SparseArray<Face> faces = null;

                    Bitmap faceSearchBitmap = Bitmap.createBitmap(bitmap);
                    for (int i = 0; (faces == null || faces.size() == 0) && i < 4; i++) {
                        Frame frame = new Frame.Builder().setBitmap(faceSearchBitmap).build();
                        faces = faceDetector.detect(frame);

                        if (faces.size() == 0) {
                            Matrix rotateMat = new Matrix();
                            rotateMat.postRotate(90);
                            faceSearchBitmap = Bitmap.createBitmap(faceSearchBitmap, 0, 0, faceSearchBitmap.getWidth(), faceSearchBitmap.getHeight(), rotateMat, true);
                        }
                    }

                    SparseArray<Face> finalFaces = faces;
                    Bitmap finalFaceSearchBitmap = faceSearchBitmap;
                    runOnUiThread(() -> updateImageView(finalFaceSearchBitmap, finalFaces));
                }
        );



    }

    public void updateImageView(final Bitmap faceSearchBitmap, final SparseArray<Face> faces) {
        Bitmap tempBitmap = Bitmap.createBitmap(faceSearchBitmap.getWidth(), faceSearchBitmap.getHeight(), Bitmap.Config.RGB_565);
        Canvas tempCanvas = new Canvas(tempBitmap);
        tempCanvas.drawBitmap(faceSearchBitmap, 0, 0, null);

        Paint myRectPaint = new Paint();
        myRectPaint.setStrokeWidth(12.0f);
        myRectPaint.setColor(Color.GREEN);
        myRectPaint.setStyle(Paint.Style.STROKE);

        for (int i = 0; i < faces.size(); i++) {
            Face thisFace = faces.valueAt(i);
            float x1 = thisFace.getPosition().x;
            float y1 = thisFace.getPosition().y;
            float x2 = x1 + thisFace.getWidth();
            float y2 = y1 + thisFace.getHeight();

            recognizer.addIDImage(faceSearchBitmap, new RectF(x1, y1, x2, y2));

            tempCanvas.drawRoundRect(new RectF(x1, y1, x2, y2), 2, 2, myRectPaint);
        }

        imageView.setImageDrawable(new BitmapDrawable(getResources(), tempBitmap));

        if (faces.size() > 0) {
            nextButton.setVisibility(View.VISIBLE);

            next(null);
        } else {
            /*
            Toast toast = Toast.makeText(getApplicationContext(), "", Toast.LENGTH_LONG);
            toast.show();
            */
            AlertDialog alertDialog = new AlertDialog.Builder(MainActivity.this).create();
            alertDialog.setTitle("No face detected");
            alertDialog.setMessage("On your photo was no face detected. Have you taken a photo of an ID card with a photo on it? Was it sharp? Please retake the photo...");
            alertDialog.setButton(AlertDialog.BUTTON_NEUTRAL, "OK",
                    (dialog, which) -> {
                        dialog.dismiss();
                        takeIDPhoto();
                    });
            alertDialog.show();
        }
    }


}
