package com.lab6.opencv;

import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class FaceDetectionActivity extends CameraActivity
        implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "FaceDetect";

    private JavaCameraView    cameraView;
    private TextView          faceCountText;
    private TextView          cameraLabel;
    private CascadeClassifier faceDetector;
    private Mat               gray;
    private volatile int      faceCount   = 0;

    // Track which camera is active: BACK = 99, FRONT = 98
    private int currentCameraId = CameraBridgeViewBase.CAMERA_ID_BACK;

    // Cascade file written once per app session
    private File cascadeXml = null;

    // ── Lifecycle ──────────────────────────────────────────────────────────────

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_face_detection);

        cameraView    = findViewById(R.id.cameraView);
        faceCountText = findViewById(R.id.faceCountText);
        cameraLabel   = findViewById(R.id.cameraLabel);
        Button btnBack = findViewById(R.id.btnBack);
        Button btnFlip = findViewById(R.id.btnFlip);

        cameraView.setVisibility(SurfaceView.VISIBLE);
        cameraView.setCvCameraViewListener(this);
        cameraView.setMaxFrameSize(640, 480);
        cameraView.setCameraIndex(currentCameraId);

        btnBack.setOnClickListener(v -> finish());

        btnFlip.setOnClickListener(v -> flipCamera());
    }

    private void flipCamera() {
        // Toggle between back (99) and front (98)
        currentCameraId = (currentCameraId == CameraBridgeViewBase.CAMERA_ID_BACK)
                ? CameraBridgeViewBase.CAMERA_ID_FRONT
                : CameraBridgeViewBase.CAMERA_ID_BACK;

        cameraView.disableView();
        cameraView.setCameraIndex(currentCameraId);
        cameraView.enableView();

        String label = (currentCameraId == CameraBridgeViewBase.CAMERA_ID_FRONT)
                ? "Camera: Front" : "Camera: Back";
        cameraLabel.setText(label);
    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        List<CameraBridgeViewBase> list = new ArrayList<>();
        list.add(cameraView);
        return list;
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initLocal()) {
            if (faceDetector == null) loadCascade(); // load only once
            cameraView.enableView();
        } else {
            Toast.makeText(this, "OpenCV init failed", Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (cameraView != null) cameraView.disableView();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraView != null) cameraView.disableView();
        if (gray != null) { gray.release(); gray = null; }
    }

    // ── Cascade setup ─────────────────────────────────────────────────────────

    private void loadCascade() {
        try {
            File dir = getDir("cascade", MODE_PRIVATE);
            cascadeXml = new File(dir, "haarcascade_frontalface_default.xml");

            // Write from raw resource only if not already present
            if (!cascadeXml.exists()) {
                InputStream      is  = getResources().openRawResource(R.raw.haarcascade_frontalface_default);
                FileOutputStream fos = new FileOutputStream(cascadeXml);
                byte[] buf = new byte[4096];
                int    n;
                while ((n = is.read(buf)) != -1) fos.write(buf, 0, n);
                fos.close();
                is.close();
            }

            faceDetector = new CascadeClassifier(cascadeXml.getAbsolutePath());
            if (faceDetector.empty()) {
                Log.e(TAG, "Cascade empty — replace res/raw/haarcascade_frontalface_default.xml with the real file");
                faceDetector = null;
                runOnUiThread(() -> Toast.makeText(this, "Face detector not loaded", Toast.LENGTH_SHORT).show());
            }
        } catch (Exception e) {
            Log.e(TAG, "loadCascade: " + e.getMessage());
        }
    }

    // ── Camera callbacks ──────────────────────────────────────────────────────

    @Override
    public void onCameraViewStarted(int width, int height) {
        gray = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        if (gray != null) { gray.release(); gray = null; }
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat rgba = inputFrame.rgba();

        if (faceDetector != null && gray != null) {
            Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY);
            Imgproc.equalizeHist(gray, gray);

            int minPx = Math.min(rgba.rows(), rgba.cols()) / 10;

            // Detect in original orientation first
            List<Rect> allFaces = detectOrientation(gray, Core.ROTATE_90_CLOCKWISE, false);

            // If nothing found, try 90° CW (portrait image in landscape scene)
            if (allFaces.isEmpty()) {
                allFaces = detectOrientation(gray, Core.ROTATE_90_CLOCKWISE, true);
            }
            // If still nothing, try 90° CCW
            if (allFaces.isEmpty()) {
                allFaces = detectOrientation(gray, Core.ROTATE_90_COUNTERCLOCKWISE, true);
            }

            faceCount = allFaces.size();

            Scalar green = new Scalar(0, 255, 0, 255);
            for (Rect r : allFaces) {
                Imgproc.rectangle(rgba,
                        new Point(r.x, r.y),
                        new Point(r.x + r.width, r.y + r.height),
                        green, 3);
                Imgproc.putText(rgba, "Face",
                        new Point(r.x, Math.max(r.y - 8, 20)),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, green, 2);
            }
        }

        final int cnt = faceCount;
        runOnUiThread(() -> faceCountText.setText("Faces detected: " + cnt));
        return rgba;
    }

    /**
     * Detect faces in the gray frame.
     * @param rotate  true = rotate the frame before detection then map coords back
     * @param rotCode Core.ROTATE_90_CLOCKWISE or ROTATE_90_COUNTERCLOCKWISE
     */
    private List<Rect> detectOrientation(Mat grayFrame, int rotCode, boolean rotate) {
        int H = grayFrame.rows();
        int W = grayFrame.cols();
        int minPx = Math.min(H, W) / 10;

        Mat toDetect;
        if (rotate) {
            toDetect = new Mat();
            Core.rotate(grayFrame, toDetect, rotCode);
        } else {
            toDetect = grayFrame;
        }

        MatOfRect faceMat = new MatOfRect();
        faceDetector.detectMultiScale(toDetect, faceMat,
                1.1, 2, 0,
                new Size(minPx, minPx), new Size());

        List<Rect> result = new ArrayList<>();
        for (Rect r : faceMat.toArray()) {
            if (rotate) {
                result.add(mapBack(r, rotCode, H, W));
            } else {
                result.add(r);
            }
        }

        if (rotate) toDetect.release();
        faceMat.release();
        return result;
    }

    /** Map a Rect from the rotated frame back to the original frame coordinates. */
    private static Rect mapBack(Rect r, int rotCode, int origH, int origW) {
        if (rotCode == Core.ROTATE_90_CLOCKWISE) {
            // Rotated frame is (origW rows × origH cols)
            // Inverse: orig_x = r.y,  orig_y = origH - r.x - r.width
            return new Rect(r.y, origH - r.x - r.width, r.height, r.width);
        } else {
            // ROTATE_90_COUNTERCLOCKWISE
            // Inverse: orig_x = origW - r.y - r.height,  orig_y = r.x
            return new Rect(origW - r.y - r.height, r.x, r.height, r.width);
        }
    }
}
