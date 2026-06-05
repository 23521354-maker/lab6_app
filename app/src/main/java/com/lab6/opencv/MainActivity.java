package com.lab6.opencv;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    private ImageView imageView;
    private Bitmap    originalBitmap;
    private Bitmap    currentBitmap;

    private final ActivityResultLauncher<Intent> pickImage =
        registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            (ActivityResult result) -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    Uri uri = result.getData().getData();
                    if (uri != null) loadBitmapFromUri(uri);
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (!OpenCVLoader.initLocal()) {
            toast("OpenCV init failed");
            finish();
            return;
        }

        setContentView(R.layout.activity_main);

        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        imageView = findViewById(R.id.imageView);

        Button load          = findViewById(R.id.load);
        Button g1            = findViewById(R.id.g1);
        Button g2            = findViewById(R.id.g2);
        Button g3            = findViewById(R.id.g3);
        Button g4            = findViewById(R.id.g4);
        Button g5            = findViewById(R.id.g5);
        Button btnFaceDetect = findViewById(R.id.btnFaceDetect);

        load.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
            intent.setType("image/*");
            pickImage.launch(intent);
        });

        g1.setOnClickListener(v -> {
            if (noImage()) return;
            Mat src  = bitmapToMat(currentBitmap);
            Mat gray = new Mat();
            Mat rgba = new Mat();
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY);
            Imgproc.cvtColor(gray, rgba, Imgproc.COLOR_GRAY2RGBA);
            src.release(); gray.release();
            showMat(rgba);
        });

        g2.setOnClickListener(v -> {
            if (noImage()) return;
            Mat src     = bitmapToMat(currentBitmap);
            Mat blurred = new Mat();
            Imgproc.GaussianBlur(src, blurred, new Size(15, 15), 0);
            src.release();
            showMat(blurred);
        });

        g3.setOnClickListener(v -> {
            if (noImage()) return;
            Mat src   = bitmapToMat(currentBitmap);
            Mat gray  = new Mat();
            Mat edges = new Mat();
            Mat rgba  = new Mat();
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY);
            Imgproc.Canny(gray, edges, 80, 200);
            Imgproc.cvtColor(edges, rgba, Imgproc.COLOR_GRAY2RGBA);
            src.release(); gray.release(); edges.release();
            showMat(rgba);
        });

        g4.setOnClickListener(v -> {
            if (noImage()) return;
            Mat src      = bitmapToMat(currentBitmap);
            Mat gray     = new Mat();
            Mat claheOut = new Mat();
            Mat rgba     = new Mat();
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY);
            CLAHE clahe = Imgproc.createCLAHE(2.0, new Size(8, 8));
            clahe.apply(gray, claheOut);
            Imgproc.cvtColor(claheOut, rgba, Imgproc.COLOR_GRAY2RGBA);
            src.release(); gray.release(); claheOut.release();
            showMat(rgba);
        });

        g5.setOnClickListener(v -> {
            if (noImage()) return;
            ShadowRemoveFilter.getShadowFilteredImage(currentBitmap, new ShadowRemoveFilter() {
                @Override
                public void onComplete(Bitmap bitmap) {
                    currentBitmap = bitmap;
                    imageView.setImageBitmap(bitmap);
                }
            });
        });

        btnFaceDetect.setOnClickListener(v ->
                startActivity(new Intent(this, FaceDetectionActivity.class)));
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private void loadBitmapFromUri(Uri uri) {
        try (InputStream is = getContentResolver().openInputStream(uri)) {
            Bitmap bmp = BitmapFactory.decodeStream(is);
            if (bmp == null) { toast("Could not decode image"); return; }
            originalBitmap = bmp;
            currentBitmap  = bmp.copy(Bitmap.Config.ARGB_8888, true);
            imageView.setImageBitmap(currentBitmap);
        } catch (Exception e) {
            toast("Failed to load: " + e.getMessage());
        }
    }

    private Mat bitmapToMat(Bitmap bmp) {
        Mat m = new Mat();
        Utils.bitmapToMat(bmp, m);
        return m;
    }

    private void showMat(Mat mat) {
        Bitmap out = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat, out);
        mat.release();
        currentBitmap = out;
        imageView.setImageBitmap(out);
    }

    private boolean noImage() {
        if (currentBitmap == null) { toast("Load an image first"); return true; }
        return false;
    }

    private void toast(String msg) {
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show();
    }
}
