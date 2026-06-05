package com.lab6.opencv;

import android.graphics.Bitmap;
import android.os.Handler;
import android.os.Looper;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Abstract base – subclass (anonymous or named) and override onComplete() to receive result.
 *
 * Usage:
 *   ShadowRemoveFilter.getShadowFilteredImage(bitmap, new ShadowRemoveFilter() {
 *       public void onComplete(Bitmap result) { imageView.setImageBitmap(result); }
 *   });
 */
public abstract class ShadowRemoveFilter {

    // ── Callback interface ────────────────────────────────────────────────────
    public interface MyCallBack<T> {
        void onComplete(T t);
    }

    // Abstract method that anonymous subclasses must implement
    public abstract void onComplete(Bitmap bitmap);

    // ── Threading ────────────────────────────────────────────────────────────
    // Declared as instance-less statics inside a helper class to avoid
    // the "modifier static not allowed here" issue in some compiler versions.
    private static final class Bg {
        static final ExecutorService executor    = Executors.newSingleThreadExecutor();
        static final Handler          mainHandler = new Handler(Looper.getMainLooper());
    }

    // ── Public entry-point ───────────────────────────────────────────────────
    public static void getShadowFilteredImage(Bitmap bitmap, ShadowRemoveFilter filter) {
        final Bitmap input = bitmap.copy(Bitmap.Config.ARGB_8888, false);
        Bg.executor.execute(() -> {
            final Bitmap result = processShadow(input);
            Bg.mainHandler.post(() -> filter.onComplete(result));
        });
    }

    // ── Core shadow-removal algorithm ─────────────────────────────────────────
    private static Bitmap processShadow(Bitmap bitmap) {

        // Step 1 – Bitmap → Mat (CV_8UC1 container; bitmapToMat sets actual type to CV_8UC4)
        Mat srcArr = new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType.CV_8UC1);
        Utils.bitmapToMat(bitmap, srcArr);              // RGBA after this call

        // RGBA → BGR (prerequisite for BGR2HSV)
        Imgproc.cvtColor(srcArr, srcArr, Imgproc.COLOR_RGBA2BGR);

        // Step 2 – BGR → HSV
        Imgproc.cvtColor(srcArr, srcArr, Imgproc.COLOR_BGR2HSV);

        // Step 3 – Split channels; add V channel (index 2) to processing list
        List<Mat> bgrPlanes = new ArrayList<>();
        Core.split(srcArr, bgrPlanes);                  // [H, S, V]

        List<Mat> list = new ArrayList<>();
        list.add(bgrPlanes.get(2));                     // only V processed

        // Step 4 – dilate → medianBlur → absdiff → bitwise_not → normalize
        List<Mat> processedList = new ArrayList<>();
        Mat dilateKernel = Mat.ones(new Size(7, 7), CvType.CV_32F);

        for (Mat plane : list) {
            Mat dilated = new Mat();
            Mat blurMat = new Mat();
            Mat diffMat = new Mat();
            Mat normMat = new Mat();

            Imgproc.dilate(plane, dilated, dilateKernel);
            Imgproc.medianBlur(dilated, blurMat, 21);
            Core.absdiff(plane, blurMat, diffMat);
            Core.bitwise_not(diffMat, diffMat);
            Core.normalize(diffMat, normMat, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC1);

            processedList.add(normMat);
            dilated.release();
            blurMat.release();
            diffMat.release();
        }
        dilateKernel.release();

        // Step 5 – result list: H (index 0), S (index 1), processed V
        List<Mat> result = new ArrayList<>();
        result.add(bgrPlanes.get(0));       // H
        result.add(bgrPlanes.get(1));       // S
        result.add(processedList.get(0));   // processed V

        // Step 6 – merge
        Mat result_norm = new Mat();
        Core.merge(result, result_norm);

        // Step 7 – HSV → BGR → RGBA
        Imgproc.cvtColor(result_norm, result_norm, Imgproc.COLOR_HSV2BGR);
        Imgproc.cvtColor(result_norm, result_norm, Imgproc.COLOR_BGR2RGBA);

        // Step 8 – Mat → Bitmap
        Bitmap out = Bitmap.createBitmap(
                result_norm.cols(), result_norm.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(result_norm, out);

        // Cleanup
        srcArr.release();
        result_norm.release();
        for (Mat ch : bgrPlanes)     ch.release();
        for (Mat ch : processedList) ch.release();

        return out;
    }
}
