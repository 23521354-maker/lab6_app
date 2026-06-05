package com.lab6.opencv;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Converts an image to a high-contrast document scan appearance
 * using adaptive thresholding.
 */
public class DocumentFilter {

    public static Mat apply(Mat src) {
        Mat gray    = new Mat();
        Mat blurred = new Mat();
        Mat thresh  = new Mat();
        Mat result  = new Mat();

        // ── Grayscale ────────────────────────────────────────────────────────
        if (src.channels() == 4) {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY);
        } else if (src.channels() == 3) {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
        } else {
            src.copyTo(gray);
        }

        // ── Gaussian blur to reduce noise before thresholding ─────────────────
        Imgproc.GaussianBlur(gray, blurred, new Size(5, 5), 0);
        gray.release();

        // ── Adaptive threshold → clean black-and-white document look ──────────
        Imgproc.adaptiveThreshold(blurred, thresh, 255,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY, 11, 2);
        blurred.release();

        // ── Back to RGBA so bitmapToMat/matToBitmap stays consistent ──────────
        Imgproc.cvtColor(thresh, result, Imgproc.COLOR_GRAY2RGBA);
        thresh.release();

        return result;  // caller is responsible for releasing
    }
}
