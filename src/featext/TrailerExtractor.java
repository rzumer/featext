package featext;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;

public class TrailerExtractor {
	public static double[] extractMovementGradient(VideoCapture video) {
		List<Double> gradientMagnitudes = new ArrayList<>();
		List<Double> gradientStdDevs = new ArrayList<>();
		
		int processedFrames = 0;
		
		Mat frame = new Mat();
		Mat lastFrame = new Mat();
		
		if (!video.read(frame)) {
			return new double[] { 0, 0 };
		}
		
		lastFrame = frame.clone();
		
		// Read the sequence.
		while (video.read(frame)) {
			Mat frameDifference = new Mat();
			Mat greyFrame = new Mat();
			
			// Calculate the frame difference (movement).
			Core.subtract(frame, lastFrame, frameDifference);
			
			// Convert the frame to greyscale.
			Imgproc.cvtColor(frameDifference, greyFrame, Imgproc.COLOR_BGR2GRAY);
			
			Mat sobelX = new Mat();
			Mat sobelY = new Mat();
			Mat sobel = new Mat();
			
			// Calculate the Sobel gradient in X and Y directions.
			Imgproc.Sobel(greyFrame, sobelX, CvType.CV_32F, 1, 0);
			Imgproc.Sobel(greyFrame, sobelY, CvType.CV_32F, 0, 1);
			
			// Calculate the absolute gradient value.
			Core.convertScaleAbs(sobelX, sobelX);
			Core.convertScaleAbs(sobelY, sobelY);
			
			// Merge X and Y gradients.
			Core.addWeighted(sobelX, 0.5, sobelY, 0.5, 0, sobel);
			
			MatOfDouble gradientMean = new MatOfDouble();
			MatOfDouble gradientStdDev = new MatOfDouble();
			Core.meanStdDev(sobel, gradientMean, gradientStdDev);
			
			gradientMagnitudes.add(gradientMean.get(0, 0)[0]);
			gradientStdDevs.add(gradientStdDev.get(0, 0)[0]);
			
			processedFrames++;
			lastFrame = frame.clone();
		}
		
		double meanGradientMagnitude = gradientMagnitudes.stream().mapToDouble(mag -> mag).average().orElse(0);
		double meanGradientStdDev = gradientStdDevs.stream().mapToDouble(dev -> dev).average().orElse(0);
		
		System.out.println("Processed " + processedFrames + " frames (mag: " + meanGradientMagnitude + ", stddev: " + meanGradientStdDev + ").");
		return new double[] { meanGradientMagnitude, meanGradientStdDev };
	}
}
