package featext;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.imgproc.Imgproc;

public class PosterExtractor {
	// Based on approach by Daniel Calandria on https://github.com/fergunet/osgiliath/blob/master/FeaturesOpenCV/)
	public static double[] extractHueHistogram(Mat image, int bins) {
		double[] hueHistogram = new double[16];
		
		Mat histogram = new Mat();
		
		Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2HSV);
		List<Mat> channels = new ArrayList<Mat>();
		Core.split(image, channels);
		
		Imgproc.calcHist(
				Arrays.asList(new Mat[]{channels.get(0)}),
				new MatOfInt(0),
				new Mat(),
				histogram,
				new MatOfInt(bins),
				new MatOfFloat(0, 179));
		
		for (int i = 0; i < bins; i++) {
			hueHistogram[i] = histogram.get(i, 0)[0];
		}
		
		return hueHistogram;
	}
}
