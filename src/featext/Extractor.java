package featext;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalDouble;
import java.util.stream.Collectors;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;

public class Extractor {

	public static void main(String[] args) {
		if (args.length < 1) {
			System.err.println("Usage: Extractor <poster> [<trailer1> [<trailer2> [...]]]");
			System.exit(1);
		}
		
		String posterFilename = args[0];
		String[] trailerFilenames = Arrays.copyOfRange(args, 1, args.length);
		
		File posterFile = new File(posterFilename);
		if (!posterFile.exists()) {
			System.err.println("Specified poster file does not exist. Aborting.");
			System.exit(1);
		}
		
		// Load OpenCV libraries.
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		//System.loadLibrary("opencv_ffmpeg2411_64.dll");
		
		Mat image = Highgui.imread(posterFilename);
		double[] hueHistogram = PosterExtractor.extractHueHistogram(image, 16);
		
		System.out.println("Poster hue histogram: " + (Arrays.stream(hueHistogram).mapToObj(hue -> Double.toString(hue))).collect(Collectors.joining(", ")));
		// do something with the histogram
		
		List<Double> gradientMagnitudes = new ArrayList<>();
		List<Double> gradientStdDevs = new ArrayList<>();
		
		for(String trailerFilename : trailerFilenames) {
			File trailerFile = new File(trailerFilename);
			if (!trailerFile.exists()) {
				System.err.println("Specified trailer file does not exist. Skipping.");
			}
			else {
				VideoCapture video = new VideoCapture(trailerFilename);
				double[] gradient = TrailerExtractor.extractMovementGradient(video);
				gradientMagnitudes.add(gradient[0]);
				gradientStdDevs.add(gradient[1]);
			}
		}
		
		double meanGradientMagnitude = gradientMagnitudes.stream().mapToDouble(mag -> mag).average().orElse(0);
		double meanGradientStdDev = gradientStdDevs.stream().mapToDouble(dev -> dev).average().orElse(0);
		
		System.out.println("Mean trailer gradient magnitude: " + meanGradientMagnitude);
		System.out.println("Mean trailer gradient standard deviation: " + meanGradientStdDev);
		// do something with the gradient
	}
	
}
