package mltk.predictor.io;

import java.io.BufferedReader;
import java.io.FileReader;

import mltk.predictor.Predictor;

/**
 * Class for reading predictors.
 * 
 * @author Yin Lou
 * 
 */
public class PredictorReader {

	/**
	 * Reads a predictor. The caller is responsible for converting the predictor to correct type.
	 * 
	 * @param path the file path for the predictor.
	 * @return the parsed predictor.
	 * @throws Exception
	 */
	public static Predictor read(String path) throws Exception {
		BufferedReader in = new BufferedReader(new FileReader(path));
		String line = in.readLine();
		String predictorName = line.substring(1, line.length() - 1).split(": ")[1];
		Class<?> clazz = Class.forName(predictorName);
		Predictor predictor = (Predictor) clazz.newInstance();
		predictor.read(in);
		in.close();
		return predictor;
	}

	/**
	 * Reads a predictor. The caller is responsible for providing the correct predictor type.
	 * 
	 * @param path the file path for the predictor.
	 * @param clazz the class of the predictor.
	 * @return the parsed predictor.
	 * @throws Exception
	 */
	public static <T extends Predictor> T read(String path, Class<T> clazz) throws Exception {
		Predictor predictor = read(path);
		return clazz.cast(predictor);
	}

	/**
	 * Reads a predictor from an input reader. The caller is responsible for converting the predictor to correct type.
	 * 
	 * @param in the input reader.
	 * @return the parsed predictor.
	 * @throws Exception
	 */
	public static Predictor read(BufferedReader in) throws Exception {
		String line = in.readLine();
		String predictorName = line.substring(1, line.length() - 1).split(": ")[1];
		Class<?> clazz = Class.forName(predictorName);
		Predictor predictor = (Predictor) clazz.newInstance();
		predictor.read(in);
		return predictor;
	}

}
