package mltk.predictor.io;

import java.io.PrintWriter;

import mltk.predictor.Predictor;

/**
 * Class for writing predictors.
 * 
 * @author Yin Lou
 * 
 */
public class PredictorWriter {

	/**
	 * Writes a predictor to file.
	 * 
	 * @param predictor the predictor to write.
	 * @param path the file path.
	 * @throws Exception
	 */
	public static void write(Predictor predictor, String path) throws Exception {
		PrintWriter out = new PrintWriter(path);
		predictor.write(out);
		out.flush();
		out.close();
	}

}
