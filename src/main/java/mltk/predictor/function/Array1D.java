package mltk.predictor.function;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.Arrays;

import mltk.core.Instance;
import mltk.predictor.Regressor;
import mltk.util.ArrayUtils;

/**
 * Class for 1D lookup tables.
 * 
 * @author Yin Lou
 * 
 */
public class Array1D implements Regressor, UnivariateFunction {

	/**
	 * Attribute index. Must be binned/nominal attribute; otherwise the behavior is not guaranteed.
	 */
	protected int attIndex;

	/**
	 * Predictions.
	 */
	protected double[] predictions;

	/**
	 * Constructor.
	 */
	public Array1D() {

	}

	/**
	 * Constructs a 1D lookup table.
	 * 
	 * @param attIndex the attribute index. The attribute must be discretized or nominal.
	 * @param predictions the prediction array.
	 */
	public Array1D(int attIndex, double[] predictions) {
		this.attIndex = attIndex;
		this.predictions = predictions;
	}

	/**
	 * Returns the attribute index.
	 * 
	 * @return the attribute index.
	 */
	public int getAttributeIndex() {
		return attIndex;
	}

	/**
	 * Sets the attribute index.
	 * 
	 * @param attIndex the new attribute index.
	 */
	public void setAttributeIndex(int attIndex) {
		this.attIndex = attIndex;
	}

	/**
	 * Returns the internal prediction array.
	 * 
	 * @return the internal prediction array.
	 */
	public double[] getPredictions() {
		return predictions;
	}

	/**
	 * Sets the internal prediction array.
	 * 
	 * @param predictions the new prediction array.
	 */
	public void setPredictions(double[] predictions) {
		this.predictions = predictions;
	}

	@Override
	public void read(BufferedReader in) throws Exception {
		String line = in.readLine();
		String[] data = line.split(": ");
		attIndex = Integer.parseInt(data[1]);

		in.readLine();
		line = in.readLine();
		predictions = ArrayUtils.parseDoubleArray(line);
	}

	@Override
	public void write(PrintWriter out) throws Exception {
		out.printf("[Predictor: %s]\n", this.getClass().getCanonicalName());
		out.println("AttIndex: " + attIndex);
		out.println("Predictions: " + predictions.length);
		out.println(Arrays.toString(predictions));
	}

	@Override
	public double regress(Instance instance) {
		int idx = (int) instance.getValue(attIndex);
		return predictions[idx];
	}

	/**
	 * Adds this lookup table with another one.
	 * 
	 * @param ary the other lookup table.
	 * @return this lookup table.
	 */
	public Array1D add(Array1D ary) {
		if (attIndex != ary.attIndex) {
			throw new IllegalArgumentException("Cannot add arrays on different terms");
		}
		for (int i = 0; i < predictions.length; i++) {
			predictions[i] += ary.predictions[i];
		}
		return this;
	}

	@Override
	public double evaluate(double x) {
		return predictions[(int) x];
	}

	@Override
	public Array1D copy() {
		double[] predictionsCopy = Arrays.copyOf(predictions, predictions.length);
		return new Array1D(attIndex, predictionsCopy);
	}

}
