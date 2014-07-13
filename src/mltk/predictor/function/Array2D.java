package mltk.predictor.function;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.Arrays;

import mltk.core.Instance;
import mltk.predictor.Regressor;
import mltk.util.ArrayUtils;
import mltk.util.tuple.IntPair;

/**
 * Class for 2D lookup tables.
 * 
 * @author Yin Lou
 * 
 */
public class Array2D implements Regressor, BivariateFunction {

	/**
	 * First attribute index.
	 */
	protected int attIndex1;

	/**
	 * Second attribute index.
	 */
	protected int attIndex2;

	/**
	 * Predictions.
	 */
	protected double[][] predictions;

	/**
	 * Constructor.
	 */
	public Array2D() {

	}

	/**
	 * Constructs a 2D lookup table.
	 * 
	 * @param attIndex1 the 1st attribute index. The attribute must be discretized or nominal.
	 * @param attIndex2 the 2nd attribute index. The attribute must be discretized or nominal.
	 * @param predictions the prediction matrix.
	 */
	public Array2D(int attIndex1, int attIndex2, double[][] predictions) {
		this.attIndex1 = attIndex1;
		this.attIndex2 = attIndex2;
		this.predictions = predictions;
	}

	/**
	 * Returns the index of 1st attribute.
	 * 
	 * @return the index of 1st attribute.
	 */
	public int getAttributeIndex1() {
		return attIndex1;
	}

	/**
	 * Returns the index of 2nd attribute.
	 * 
	 * @return the index of 2nd attribute.
	 */
	public int getAttributeIndex2() {
		return attIndex2;
	}

	/**
	 * Returns the attribute indices pair.
	 * 
	 * @return the attribute indices pair.
	 */
	public IntPair getAttributeIndices() {
		return new IntPair(attIndex1, attIndex2);
	}

	/**
	 * Sets the attribute indices.
	 * 
	 * @param attIndex1 the new 1st attribute index.
	 * @param attIndex2 the new 2nd attribute index.
	 */
	public void setAttributeIndices(int attIndex1, int attIndex2) {
		this.attIndex1 = attIndex1;
		this.attIndex2 = attIndex2;
	}

	/**
	 * Returns the internal prediction matrix.
	 * 
	 * @return the internal prediction matrix.
	 */
	public double[][] getPredictions() {
		return predictions;
	}

	/**
	 * Sets the internal prediction matrix.
	 * 
	 * @param predictions the new prediction matrix.
	 */
	public void setPredictions(double[][] predictions) {
		this.predictions = predictions;
	}

	@Override
	public void read(BufferedReader in) throws Exception {
		String line = in.readLine();
		String[] data = line.split(": ");
		attIndex1 = Integer.parseInt(data[1]);
		line = in.readLine();
		data = line.split(": ");
		attIndex2 = Integer.parseInt(data[1]);

		String[] dim = in.readLine().split(": ")[1].split("x");
		predictions = new double[Integer.parseInt(dim[0])][];
		for (int i = 0; i < predictions.length; i++) {
			predictions[i] = ArrayUtils.parseDoubleArray(in.readLine());
		}
	}

	@Override
	public void write(PrintWriter out) throws Exception {
		out.printf("[Predictor: %s]\n", this.getClass().getCanonicalName());
		out.println("AttIndex1: " + attIndex1);
		out.println("AttIndex2: " + attIndex2);
		out.println("Predictions: " + predictions.length + "x" + predictions[0].length);
		for (int i = 0; i < predictions.length; i++) {
			out.println(Arrays.toString(predictions[i]));
		}
	}

	@Override
	public double regress(Instance instance) {
		int idx1 = (int) instance.getValue(attIndex1);
		int idx2 = (int) instance.getValue(attIndex2);
		return predictions[idx1][idx2];
	}

	/**
	 * Adds this lookup table with another one.
	 * 
	 * @param ary the other lookup table.
	 * @return this lookup table.
	 */
	public Array2D add(Array2D ary) {
		if (attIndex1 != ary.attIndex1 || attIndex2 != ary.attIndex2) {
			throw new IllegalArgumentException("Cannot add arrays on differnt terms");
		}
		for (int i = 0; i < predictions.length; i++) {
			double[] preds1 = predictions[i];
			double[] preds2 = ary.predictions[i];
			for (int j = 0; j < preds1.length; j++) {
				preds1[j] += preds2[j];
			}
		}
		return this;
	}

	@Override
	public double evaluate(double x, double y) {
		return predictions[(int) x][(int) y];
	}

	@Override
	public Array2D copy() {
		double[][] predictionsCopy = new double[predictions.length][];
		for (int i = 0; i < predictionsCopy.length; i++) {
			predictionsCopy[i] = Arrays.copyOf(predictions[i], predictions[i].length);
		}
		return new Array2D(attIndex1, attIndex2, predictionsCopy);
	}

}
