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
	 * Predictions on missing value for attribute 1.
	 */
	protected double[] predictionsOnMV1;
	
	/**
	 * Predictions on missing value for attribute 2.
	 */
	protected double[] predictionsOnMV2;
	
	/**
	 * Prediction when both attribute 1 and 2 are missing.
	 */
	protected double predictionOnMV12;

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
	 * Constructs a 2D lookup table.
	 * 
	 * @param attIndex1 the 1st attribute index. The attribute must be discretized or nominal.
	 * @param attIndex2 the 2nd attribute index. The attribute must be discretized or nominal.
	 * @param predictions the prediction matrix.
	 * @param predictionsOnMV1 the prediction array when the 1st attribute is missing.
	 * @param predictionsOnMV2 the prediction array when the 2nd attribute is missing.
	 * @param predictionOnMV12 the prediction when both attributes are missing.
	 */
	public Array2D(int attIndex1, int attIndex2, double[][] predictions,
			double[] predictionsOnMV1, double[] predictionsOnMV2, double predictionOnMV12) {
		this.attIndex1 = attIndex1;
		this.attIndex2 = attIndex2;
		this.predictions = predictions;
		this.predictionsOnMV1 = predictionsOnMV1;
		this.predictionsOnMV2 = predictionsOnMV2;
		this.predictionOnMV12 = predictionOnMV12;
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
	
	/**
	 * Returns the internal prediction array when the 1st attribute is missing.
	 * 
	 * @return the internal prediction array when the 1st attribute is missing.
	 */
	public double[] getPredictionsOnMV1() {
		return predictionsOnMV1;
	}
	
	/**
	 * Sets the internal prediction array when the 1st attribute is missing.
	 * 
	 * @param predictionsOnMV1 the new prediction array.
	 */
	public void setPredictionsOnMV1(double[] predictionsOnMV1) {
		this.predictionsOnMV1 = predictionsOnMV1;
	}
	
	/**
	 * Returns the internal prediction array when the 2nd attribute is missing.
	 * 
	 * @return the internal prediction array when the 2nd attribute is missing.
	 */
	public double[] getPredictionsOnMV2() {
		return predictionsOnMV2;
	}
	
	/**
	 * Sets the internal prediction array when the 2nd attribute is missing.
	 * 
	 * @param predictionsOnMV2 the new prediction array.
	 */
	public void setPredictionsOnMV2(double[] predictionsOnMV2) {
		this.predictionsOnMV2 = predictionsOnMV2;
	}
	
	/**
	 * Returns the prediction when both attributes are missing.
	 * 
	 * @return the prediction when both attributes are missing.
	 */
	public double getPredictionOnMV12() {
		return predictionOnMV12;
	}
	
	/**
	 * Sets the prediction when both attributes are missing.
	 * 
	 * @param predictionOnMV12 the new prediction.
	 */
	public void setPredictionOnMV12(double predictionOnMV12) {
		this.predictionOnMV12 = predictionOnMV12;
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
		
		in.readLine();
		predictionsOnMV1 = ArrayUtils.parseDoubleArray(in.readLine());
		
		in.readLine();
		predictionsOnMV2 = ArrayUtils.parseDoubleArray(in.readLine());
		
		data = in.readLine().split(": ");
		predictionOnMV12 = Double.parseDouble(data[1]);
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
		out.println("PredictionsOnMV1: " + predictionsOnMV1.length);
		out.println(Arrays.toString(predictionsOnMV1));
		out.println("PredictionsOnMV2: " + predictionsOnMV2.length);
		out.println(Arrays.toString(predictionsOnMV2));
		out.println("PredictionOnMV12: " + predictionOnMV12);
	}

	@Override
	public double regress(Instance instance) {
		return evaluate(instance.getValue(attIndex1), instance.getValue(attIndex2));
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
			predictionsOnMV2[i] += ary.predictionsOnMV2[i];
			
			double[] preds1 = predictions[i];
			double[] preds2 = ary.predictions[i];
			for (int j = 0; j < preds1.length; j++) {
				predictionsOnMV1[j] += ary.predictionsOnMV1[j];
				
				preds1[j] += preds2[j];
			}
		}
		predictionOnMV12 += ary.predictionOnMV12;
		return this;
	}

	@Override
	public double evaluate(double x, double y) {
		if (!Double.isNaN(x) && !Double.isNaN(y)) {
			return predictions[(int) x][(int) y];
		} else if (Double.isNaN(x) && !Double.isNaN(y)) {
			return predictionsOnMV1[(int) y];
		} else if (!Double.isNaN(x) && Double.isNaN(y)) {
			return predictionsOnMV2[(int) x];
		} else {
			return predictionOnMV12;
		}
		
	}

	@Override
	public Array2D copy() {
		double[][] predictionsCopy = new double[predictions.length][];
		for (int i = 0; i < predictionsCopy.length; i++) {
			predictionsCopy[i] = Arrays.copyOf(predictions[i], predictions[i].length);
		}
		double[] predictionsOnMV1Copy = Arrays.copyOf(predictionsOnMV1, predictionsOnMV1.length);
		double[] predictionsOnMV2Copy = Arrays.copyOf(predictionsOnMV2, predictionsOnMV2.length);
		return new Array2D(attIndex1, attIndex2, predictionsCopy,
				predictionsOnMV1Copy, predictionsOnMV2Copy, predictionOnMV12);
	}

}
