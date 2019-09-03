package mltk.predictor.function;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.Arrays;

import mltk.core.Instance;
import mltk.predictor.Regressor;
import mltk.util.ArrayUtils;
import mltk.util.VectorUtils;
import mltk.util.tuple.IntPair;

/**
 * Class for 2D functions.
 * 
 * <p>
 * This class represents a segmented 2D function. Segments are defined in split arrays for the two attributes. For
 * example, [3, 5, +INF] defines three segments: (-INF, 3], (3, 5], (5, +INF). The last value in the split array is
 * always +INF. The prediction matrix is the corresponding predictions for segments defined in splits.
 * </p>
 * 
 * @author Yin Lou
 * 
 */
public class Function2D implements Regressor, BivariateFunction {

	/**
	 * First attribute index.
	 */
	protected int attIndex1;

	/**
	 * Second attribute index.
	 */
	protected int attIndex2;
	
	/**
	 * Last value is always Double.POSITIVE_INFINITY. e.g. [3, 5, +INF] defines three segments: (-INF, 3], (3, 5], (5,
	 * +INF)
	 */
	protected double[] splits1;
	protected double[] splits2;

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
	public Function2D() {

	}

	/**
	 * Constructor.
	 * 
	 * @param attIndex1 the 1st attribute index.
	 * @param attIndex2 the 2nd attribute index.
	 * @param splits1 the split array for the 1st attribute.
	 * @param splits2 the split array for the 2nd attribute.
	 * @param predictions the prediction matrix.
	 */
	public Function2D(int attIndex1, int attIndex2, double[] splits1, double[] splits2, double[][] predictions) {
		this(attIndex1, attIndex2, splits1, splits2, predictions, new double[splits2.length], new double[splits1.length], 0.0);
	}
	
	/**
	 * Constructor.
	 * 
	 * @param attIndex1 the 1st attribute index.
	 * @param attIndex2 the 2nd attribute index.
	 * @param splits1 the split array for the 1st attribute.
	 * @param splits2 the split array for the 2nd attribute.
	 * @param predictions the prediction matrix.
	 * @param predictionsOnMV1 the prediction array when the 1st attribute is missing.
	 * @param predictionsOnMV2 the prediction array when the 2nd attribute is missing.
	 * @param predictionOnMV12 the prediction when both attributes are missing.
	 */
	public Function2D(int attIndex1, int attIndex2,
			double[] splits1, double[] splits2, double[][] predictions,
			double[] predictionsOnMV1, double[] predictionsOnMV2, double predictionOnMV12) {
		this.attIndex1 = attIndex1;
		this.attIndex2 = attIndex2;
		this.predictions = predictions;
		this.splits1 = splits1;
		this.splits2 = splits2;
		this.predictionsOnMV1 = predictionsOnMV1;
		this.predictionsOnMV2 = predictionsOnMV2;
		this.predictionOnMV12 = predictionOnMV12;
	}

	/**
	 * Returns a constant 2D function.
	 * 
	 * @param attIndex1 the 1st attribute index.
	 * @param attIndex2 the 2nd attribute index.
	 * @param prediction the constant.
	 * @return a constant 2D function.
	 */
	public static Function2D getConstantFunction(int attIndex1, int attIndex2, double prediction) {
		Function2D func = new Function2D(attIndex1, attIndex2, new double[] { Double.POSITIVE_INFINITY },
				new double[] { Double.POSITIVE_INFINITY }, new double[][] { { prediction } });
		return func;
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
	 * @param attIndex1 the new index for the 1st attribute.
	 * @param attIndex2 the new index for the 2nd attribute.
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
	 * Sets the prediction matrix.
	 * 
	 * @param predictions the new prediction matrix.
	 */
	public void setPredictions(double[][] predictions) {
		this.predictions = predictions;
	}
	
	/**
	 * Returns the prediction array when the 1st attribute is missing.
	 * 
	 * @return the prediction array when the 1st attribute is missing.
	 */
	public double[] getPredictionsOnMV1() {
		return predictionsOnMV1;
	}
	
	/**
	 * Sets the prediction array when the 1st attribute is missing.
	 * 
	 * @param predictionsOnMV1 the new prediction array.
	 */
	public void setPredictionsOnMV1(double[] predictionsOnMV1) {
		this.predictionsOnMV1 = predictionsOnMV1;
	}
	
	/**
	 * Returns the prediction array when the 2nd attribute is missing.
	 * 
	 * @return the prediction array when the 2nd attribute is missing.
	 */
	public double[] getPredictionsOnMV2() {
		return predictionsOnMV2;
	}
	
	/**
	 * Sets the prediction array when the 2nd attribute is missing.
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

	/**
	 * Multiplies this function with a constant.
	 * 
	 * @param c the constant.
	 * @return this function.
	 */
	public Function2D multiply(double c) {
		for (double[] preds : predictions) {
			VectorUtils.multiply(preds, c);
		}
		VectorUtils.multiply(predictionsOnMV1, c);
		VectorUtils.multiply(predictionsOnMV2, c);
		predictionOnMV12 *= c;
		return this;
	}

	/**
	 * Divides this function with a constant.
	 * 
	 * @param c the constant.
	 * @return this function.
	 */
	public Function2D divide(double c) {
		for (double[] preds : predictions) {
			VectorUtils.divide(preds, c);
		}
		VectorUtils.divide(predictionsOnMV1, c);
		VectorUtils.divide(predictionsOnMV2, c);
		predictionOnMV12 /= c;
		return this;
	}

	/**
	 * Adds this function with a constant.
	 * 
	 * @param c the constant.
	 * @return this function.
	 */
	public Function2D add(double c) {
		for (double[] preds : predictions) {
			VectorUtils.add(preds, c);
		}
		VectorUtils.add(predictionsOnMV1, c);
		VectorUtils.add(predictionsOnMV2, c);
		predictionOnMV12 += c;
		return this;
	}

	/**
	 * Subtracts this function with a constant.
	 * 
	 * @param c the constant.
	 * @return this function.
	 */
	public Function2D subtract(double c) {
		for (double[] preds : predictions) {
			VectorUtils.subtract(preds, c);
		}
		VectorUtils.subtract(predictionsOnMV1, c);
		VectorUtils.subtract(predictionsOnMV2, c);
		predictionOnMV12 -= c;
		return this;
	}

	/**
	 * Adds this function with another one.
	 * 
	 * @param func the other function.
	 * @return this function.
	 */
	public Function2D add(Function2D func) {
		if (attIndex1 != func.attIndex1 || attIndex2 != func.attIndex2) {
			throw new IllegalArgumentException("Cannot add arrays on differnt terms");
		}
		double[] s1 = splits1;
		int[] insertionPoints1 = new int[func.splits1.length - 1];
		int newElements1 = 0;
		for (int i = 0; i < insertionPoints1.length; i++) {
			insertionPoints1[i] = Arrays.binarySearch(splits1, func.splits1[i]);
			if (insertionPoints1[i] < 0) {
				newElements1++;
			}
		}
		if (newElements1 > 0) {
			double[] newSplits1 = new double[splits1.length + newElements1];
			System.arraycopy(splits1, 0, newSplits1, 0, splits1.length);
			int k = splits1.length;
			for (int i = 0; i < insertionPoints1.length; i++) {
				if (insertionPoints1[i] < 0) {
					newSplits1[k++] = func.splits1[i];
				}
			}
			Arrays.sort(newSplits1);
			s1 = newSplits1;
		}
		double[] s2 = splits2;
		int[] insertionPoints2 = new int[func.splits2.length - 1];
		int newElements2 = 0;
		for (int j = 0; j < insertionPoints2.length; j++) {
			insertionPoints2[j] = Arrays.binarySearch(splits2, func.splits2[j]);
			if (insertionPoints2[j] < 0) {
				newElements2++;
			}
		}
		if (newElements2 > 0) {
			double[] newSplits2 = new double[splits2.length + newElements2];
			System.arraycopy(splits2, 0, newSplits2, 0, splits2.length);
			int k = splits2.length;
			for (int j = 0; j < insertionPoints2.length; j++) {
				if (insertionPoints2[j] < 0) {
					newSplits2[k++] = func.splits2[j];
				}
			}
			Arrays.sort(newSplits2);
			s2 = newSplits2;
		}

		if (newElements1 == 0 && newElements2 == 0) {
			for (int i = 0; i < splits1.length; i++) {
				predictionsOnMV2[i] += func.evaluate(splits1[i], Double.NaN);
				
				double[] ps = predictions[i];
				for (int j = 0; j < splits2.length; j++) {
					predictionsOnMV1[j] += func.evaluate(Double.NaN, splits2[j]);
					
					ps[j] += func.evaluate(splits1[i], splits2[j]);
				}
			}
			predictionOnMV12 += func.predictionOnMV12;
		} else {
			double[][] newPredictions = new double[s1.length][s2.length];
			predictionsOnMV1 = new double[s2.length];
			predictionsOnMV2 = new double[s1.length];
			for (int i = 0; i < s1.length; i++) {
				predictionsOnMV2[i] = this.evaluate(s1[i], Double.NaN) + func.evaluate(s1[i], Double.NaN);
				
				double[] ps = newPredictions[i];
				for (int j = 0; j < s2.length; j++) {
					predictionsOnMV1[j] = this.evaluate(Double.NaN, s2[j]) + func.evaluate(Double.NaN, s2[j]);
					
					ps[j] = this.evaluate(s1[i], s2[j]) + func.evaluate(s1[i], s2[j]);
				}
			}
			splits1 = s1;
			splits2 = s2;
			predictions = newPredictions;
			predictionOnMV12 = this.predictionOnMV12 + func.predictionOnMV12;
		}
		
		return this;
	}

	@Override
	public void read(BufferedReader in) throws Exception {
		String line = in.readLine();
		String[] data = line.split(": ");
		attIndex1 = Integer.parseInt(data[1]);
		line = in.readLine();
		data = line.split(": ");
		attIndex2 = Integer.parseInt(data[1]);

		in.readLine();
		line = in.readLine();
		splits1 = ArrayUtils.parseDoubleArray(line);

		in.readLine();
		line = in.readLine();
		splits2 = ArrayUtils.parseDoubleArray(line);

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
		out.println("Splits1: " + splits1.length);
		out.println(Arrays.toString(splits1));
		out.println("Splits2: " + splits2.length);
		out.println(Arrays.toString(splits2));
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

	@Override
	public double evaluate(double x, double y) {
		if (!Double.isNaN(x) && !Double.isNaN(y)) {
			IntPair idx = getSegmentIndex(x, y);
			return predictions[idx.v1][idx.v2];
		} else if (Double.isNaN(x) && !Double.isNaN(y)) {
			int idx = ArrayUtils.findInsertionPoint(splits2, y);
			return predictionsOnMV1[idx];
		} else if (!Double.isNaN(x) && Double.isNaN(y)) {
			int idx = ArrayUtils.findInsertionPoint(splits1, x);
			return predictionsOnMV2[idx];
		} else {
			return predictionOnMV12;
		}
	}

	@Override
	public Function2D copy() {
		double[] splits1Copy = Arrays.copyOf(splits1, splits1.length);
		double[] splits2Copy = Arrays.copyOf(splits2, splits2.length);
		double[][] predictionsCopy = new double[predictions.length][];
		for (int i = 0; i < predictionsCopy.length; i++) {
			predictionsCopy[i] = Arrays.copyOf(predictions[i], predictions[i].length);
		}
		double[] predictionsOnMV1Copy = Arrays.copyOf(predictionsOnMV1, predictionsOnMV1.length);
		double[] predictionsOnMV2Copy = Arrays.copyOf(predictionsOnMV2, predictionsOnMV2.length);
		return new Function2D(attIndex1, attIndex2, splits1Copy, splits2Copy, predictionsCopy,
				predictionsOnMV1Copy, predictionsOnMV2Copy, predictionOnMV12);
	}
	
	/**
	 * Returns the segment indices pair given (x1, x2). Assume x1 and x2 are not missing values.
	 * 
	 * @param x1 the 1st search key.
	 * @param x2 the 2nd search key.
	 * @return segment indices pair at (x1, x2).
	 */
	protected IntPair getSegmentIndex(double x1, double x2) {
		int idx1 = ArrayUtils.findInsertionPoint(splits1, x1);
		int idx2 = ArrayUtils.findInsertionPoint(splits2, x2);
		return new IntPair(idx1, idx2);
	}

}
