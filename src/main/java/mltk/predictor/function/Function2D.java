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
	 * Predictions.
	 */
	protected double[][] predictions;

	/**
	 * Last value is always Double.POSITIVE_INFINITY. e.g. [3, 5, +INF] defines three segments: (-INF, 3], (3, 5], (5,
	 * +INF)
	 */
	protected double[] splits1;
	protected double[] splits2;

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
		this.attIndex1 = attIndex1;
		this.attIndex2 = attIndex2;
		this.predictions = predictions;
		this.splits1 = splits1;
		this.splits2 = splits2;
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
	 * Multiplies this function with a constant.
	 * 
	 * @param c the constant.
	 * @return this function.
	 */
	public Function2D multiply(double c) {
		for (double[] preds : predictions) {
			VectorUtils.multiply(preds, c);
		}
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
		for (int i = 0; i < insertionPoints2.length; i++) {
			insertionPoints2[i] = Arrays.binarySearch(splits2, func.splits2[i]);
			if (insertionPoints2[i] < 0) {
				newElements2++;
			}
		}
		if (newElements2 > 0) {
			double[] newSplits2 = new double[splits2.length + newElements2];
			System.arraycopy(splits2, 0, newSplits2, 0, splits2.length);
			int k = splits2.length;
			for (int i = 0; i < insertionPoints2.length; i++) {
				if (insertionPoints2[i] < 0) {
					newSplits2[k++] = func.splits2[i];
				}
			}
			Arrays.sort(newSplits2);
			s2 = newSplits2;
		}

		if (newElements1 == 0 && newElements2 == 0) {
			for (int i = 0; i < splits1.length; i++) {
				for (int j = 0; j < splits2.length; j++) {
					predictions[i][j] += func.evaluate(splits1[i], splits2[j]);
				}
			}
		} else {
			double[][] newPredictions = new double[s1.length][s2.length];
			for (int i = 0; i < s1.length; i++) {
				for (int j = 0; j < s2.length; j++) {
					newPredictions[i][j] = this.evaluate(s1[i], s2[j]) + func.evaluate(s1[i], s2[j]);
				}
			}
			splits1 = s1;
			splits2 = s2;
			predictions = newPredictions;
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
	}

	/**
	 * Returns the segment indices pair given (x1, x2).
	 * 
	 * @param x1 the 1st argument.
	 * @param x2 the 2nd argument.
	 * @return segment indices pair at (x1, x2).
	 */
	public IntPair getSegmentIndex(double x1, double x2) {
		int idx1 = Arrays.binarySearch(splits1, x1);
		if (idx1 < 0) {
			idx1 = -idx1 - 1;
		}
		int idx2 = Arrays.binarySearch(splits2, x2);
		if (idx2 < 0) {
			idx2 = -idx2 - 1;
		}
		return new IntPair(idx1, idx2);
	}

	/**
	 * Returns the segment indices pair.
	 * 
	 * @param instance the instance.
	 * @return the segment indices pair.
	 */
	public IntPair getSegmentIndex(Instance instance) {
		int key1 = (int) instance.getValue(attIndex1);
		int key2 = (int) instance.getValue(attIndex2);
		return getSegmentIndex(key1, key2);
	}

	@Override
	public double regress(Instance instance) {
		IntPair idx = getSegmentIndex(instance);
		return predictions[idx.v1][idx.v2];
	}

	@Override
	public double evaluate(double x, double y) {
		IntPair idx = getSegmentIndex(x, y);
		return predictions[idx.v1][idx.v2];
	}

	@Override
	public Function2D copy() {
		double[] splits1Copy = Arrays.copyOf(splits1, splits1.length);
		double[] splits2Copy = Arrays.copyOf(splits2, splits2.length);
		double[][] predictionsCopy = new double[predictions.length][];
		for (int i = 0; i < predictionsCopy.length; i++) {
			predictionsCopy[i] = Arrays.copyOf(predictions[i], predictions[i].length);
		}
		return new Function2D(attIndex1, attIndex2, splits1Copy, splits2Copy, predictionsCopy);
	}

}
