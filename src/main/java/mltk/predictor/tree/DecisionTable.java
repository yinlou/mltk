package mltk.predictor.tree;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.Arrays;

import mltk.core.Instance;
import mltk.util.ArrayUtils;
import mltk.util.VectorUtils;

/**
 * Class for decision tables.
 * 
 * @author Yin Lou
 *
 */
public class DecisionTable implements RTree {

	protected int[] attIndices;
	protected double[] splits;
	protected long[] predIndices;
	protected double[] predValues;
	
	/**
	 * Constructor.
	 */
	public DecisionTable() {
		
	}
	
	/**
	 * Constructor.
	 * 
	 * @param attIndices the attribute indices.
	 * @param splits the splits.
	 * @param predIndices the prediction indices.
	 * @param predValues the prediction values.
	 */
	public DecisionTable(int[] attIndices, double[] splits,
			long[] predIndices, double[] predValues) {
		this.attIndices = attIndices;
		this.splits = splits;
		this.predIndices = predIndices;
		this.predValues = predValues;
	}
	
	/**
	 * Returns the attribute indices in this tree.
	 * 
	 * @return the attribute indices in this tree.
	 */
	public int[] getAttributeIndices() {
		return attIndices;
	}
	
	/**
	 * Returns the splits in this tree.
	 * 
	 * @return the splits in this tree.
	 */
	public double[] getSplits() {
		return splits;
	}
	
	@Override
	public void multiply(double c) {
		VectorUtils.multiply(predValues, c);
	}

	@Override
	public void read(BufferedReader in) throws Exception {
		in.readLine();
		attIndices = ArrayUtils.parseIntArray(in.readLine());
		in.readLine();
		splits = ArrayUtils.parseDoubleArray(in.readLine());
		in.readLine();
		predIndices = ArrayUtils.parseLongArray(in.readLine());
		in.readLine();
		predValues = ArrayUtils.parseDoubleArray(in.readLine());
	}

	@Override
	public void write(PrintWriter out) throws Exception {
		out.printf("[Predictor: %s]\n", this.getClass().getCanonicalName());
		out.println("Attributes: " + attIndices.length);
		out.println(Arrays.toString(attIndices));
		out.println("Splits: " + splits.length);
		out.println(Arrays.toString(splits));
		out.println("Prediction Indices: " + predIndices.length);
		out.println(Arrays.toString(predIndices));
		out.println("Prediction Values: " + predValues.length);
		out.println(Arrays.toString(predValues));
	}

	@Override
	public DecisionTable copy() {
		int[] attIndicesCopy = Arrays.copyOf(attIndices, attIndices.length);
		double[] splitsCopy = Arrays.copyOf(splits, splits.length);
		long[] predIndicesCopy = Arrays.copyOf(predIndices, predIndices.length);
		double[] predValuesCopy = Arrays.copyOf(predValues, predValues.length);
		return new DecisionTable(attIndicesCopy, splitsCopy, predIndicesCopy, predValuesCopy);
	}

	@Override
	public double regress(Instance instance) {
		long predIdx = 0L;
		for (int j = 0; j < attIndices.length; j++) {
			int attIndex = attIndices[j];
			double split = splits[j];
			if (instance.getValue(attIndex) <= split) {
				predIdx = (predIdx << 1) | 1L;
			} else {
				predIdx <<= 1;
			}
		}
		return regress(predIdx);
	}
	
	/**
	 * Returns the prediction based on prediction index.
	 * 
	 * @param predIdx
	 * @return the prediction based on prediction index.
	 */
	public double regress(long predIdx) {
		int idx = Arrays.binarySearch(predIndices, predIdx);
		if (idx < 0) {
			return 0;
		} else {
			return predValues[idx];
		}
	}
	
}
