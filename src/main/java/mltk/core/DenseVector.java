package mltk.core;

import java.util.Arrays;

/**
 * Class for dense vectors.
 * 
 * @author Yin Lou
 * 
 */
public class DenseVector implements Vector {

	protected double[] values;

	/**
	 * Constructs a dense vector from a double array.
	 * 
	 * @param values the double array.
	 */
	public DenseVector(double[] values) {
		this.values = values;
	}

	@Override
	public double getValue(int index) {
		return values[index];
	}

	@Override
	public double[] getValues() {
		return values;
	}

	@Override
	public double[] getValues(int... indices) {
		double[] values = new double[indices.length];
		for (int i = 0; i < values.length; i++) {
			values[i] = getValue(indices[i]);
		}
		return values;
	}

	@Override
	public void setValue(int index, double value) {
		values[index] = value;
	}

	@Override
	public void setValue(int[] indices, double[] v) {
		for (int i = 0; i < indices.length; i++) {
			values[indices[i]] = v[i];
		}
	}

	@Override
	public DenseVector copy() {
		double[] copyValues = Arrays.copyOf(values, values.length);
		return new DenseVector(copyValues);
	}

	@Override
	public boolean isSparse() {
		return false;
	}

}
