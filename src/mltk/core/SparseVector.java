package mltk.core;

import java.util.Arrays;

/**
 * Class for sparse vectors.
 * 
 * @author Yin Lou
 * 
 */
public class SparseVector implements Vector {

	protected int[] indices;
	protected double[] values;

	/**
	 * Constructs a sparse vector from sparse-format arrays.
	 * 
	 * @param indices the indices array.
	 * @param values the values array.
	 */
	public SparseVector(int[] indices, double[] values) {
		this.indices = indices;
		this.values = values;
	}

	@Override
	public SparseVector copy() {
		int[] copyIndices = Arrays.copyOf(indices, indices.length);
		double[] copyValues = Arrays.copyOf(values, values.length);
		return new SparseVector(copyIndices, copyValues);
	}

	@Override
	public double getValue(int index) {
		int idx = Arrays.binarySearch(indices, index);
		if (idx >= 0) {
			return values[idx];
		} else {
			return 0;
		}
	}

	@Override
	public double[] getValues() {
		return values;
	}

	/**
	 * Returns the internal representation of indices.
	 * 
	 * @return the internal representation of indices.
	 */
	public int[] getIndices() {
		return indices;
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
		int idx = Arrays.binarySearch(indices, index);
		if (idx >= 0) {
			values[idx] = value;
		} else {
			throw new UnsupportedOperationException();
		}
	}

	@Override
	public void setValue(int[] indices, double[] v) {
		for (int i = 0; i < indices.length; i++) {
			setValue(indices[i], v[i]);
		}
	}

	@Override
	public boolean isSparse() {
		return true;
	}

}
