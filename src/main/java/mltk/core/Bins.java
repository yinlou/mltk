package mltk.core;

import java.util.Arrays;

/**
 * Class for bins. Each bin is defined as its upper bound and median.
 * 
 * @author Yin Lou
 * 
 */
public class Bins {

	/**
	 * The upper bounds for each bin
	 */
	protected double[] boundaries;

	/**
	 * The medians for each bin
	 */
	protected double[] medians;

	protected Bins() {

	}

	/**
	 * Constructor.
	 * 
	 * @param boundaries the uppber bounds for each bin.
	 * @param medians the medians for each bin.
	 */
	public Bins(double[] boundaries, double[] medians) {
		if (boundaries.length != medians.length) {
			throw new IllegalArgumentException("Boundary size doesn't match medians size");
		}
		this.boundaries = boundaries;
		this.medians = medians;
	}

	/**
	 * Returns the number of bins.
	 * 
	 * @return the number of bins.
	 */
	public int size() {
		return boundaries.length;
	}

	/**
	 * Returns the bin index given a real value using binary search.
	 * 
	 * @param value the real value to discretize.
	 * @return the discretized index.
	 */
	public int getIndex(double value) {
		if (value < boundaries[0]) {
			return 0;
		} else if (value >= boundaries[boundaries.length - 1]) {
			return boundaries.length - 1;
		} else {
			int idx = Arrays.binarySearch(boundaries, value);
			if (idx < 0) {
				idx = -idx - 1;
			}
			return idx;
		}
	}

	/**
	 * Returns the median of a bin.
	 * 
	 * @param index the index of the bin.
	 * @return the median of the bin.
	 */
	public double getValue(int index) {
		return medians[index];
	}

	/**
	 * Returns the upper bounds for each bin.
	 * 
	 * @return the upper bounds for each bin.
	 */
	public double[] getBoundaries() {
		return boundaries;
	}

	/**
	 * Returns the medians for each bin.
	 * 
	 * @return the medians for each bin.
	 */
	public double[] getMedians() {
		return medians;
	}

}
