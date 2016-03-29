package mltk.core;

import java.util.Arrays;

import mltk.util.ArrayUtils;

/**
 * Class for discretized attributes.
 * 
 * @author Yin Lou
 * 
 */
public class BinnedAttribute extends Attribute {

	protected int numBins;
	protected Bins bins;

	/**
	 * Constructor.
	 * 
	 * @param name the name of this attribute.
	 * @param numBins number of bins for this attribute.
	 */
	public BinnedAttribute(String name, int numBins) {
		this(name, numBins, -1);
	}

	/**
	 * Constructor.
	 * 
	 * @param name the name of this attribute.
	 * @param numBins number of bins for this attribute.
	 * @param index the index of this attribute.
	 */
	public BinnedAttribute(String name, int numBins, int index) {
		this.name = name;
		this.numBins = numBins;
		this.bins = null;
		this.index = index;
		this.type = Type.BINNED;
	}

	/**
	 * Constructor.
	 * 
	 * @param name the name of this attribute.
	 * @param bins bins for this attribute.
	 */
	public BinnedAttribute(String name, Bins bins) {
		this(name, bins, -1);
	}

	/**
	 * Constructor.
	 * 
	 * @param name the name of this attribute.
	 * @param bins bins for this attribute.
	 * @param index the index of this attribute.
	 */
	public BinnedAttribute(String name, Bins bins, int index) {
		this(name, bins.size());
		this.bins = bins;
		this.index = index;
	}

	@Override
	public BinnedAttribute copy() {
		BinnedAttribute copy = (bins == null ? new BinnedAttribute(name, numBins) : new BinnedAttribute(name, bins));
		copy.index = index;
		return copy;
	}

	/**
	 * Returns the number of bins.
	 * 
	 * @return the number of bins.
	 */
	public int getNumBins() {
		return numBins;
	}

	/**
	 * Returns the bins.
	 * 
	 * @return the bins.
	 */
	public Bins getBins() {
		return bins;
	}

	public String toString() {
		if (bins == null) {
			return name + ": binned (" + numBins + ")";
		} else {
			return name + ": binned (" + bins.size() + ";" + Arrays.toString(bins.boundaries) + ";"
					+ Arrays.toString(bins.medians) + ")";
		}
	}

	/**
	 * Parses a binned attribute object from a string.
	 * 
	 * @param str the string.
	 * @return a parsed binned attribute.
	 */
	public static BinnedAttribute parse(String str) {
		String[] data = str.split(": ");
		int start = data[1].indexOf('(') + 1;
		int end = data[1].indexOf(')');
		String[] strs = data[1].substring(start, end).split(";");
		int numBins = Integer.parseInt(strs[0]);
		if (strs.length == 1) {
			return new BinnedAttribute(data[0], numBins);
		} else {
			double[] boundaries = ArrayUtils.parseDoubleArray(strs[1]);
			double[] medians = ArrayUtils.parseDoubleArray(strs[2]);
			Bins bins = new Bins(boundaries, medians);
			return new BinnedAttribute(data[0], bins);
		}
	}

}
