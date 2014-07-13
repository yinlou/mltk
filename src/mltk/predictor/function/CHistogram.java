package mltk.predictor.function;

/**
 * Class for cumulative histograms.
 * 
 * @author Yin Lou
 * 
 */
public class CHistogram {

	public double[] sum;
	public double[] count;

	/**
	 * Constructor.
	 * 
	 * @param n the size of this cumulative histogram.
	 */
	public CHistogram(int n) {
		sum = new double[n];
		count = new double[n];
	}

	/**
	 * Returns the size of this cumulative histogram.
	 * 
	 * @return the size of this cumulative histogram.
	 */
	public int size() {
		return sum.length;
	}

}
