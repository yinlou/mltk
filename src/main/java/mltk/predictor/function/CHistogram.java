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
	public double sumOnMV;
	public double countOnMV;

	/**
	 * Constructor.
	 * 
	 * @param n the size of this cumulative histogram.
	 */
	public CHistogram(int n) {
		sum = new double[n];
		count = new double[n];
		sumOnMV = 0.0;
		countOnMV = 0.0;
	}

	/**
	 * Returns the size of this cumulative histogram.
	 * 
	 * @return the size of this cumulative histogram.
	 */
	public int size() {
		return sum.length;
	}
	
	/**
	 * Returns {@code true} if missing values are present.
	 * 
	 * @return {@code true} if missing values are present.
	 */
	public boolean hasMissingValue() {
		return countOnMV > 0;
	}

}
