package mltk.util;

/**
 * Class for utility functions for computing statistics.
 * 
 * @author Yin Lou
 * 
 */
public class StatUtils {
	
	/**
	 * Returns the maximum element in an array.
	 * 
	 * @param a the array.
	 * @return the maximum element in an array.
	 */
	public static int max(int[] a) {
		int max = a[0];
		for (int i = 1; i < a.length; i++) {
			if (a[i] > max) {
				max = a[i];
			}
		}
		return max;
	}

	/**
	 * Returns the maximum element in an array.
	 * 
	 * @param a the array.
	 * @return the maximum element in an array.
	 */
	public static double max(double[] a) {
		double max = a[0];
		for (int i = 1; i < a.length; i++) {
			if (a[i] > max) {
				max = a[i];
			}
		}
		return max;
	}
	
	/**
	 * Returns the index of maximum element.
	 * 
	 * @return the index of maximum element.
	 */
	public static int indexOfMax(int[] a) {
		int max = a[0];
		int idx = 0;
		for (int i = 1; i < a.length; i++) {
			if (a[i] > max) {
				max = a[i];
				idx = i;
			}
		}
		return idx;
	}

	/**
	 * Returns the index of maximum element.
	 * 
	 * @return the index of maximum element.
	 */
	public static int indexOfMax(double[] a) {
		double max = a[0];
		int idx = 0;
		for (int i = 1; i < a.length; i++) {
			if (a[i] > max) {
				max = a[i];
				idx = i;
			}
		}
		return idx;
	}
	
	/**
	 * Returns the minimum element in an array.
	 * 
	 * @param a the array.
	 * @return the minimum element in an array.
	 */
	public static int min(int[] a) {
		int min = a[0];
		for (int i = 1; i < a.length; i++) {
			if (a[i] < min) {
				min = a[i];
			}
		}
		return min;
	}

	/**
	 * Returns the minimum element in an array.
	 * 
	 * @param a the array.
	 * @return the minimum element in an array.
	 */
	public static double min(double[] a) {
		double min = a[0];
		for (int i = 1; i < a.length; i++) {
			if (a[i] < min) {
				min = a[i];
			}
		}
		return min;
	}
	
	/**
	 * Returns the index of minimum element.
	 * 
	 * @return the index of minimum element.
	 */
	public static int indexOfMin(int[] a) {
		int min = a[0];
		int idx = 0;
		for (int i = 1; i < a.length; i++) {
			if (a[i] < min) {
				min = a[i];
				idx = i;
			}
		}
		return idx;
	}

	/**
	 * Returns the index of minimum element.
	 * 
	 * @return the index of minimum element.
	 */
	public static int indexOfMin(double[] a) {
		double min = a[0];
		int idx = 0;
		for (int i = 1; i < a.length; i++) {
			if (a[i] < min) {
				min = a[i];
				idx = i;
			}
		}
		return idx;
	}

	/**
	 * Returns the sum of elements in an array.
	 * 
	 * @param a the array.
	 * @return the sum of elements in an array.
	 */
	public static double sum(double[] a) {
		double sum = 0;
		for (double v : a) {
			sum += v;
		}
		return sum;
	}

	/**
	 * Returns the sum of squares.
	 * 
	 * @param a the array.
	 * @return the sum of squares.
	 */
	public static double sumSq(double[] a) {
		return sumSq(a, 0, a.length);
	}

	/**
	 * Returns the sum of squares within a specific range.
	 * 
	 * @param a the array.
	 * @param fromIndex the index of the first element (inclusive).
	 * @param toIndex the index of the last element (exclusive).
	 * @return the sum of squares.
	 */
	public static double sumSq(double[] a, int fromIndex, int toIndex) {
		double sq = 0.0;
		for (int i = fromIndex; i < toIndex; i++) {
			sq += a[i] * a[i];
		}
		return sq;
	}

	/**
	 * Returns the mean.
	 * 
	 * @param a the array.
	 * @return the mean.
	 */
	public static double mean(double[] a) {
		return mean(a, a.length);
	}

	/**
	 * Returns the mean.
	 * 
	 * @param a the array.
	 * @param n the total number of elements.
	 * @return the mean.
	 */
	public static double mean(double[] a, int n) {
		double avg = 0.0;
		for (double v : a) {
			avg += v;
		}
		return avg / n;
	}

	/**
	 * Returns the variance.
	 * 
	 * @param a the array.
	 * @return the variance.
	 */
	public static double variance(double[] a) {
		return variance(a, a.length);
	}

	/**
	 * Returns the variance.
	 * 
	 * @param a the array.
	 * @param n the total number of elements.
	 * @return the variance.
	 */
	public static double variance(double[] a, int n) {
		double sq = 0.0;
		double avg = 0.0;
		for (double v : a) {
			sq += v * v;
			avg += v;
		}
		avg /= n;
		return sq / (n - 1.0) - avg * avg * n / (n - 1.0);
	}

	/**
	 * Returns the standard variance.
	 * 
	 * @param a the array.
	 * @return the standard variance.
	 */
	public static double std(double[] a) {
		return std(a, a.length);
	}

	/**
	 * Returns the standard variance.
	 * 
	 * @param a the array.
	 * @param n the total number of elements.
	 * @return the standard variance.
	 */
	public static double std(double[] a, int n) {
		return Math.sqrt(variance(a, n));
	}

	/**
	 * Returns the root mean square.
	 * 
	 * @param a the array.
	 * @return the root mean square.
	 */
	public static double rms(double[] a) {
		double rms = 0.0;
		for (double v : a) {
			rms += v * v;
		}
		rms /= a.length;
		return Math.sqrt(rms);
	}

}
