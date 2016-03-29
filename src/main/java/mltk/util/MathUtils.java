package mltk.util;

/**
 * Class for utility functions for math.
 * 
 * @author Yin Lou
 * 
 */
public class MathUtils {

	public static final double EPSILON = 1e-8;
	public static final double LOG2 = Math.log(2);
	
	/**
	 * Returns the value of a sigmoid function.
	 * 
	 * @param a the number.
	 * @return the value of a sigmoid function.
	 */
	public static double sigmoid(double a) {
		return 1 / (1 + Math.exp(-a));
	}

	/**
	 * Returns the sign of a number.
	 * 
	 * @param a the number.
	 * @return the sign of a number.
	 */
	public static int sign(double a) {
		if (a < 0) {
			return -1;
		} else if (a > 0) {
			return 1;
		} else {
			return 0;
		}
	}

	/**
	 * Returns the sign of a number.
	 * 
	 * @param a the number.
	 * @return the sign of a number.
	 */
	public static int sign(int a) {
		if (a < 0) {
			return -1;
		} else if (a > 0) {
			return 1;
		} else {
			return 0;
		}
	}

	/**
	 * Returns 1 if the input is true and 0 otherwise.
	 * 
	 * @param b the input.
	 * @return 1 if the input is true and 0 otherwise.
	 */
	public static int indicator(boolean b) {
		return b ? 1 : 0;
	}

	/**
	 * Returns <code>true</code> if the floating number is integer.
	 * 
	 * @param v the floating number.
	 * @return <code>true</code> if the floating number is integer.
	 */
	public static boolean isInteger(double v) {
		return (v % 1) == 0;
	}
	
	/**
	 * Returns <code>true</code> if the floating number is zero.
	 * 
	 * @param v the floating number.
	 * @return <code>true</code> if the floating number is zero.
	 */
	public static boolean isZero(double v) {
		return Math.abs(v) < EPSILON;
	}
	
	/**
	 * Returns <code>true</code> if the first value is better.
	 * 
	 * @param a the 1st value.
	 * @param b the 2nd value.
	 * @param isLargerBetter <code>true</code> if the first value is better.
	 * @return <code>true</code> if the first value is better.
	 */
	public static boolean isFirstBetter(double a, double b, boolean isLargerBetter) {
		if (isLargerBetter) {
			return a > b;
		} else {
			return a < b;
		}
	}

}
