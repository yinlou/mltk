package mltk.util;

/**
 * Class for utility functions for math.
 * 
 * @author Yin Lou
 * 
 */
public class MathUtils {

	public static final double EPSILON = 1e-8;

	/**
	 * Returns the sign of a number.
	 * 
	 * @param a the number
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
	 * @param a the number
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
	 * @param v the floating number
	 * @return <code>true</code> if the floating number is integer.
	 */
	public static boolean isInteger(double v) {
		return (v % 1) == 0;
	}

}
