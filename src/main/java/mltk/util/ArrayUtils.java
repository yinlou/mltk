package mltk.util;

import java.util.Arrays;
import java.util.List;

/**
 * Class for utility functions for arrays.
 * 
 * @author Yin Lou
 * 
 */
public class ArrayUtils {
	
	/**
	 * Converts an integer list to int array.
	 * 
	 * @param list the list.
	 * @return an int array.
	 */
	public static int[] toIntArray(List<Integer> list) {
		int[] a = new int[list.size()];
		for (int i = 0; i < list.size(); i++) {
			a[i] = list.get(i);
		}
		return a;
	}
	
	/**
	 * Converts a double list to double array.
	 * 
	 * @param list the list.
	 * @return a double array.
	 */
	public static double[] toDoubleArray(List<Double> list) {
		double[] a = new double[list.size()];
		for (int i = 0; i < list.size(); i++) {
			a[i] = list.get(i);
		}
		return a;
	}
	
	/**
	 * Converts a double array to an int array.
	 * 
	 * @param a the double array.
	 * @return an int array.
	 */
	public static int[] toIntArray(double[] a) {
		int[] b = new int[a.length];
		for (int i = 0; i < a.length; i++) {
			b[i] = (int) a[i];
		}
		return b;
	}
	
	/**
	 * Returns a string representation of the contents of the specified sub-array.
	 * 
	 * @param a the array.
	 * @param start the starting index (inclusive).
	 * @param end the ending index (exclusive).
	 * @return Returns a string representation of the contents of the specified sub-array.
	 */
	public static String toString(double[] a, int start, int end) {
		StringBuilder sb = new StringBuilder();
		sb.append("[").append(a[start]);
		for (int i = start + 1; i < end; i++) {
			sb.append(", ").append(a[i]);
		}
		sb.append("]");
		return sb.toString();
	}

	/**
	 * Parses a double array from a string (default delimiter: ",").
	 * 
	 * @param str the string representation of a double array.
	 * @return a double array.
	 */
	public static double[] parseDoubleArray(String str) {
		return parseDoubleArray(str, ",");
	}

	/**
	 * Parses a double array from a string.
	 * 
	 * @param str the string representation of a double array.
	 * @param delimiter the delimiter.
	 * @return a double array.
	 */
	public static double[] parseDoubleArray(String str, String delimiter) {
		if (str == null || str.equalsIgnoreCase("null")) {
			return null;
		}
		String[] data = str.substring(1, str.length() - 1).split(delimiter);
		double[] a = new double[data.length];
		for (int i = 0; i < a.length; i++) {
			a[i] = Double.parseDouble(data[i].trim());
		}
		return a;
	}

	/**
	 * Parses an int array from a string (default delimiter: ",").
	 * 
	 * @param str the string representation of an int array.
	 * @return an int array.
	 */
	public static int[] parseIntArray(String str) {
		return parseIntArray(str, ",");
	}

	/**
	 * Parses an int array from a string.
	 * 
	 * @param str the string representation of an int array.
	 * @param delimiter the delimiter.
	 * @return an int array.
	 */
	public static int[] parseIntArray(String str, String delimiter) {
		if (str == null || str.equalsIgnoreCase("null")) {
			return null;
		}
		String[] data = str.substring(1, str.length() - 1).split(delimiter);
		int[] a = new int[data.length];
		for (int i = 0; i < a.length; i++) {
			a[i] = Integer.parseInt(data[i].trim());
		}
		return a;
	}
	
	/**
	 * Parses a long array from a string (default delimiter: ",").
	 * 
	 * @param str the string representation of a long array.
	 * @return an long array.
	 */
	public static long[] parseLongArray(String str) {
		return parseLongArray(str, ",");
	}
	
	/**
	 * Parses a long array from a string.
	 * 
	 * @param str the string representation of a long array.
	 * @param delimiter the delimiter.
	 * @return a long array.
	 */
	public static long[] parseLongArray(String str, String delimiter) {
		if (str == null || str.equalsIgnoreCase("null")) {
			return null;
		}
		String[] data = str.substring(1, str.length() - 1).split(delimiter);
		long[] a = new long[data.length];
		for (int i = 0; i < a.length; i++) {
			a[i] = Long.parseLong(data[i].trim());
		}
		return a;
	}

	/**
	 * Returns <code>true</code> if the specified range of an array is constant c.
	 * 
	 * @param a the array.
	 * @param begin the index of first element (inclusive).
	 * @param end the index of last element (exclusive).
	 * @param c the constant to test.
	 * @return <code>true</code> if the specified range of an array is constant c.
	 */
	public static boolean isConstant(double[] a, int begin, int end, double c) {
		for (int i = begin; i < end; i++) {
			if (!MathUtils.equals(a[i], c)) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Returns <code>true</code> if the specified range of an array is constant c.
	 * 
	 * @param a the array.
	 * @param begin the index of first element (inclusive).
	 * @param end the index of last element (exclusive).
	 * @param c the constant to test.
	 * @return <code>true</code> if the specified range of an array is constant c.
	 */
	public static boolean isConstant(int[] a, int begin, int end, int c) {
		for (int i = begin; i < end; i++) {
			if (a[i] != c) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Returns <code>true</code> if the specified range of an array is constant c.
	 * 
	 * @param a the array.
	 * @param begin the index of first element (inclusive).
	 * @param end the index of last element (exclusive).
	 * @param c the constant to test.
	 * @return <code>true</code> if the specified range of an array is constant c.
	 */
	public static boolean isConstant(byte[] a, int begin, int end, byte c) {
		for (int i = begin; i < end; i++) {
			if (a[i] != c) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Returns the median of an array.
	 * 
	 * @param a the array.
	 * @return the median of an array.
	 */
	public static double getMedian(double[] a) {
		if (a.length == 0) {
			return 0;
		}
		double[] ary = Arrays.copyOf(a, a.length);
		Arrays.sort(ary);
		int mid = ary.length / 2;
		if (ary.length % 2 == 1) {
			return ary[mid];
		} else {
			return (ary[mid - 1] + ary[mid]) / 2;
		}
	}

}
