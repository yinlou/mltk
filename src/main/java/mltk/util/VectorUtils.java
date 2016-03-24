package mltk.util;

import mltk.core.DenseVector;
import mltk.core.SparseVector;
import mltk.core.Vector;

/**
 * Class for utility functions for real vectors.
 * 
 * @author Yin Lou
 * 
 */
public class VectorUtils {

	/**
	 * Adds a constant to all elements in the array.
	 * 
	 * @param a the vector.
	 * @param v the constant.
	 */
	public static void add(double[] a, double v) {
		for (int i = 0; i < a.length; i++) {
			a[i] += v;
		}
	}

	/**
	 * Subtracts a constant from all elements in the array.
	 * 
	 * @param a the vector.
	 * @param v the constant.
	 */
	public static void subtract(double[] a, double v) {
		for (int i = 0; i < a.length; i++) {
			a[i] -= v;
		}
	}

	/**
	 * Multiplies a constant to all elements in the array.
	 * 
	 * @param a the vector.
	 * @param v the constant.
	 */
	public static void multiply(double[] a, double v) {
		for (int i = 0; i < a.length; i++) {
			a[i] *= v;
		}
	}

	/**
	 * Divides a constant to all elements in the array.
	 * 
	 * @param a the vector.
	 * @param v the constant.
	 */
	public static void divide(double[] a, double v) {
		for (int i = 0; i < a.length; i++) {
			a[i] /= v;
		}
	}

	/**
	 * Returns the L2 norm of a vector.
	 * 
	 * @param a the vector.
	 * @return the L2 norm of a vector.
	 */
	public static double l2norm(double[] a) {
		return Math.sqrt(StatUtils.sumSq(a));
	}

	/**
	 * Returns the L2 norm of a vector.
	 * 
	 * @param v the vector.
	 * @return the L2 norm of a vector.
	 */
	public static double l2norm(Vector v) {
		return l2norm(v.getValues());
	}

	/**
	 * Returns the L1 norm of a vector.
	 * 
	 * @param a the vector.
	 * @return the L1 norm of a vector.
	 */
	public static double l1norm(double[] a) {
		double norm = 0;
		for (double v : a) {
			norm += Math.abs(v);
		}
		return norm;
	}

	/**
	 * Returns the L1 norm of a vector.
	 * 
	 * @param v the vector.
	 * @return the L1 norm of a vector.
	 */
	public static double l1norm(Vector v) {
		return l1norm(v.getValues());
	}

	/**
	 * Returns the dot product of two vectors.
	 * 
	 * @param a the 1st vector.
	 * @param b the 2nd vector.
	 * @return the dot product of two vectors.
	 */
	public static double dotProduct(double[] a, double[] b) {
		double s = 0;
		for (int i = 0; i < a.length; i++) {
			s += a[i] * b[i];
		}
		return s;
	}

	/**
	 * Returns the dot product of two vectors.
	 * 
	 * @param a the 1st vector.
	 * @param b the 2nd vector.
	 * @return the dot product of two vectors.
	 */
	public static double dotProduct(DenseVector a, DenseVector b) {
		return dotProduct(a.getValues(), b.getValues());
	}

	/**
	 * Returns the dot product of two vectors.
	 * 
	 * @param a the 1st vector.
	 * @param b the 2nd vector.
	 * @return the dot product of two vectors.
	 */
	public static double dotProduct(SparseVector a, DenseVector b) {
		int[] indices1 = a.getIndices();
		double[] values1 = a.getValues();
		double[] values2 = b.getValues();
		double s = 0;
		for (int i = 0; i < indices1.length; i++) {
			s += values1[i] * values2[indices1[i]];
		}
		return s;
	}

	/**
	 * Returns the dot product of two vectors.
	 * 
	 * @param a the 1st vector.
	 * @param b the 2nd vector.
	 * @return the dot product of two vectors.
	 */
	public static double dotProduct(DenseVector a, SparseVector b) {
		return dotProduct(b, a);
	}

	/**
	 * Returns the dot product of two vectors.
	 * 
	 * @param a the 1st vector.
	 * @param b the 2nd vector.
	 * @return the dot product of two vectors.
	 */
	public static double dotProduct(SparseVector a, SparseVector b) {
		int[] indices1 = a.getIndices();
		double[] values1 = a.getValues();
		int[] indices2 = b.getIndices();
		double[] values2 = b.getValues();
		double s = 0;
		int i = 0;
		int j = 0;
		while (i < indices1.length && j < indices2.length) {
			if (indices1[i] < indices2[j]) {
				i++;
			} else if (indices1[i] > indices2[j]) {
				j++;
			} else {
				s += values1[i] * values2[j];
				i++;
				j++;
			}
		}
		return s;
	}
	
	/**
	 * Returns the Pearson correlation coefficient between two vectors.
	 * 
	 * @param a the 1st vector.
	 * @param b the 2nd vector.
	 * @return the Pearson correlation coefficient between two vectors.
	 */
	public static double correlation(double[] a, double[] b) {
		double mean1 = StatUtils.mean(a);
		double mean2 = StatUtils.mean(b);
		double x = 0;
		double s1 = 0;
		double s2 = 0;
		for (int i = 0; i < a.length; i++) {
			double d1 = (a[i] - mean1);
			double d2 = (b[i] - mean2);
			x += d1 * d2;
			s1 += d1 * d1;
			s2 += d2 * d2;
		}
		return x / Math.sqrt(s1 * s2);
	}

}
