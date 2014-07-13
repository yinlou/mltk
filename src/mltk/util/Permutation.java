package mltk.util;

/**
 * Class for handling permutation.
 * 
 * @author Yin Lou
 * 
 */
public class Permutation {

	protected int[] a;

	/**
	 * Initializes a permutation of length n.
	 * 
	 * @param n the length of a permutation.
	 */
	public Permutation(int n) {
		a = new int[n];
		for (int i = 0; i < a.length; i++) {
			a[i] = i;
		}
	}

	/**
	 * Randomly permutes this permutation.
	 * 
	 * @return this permutation.
	 */
	public Permutation permute() {
		for (int i = 0; i < a.length; i++) {
			int idx = Random.getInstance().nextInt(i + 1);
			int t = a[idx];
			a[idx] = a[i];
			a[i] = t;
		}
		return this;
	}

	/**
	 * Returns the size of this permutation.
	 * 
	 * @return the size of this permutation.
	 */
	public int size() {
		return a.length;
	}

	/**
	 * Returns the permutation.
	 * 
	 * @return the permutation.
	 */
	public int[] getPermutation() {
		return a;
	}

}
