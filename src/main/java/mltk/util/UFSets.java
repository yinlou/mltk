package mltk.util;

/**
 * Class for union-find sets.
 * 
 * @author Yin Lou
 *
 */
public class UFSets {
	
	private int[] parent;

	/**
	 * Constructor.
	 * 
	 * @param size the size.
	 */
	public UFSets(int size) {
		parent = new int[size + 1];
		for (int i = 0; i < parent.length; i++) {
			parent[i] = -1;
		}
	}
	
	/**
	 * Unions two sets.
	 * 
	 * @param root1 the root for the 1st set.
	 * @param root2 the root for the 2nd set.
	 */
	public void union(int root1, int root2) {
		int temp = parent[root1] + parent[root2];
		if (parent[root1] < parent[root2]) {
			parent[root2] = root1;
			parent[root1] = temp;
		} else {
			parent[root1] = root2;
			parent[root2] = temp;
		}
	}
	
	/**
	 * Returns the root of the set that contains the search key.
	 * 
	 * @param i the search key.
	 * @return the root of the set that contains the search key.
	 */
	public int find(int i) {
		int j;
		for (j = i; parent[j] >= 0; j = parent[j]);
		while (i != j) {
			int temp = parent[i];
			parent[i] = j;
			i = temp;
		}
		return j;
	}
	
}
