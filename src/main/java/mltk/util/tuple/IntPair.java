package mltk.util.tuple;

/**
 * Class for <int, int> pairs.
 * 
 * @author Yin Lou
 * 
 */
public class IntPair {

	public int v1;
	public int v2;

	/**
	 * Constructor.
	 * 
	 * @param v1 the 1st <code>int</code>.
	 * @param v2 the 2nd <code>int</code>.
	 */
	public IntPair(int v1, int v2) {
		this.v1 = v1;
		this.v2 = v2;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + v1;
		result = prime * result + v2;
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		IntPair other = (IntPair) obj;
		if (v1 != other.v1)
			return false;
		if (v2 != other.v2)
			return false;
		return true;
	}

}
