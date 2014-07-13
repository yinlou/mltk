package mltk.util.tuple;

/**
 * Class for <int, int, int> triples.
 * 
 * @author Yin Lou
 * 
 */
public class IntTriple {

	public int v1;
	public int v2;
	public int v3;

	/**
	 * Constructor.
	 * 
	 * @param v1 the 1st <code>int</code>.
	 * @param v2 the 2nd <code>int</code>.
	 * @param v3 the 3rd <code>int</code>.
	 */
	public IntTriple(int v1, int v2, int v3) {
		this.v1 = v1;
		this.v2 = v2;
		this.v3 = v3;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + v1;
		result = prime * result + v2;
		result = prime * result + v3;
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
		IntTriple other = (IntTriple) obj;
		if (v1 != other.v1)
			return false;
		if (v2 != other.v2)
			return false;
		if (v3 != other.v3)
			return false;
		return true;
	}

}
