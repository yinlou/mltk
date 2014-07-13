package mltk.util.tuple;

/**
 * Class for <long, long> pairs.
 * 
 * @author Yin Lou
 * 
 */
public class LongPair {

	public long v1;
	public long v2;

	/**
	 * Constructor.
	 * 
	 * @param v1 the 1st <code>long</code>.
	 * @param v2 the 2nd <code>long</code>.
	 */
	public LongPair(long v1, long v2) {
		this.v1 = v1;
		this.v2 = v2;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + (int) (v1 ^ (v1 >>> 32));
		result = prime * result + (int) (v2 ^ (v2 >>> 32));
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
		LongPair other = (LongPair) obj;
		if (v1 != other.v1)
			return false;
		if (v2 != other.v2)
			return false;
		return true;
	}

}
