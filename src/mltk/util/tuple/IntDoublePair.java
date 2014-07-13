package mltk.util.tuple;

/**
 * Class for <int, double> pairs.
 * 
 * @author Yin Lou
 * 
 */
public class IntDoublePair {

	public int v1;
	public double v2;

	/**
	 * Constructor.
	 * 
	 * @param v1 the <code>int</code> value.
	 * @param v2 the <code>double</code> value.
	 */
	public IntDoublePair(int v1, double v2) {
		this.v1 = v1;
		this.v2 = v2;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + v1;
		long temp;
		temp = Double.doubleToLongBits(v2);
		result = prime * result + (int) (temp ^ (temp >>> 32));
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
		IntDoublePair other = (IntDoublePair) obj;
		if (v1 != other.v1)
			return false;
		if (Double.doubleToLongBits(v2) != Double.doubleToLongBits(other.v2))
			return false;
		return true;
	}

}
