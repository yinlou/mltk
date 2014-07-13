package mltk.util.tuple;

/**
 * CLass for <double, double> pair.
 * 
 * @author Yin Lou
 * 
 */
public class DoublePair {

	public double v1;
	public double v2;

	/**
	 * Constructor.
	 * 
	 * @param v1 the 1st <code>double</code>.
	 * @param v2 the 2nd <code>double</code>.
	 */
	public DoublePair(double v1, double v2) {
		this.v1 = v1;
		this.v2 = v2;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		long temp;
		temp = Double.doubleToLongBits(v1);
		result = prime * result + (int) (temp ^ (temp >>> 32));
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
		DoublePair other = (DoublePair) obj;
		if (Double.doubleToLongBits(v1) != Double.doubleToLongBits(other.v1))
			return false;
		if (Double.doubleToLongBits(v2) != Double.doubleToLongBits(other.v2))
			return false;
		return true;
	}

}
