package mltk.util.tuple;

/**
 * Class for generic triples.
 * 
 * @author Yin Lou
 * 
 * @param <T1>
 * @param <T2>
 * @param <T3>
 */
public class Triple<T1, T2, T3> {

	public T1 v1;
	public T2 v2;
	public T3 v3;

	/**
	 * Constructor.
	 * 
	 * @param v1 the 1st element.
	 * @param v2 the 2nd element.
	 * @param v3 the 3rd element.
	 */
	public Triple(T1 v1, T2 v2, T3 v3) {
		this.v1 = v1;
		this.v2 = v2;
		this.v3 = v3;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((v1 == null) ? 0 : v1.hashCode());
		result = prime * result + ((v2 == null) ? 0 : v2.hashCode());
		result = prime * result + ((v3 == null) ? 0 : v3.hashCode());
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
		Triple<?, ?, ?> other = (Triple<?, ?, ?>) obj;
		if (v1 == null) {
			if (other.v1 != null)
				return false;
		} else if (!v1.equals(other.v1))
			return false;
		if (v2 == null) {
			if (other.v2 != null)
				return false;
		} else if (!v2.equals(other.v2))
			return false;
		if (v3 == null) {
			if (other.v3 != null)
				return false;
		} else if (!v3.equals(other.v3))
			return false;
		return true;
	}

}
