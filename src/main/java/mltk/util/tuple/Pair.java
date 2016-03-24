package mltk.util.tuple;

/**
 * Class for generic pairs.
 * 
 * @author Yin Lou
 * 
 * @param <T1>
 * @param <T2>
 */
public class Pair<T1, T2> {

	public T1 v1;
	public T2 v2;

	/**
	 * Constructor.
	 * 
	 * @param v1 the 1st element.
	 * @param v2 the 2nd element.
	 */
	public Pair(T1 v1, T2 v2) {
		this.v1 = v1;
		this.v2 = v2;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((v1 == null) ? 0 : v1.hashCode());
		result = prime * result + ((v2 == null) ? 0 : v2.hashCode());
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
		Pair<?, ?> other = (Pair<?, ?>) obj;
		if (v1 == null) {
			if (other.v1 != null)
				return false;
		} else if (v1.getClass() != other.v1.getClass()) {
			return false;
		} else if (!v1.equals(other.v1)) {
			return false;
		}
		if (v2 == null) {
			if (other.v2 != null)
				return false;
		} else if (v2.getClass() != other.v2.getClass()) {
			return false;
		} else if (!v2.equals(other.v2)) {
			return false;
		}
		return true;
	}

}
