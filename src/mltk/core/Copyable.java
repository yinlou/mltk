package mltk.core;

/**
 * Copyable interface.
 * 
 * @author Yin Lou
 * 
 * @param <T> the type of the object.
 */
public interface Copyable<T> {

	/**
	 * Returns a (deep) copy of the object.
	 * 
	 * @return a (deep) copy of the object.
	 */
	public T copy();

}
