package mltk.core;

/**
 * Interface for vectors.
 * 
 * @author Yin Lou
 * 
 */
public interface Vector extends Copyable<Vector> {

	/**
	 * Returns the value at specified index.
	 * 
	 * @param index the index.
	 * @return the value at specified index.
	 */
	public double getValue(int index);

	/**
	 * Returns the internal representation of values.
	 * 
	 * @return the internal representation of values.
	 */
	public double[] getValues();

	/**
	 * Returns an array representation of values at specified indices.
	 * 
	 * @param indices the indices.
	 * @return an array representation of values at specified indices.
	 */
	public double[] getValues(int... indices);

	/**
	 * Sets the value at specified index.
	 * 
	 * @param index the index.
	 * @param value the new value to set.
	 */
	public void setValue(int index, double value);

	/**
	 * Sets the values at specified indices.
	 * 
	 * @param indices the index array.
	 * @param v the value array.
	 */
	public void setValue(int[] indices, double[] v);

	/**
	 * Returns <code>true</code> if the vector is sparse.
	 * 
	 * @return <code>true</code> if the vector is sparse.
	 */
	public boolean isSparse();

	/**
	 * Returns a (deep) copy of the vector.
	 * 
	 * @return a (deep) copy of the vector.
	 */
	public Vector copy();

}
