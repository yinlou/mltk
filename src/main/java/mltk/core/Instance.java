package mltk.core;

/**
 * Class for instances.
 * 
 * @author Yin Lou
 * 
 */
public class Instance implements Copyable<Instance> {

	protected Vector vector;
	protected double[] target;
	protected double weight;

	/**
	 * Constructs a dense instance from values, target and weight.
	 * 
	 * @param values the values.
	 * @param target the target.
	 * @param weight the weight.
	 */
	public Instance(double[] values, double target, double weight) {
		this.vector = new DenseVector(values);
		this.target = new double[] { target };
		this.weight = weight;
	}

	/**
	 * Constructs a sparse instance from indices, values, target and weight.
	 * 
	 * @param indices the indices.
	 * @param values the values.
	 * @param target the target.
	 * @param weight the weight.
	 */
	public Instance(int[] indices, double[] values, double target, double weight) {
		this.vector = new SparseVector(indices, values);
		this.target = new double[] { target };
		this.weight = weight;
	}

	/**
	 * Constructs a dense instance from vector, target and weight.
	 * 
	 * @param vector the vector.
	 * @param target the target.
	 * @param weight the weight.
	 */
	public Instance(Vector vector, double target, double weight) {
		this.vector = vector;
		this.target = new double[] { target };
		this.weight = weight;
	}

	/**
	 * Constructor with default weight 1.0.
	 * 
	 * @param values the values.
	 * @param target the target.
	 */
	public Instance(double[] values, double target) {
		this(values, target, 1.0);
	}

	/**
	 * Constructor with default weight 1.0.
	 * 
	 * @param indices the indices.
	 * @param values the values.
	 * @param target the target.
	 */
	public Instance(int[] indices, double[] values, double target) {
		this(indices, values, target, 1.0);
	}

	/**
	 * Construct with default weight 1.0.
	 * 
	 * @param vector the vector.
	 * @param target the target.
	 */
	public Instance(Vector vector, double target) {
		this(vector, target, 1.0);
	}

	/**
	 * Constructor with default weight 1.0 and no target.
	 * 
	 * @param values the values.
	 */
	public Instance(double[] values) {
		this(values, Double.NaN);
	}

	/**
	 * Constructor with default weight 1.0 and no target.
	 * 
	 * @param indices the indices.
	 * @param values the values.
	 */
	public Instance(int[] indices, double[] values) {
		this(indices, values, Double.NaN);
	}

	/**
	 * Constructor with default weight 1.0 and no target.
	 * 
	 * @param vector the vector.
	 * @param values the values.
	 */
	public Instance(Vector vector, double[] values) {
		this(vector, Double.NaN);
	}

	/**
	 * Copy constructor.
	 * 
	 * @param instance the other instance to copy.
	 */
	public Instance(Instance instance) {
		this.vector = instance.vector;
		this.weight = instance.weight;
		this.target = instance.target;
	}

	/**
	 * Returns <code>true</code> if the instance is sparse.
	 * 
	 * @return <code>true</code> if the instance is sparse.
	 */
	public boolean isSparse() {
		return vector.isSparse();
	}

	/**
	 * Returns the value at specified attribute.
	 * 
	 * @param attIndex the attribute index.
	 * @return the value at specified attribute.
	 */
	public final double getValue(int attIndex) {
		return vector.getValue(attIndex);
	}

	/**
	 * Returns the values.
	 * 
	 * @return the values.
	 */
	public final double[] getValues() {
		return vector.getValues();
	}

	/**
	 * Returns an array representation of values at specified attributes.
	 * 
	 * @param attributes the attributes.
	 * @return an array representation of values at specified attributes.
	 */
	public final double[] getValues(int... attributes) {
		return vector.getValues(attributes);
	}

	/**
	 * Sets the value at specified attribute.
	 * 
	 * @param attIndex the attribute index.
	 * @param value the new value to set.
	 */
	public final void setValue(int attIndex, double value) {
		vector.setValue(attIndex, value);
	}

	/**
	 * Sets the value at specified attribute.
	 * 
	 * @param attribute the attribute.
	 * @param value the new value to set.
	 */
	public final void setValue(Attribute attribute, double value) {
		setValue(attribute.getIndex(), value);
	}

	/**
	 * Sets the values at specified attributes.
	 * 
	 * @param attributes the attribute index array.
	 * @param v the value array.
	 */
	public final void setValue(int[] attributes, double[] v) {
		for (int i = 0; i < attributes.length; i++) {
			setValue(attributes[i], v[i]);
		}
	}

	@Override
	public Instance copy() {
		Vector copyVector = vector.copy();
		return new Instance(copyVector, target[0], weight);
	}

	/**
	 * Returns a shallow copy.
	 * 
	 * @return a shallow copy.
	 */
	public Instance clone() {
		return new Instance(this);
	}

	/**
	 * Returns <code>true</code> if a specific attribute value is missing.
	 * 
	 * @param attIndex the attribute index.
	 * @return <code>true</code> if a specific attribute value is missing.
	 */
	public boolean isMissing(int attIndex) {
		return Double.isNaN(getValue(attIndex));
	}

	/**
	 * Returns the value at specified attribute.
	 * 
	 * @param att the attribute object.
	 * @return the value at specified attribute.
	 */
	public double getValue(Attribute att) {
		return getValue(att.getIndex());
	}

	/**
	 * Returns the vector.
	 * 
	 * @return the vector.
	 */
	public Vector getVector() {
		return vector;
	}

	/**
	 * Returns the weight of this instance.
	 * 
	 * @return the weight of this instance.
	 */
	public double getWeight() {
		return weight;
	}

	/**
	 * Sets the weight of this instance.
	 * 
	 * @param weight the new weight of this instance.
	 */
	public void setWeight(double weight) {
		this.weight = weight;
	}

	/**
	 * Returns the target value.
	 * 
	 * @return the target value.
	 */
	public double getTarget() {
		return target[0];
	}

	/**
	 * Sets the class value.
	 * 
	 * @param target the new class value.
	 */
	public void setTarget(double target) {
		this.target[0] = target;
	}

	/**
	 * Returns the string representation of this instance.
	 */
	public String toString() {
		StringBuilder sb = new StringBuilder();
		if (isSparse()) {
			sb.append(getTarget());
			SparseVector sv = (SparseVector) vector;
			int[] indices = sv.getIndices();
			double[] values = sv.getValues();
			for (int i = 0; i < indices.length; i++) {
				sb.append(" ").append(indices[i]).append(":").append(values[i]);
			}
		} else {
			double[] values = getValues();
			sb.append(values[0]);
			for (int i = 1; i < values.length; i++) {
				sb.append("\t").append(values[i]);
			}
			if (!Double.isNaN(getTarget())) {
				sb.append("\t").append(getTarget());
			}
		}
		return sb.toString();
	}

}
