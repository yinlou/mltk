package mltk.core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import mltk.util.Random;

/**
 * Class for handling an ordered set of instances.
 * 
 * @author Yin Lou
 * 
 */
public class Instances implements Iterable<Instance>, Copyable<Instances> {

	protected List<Attribute> attributes;
	protected List<Instance> instances;
	protected Attribute targetAtt;

	/**
	 * Constructs a dataset from attributes.
	 * 
	 * @param attributes the attributes.
	 */
	public Instances(List<Attribute> attributes) {
		this(attributes, null);
	}

	/**
	 * Constructs a dataset from attributes, with specified capacity.
	 * 
	 * @param attributes the attributes.
	 * @param capacity the capacity.
	 */
	public Instances(List<Attribute> attributes, int capacity) {
		this(attributes, null, capacity);
	}

	/**
	 * Constructs a dataset from attributes and target attribute.
	 * 
	 * @param attributes the attributes.
	 * @param targetAtt the target attribute.
	 */
	public Instances(List<Attribute> attributes, Attribute targetAtt) {
		this(attributes, targetAtt, 1000);
	}

	/**
	 * Constructs a dataset from attributes and target attribute, with specified capacity.
	 * 
	 * @param attributes the attributes.
	 * @param targetAtt the target attribute.
	 * @param capacity the capacity.
	 */
	public Instances(List<Attribute> attributes, Attribute targetAtt, int capacity) {
		this.attributes = attributes;
		this.targetAtt = targetAtt;
		this.instances = new ArrayList<>(capacity);
	}

	/**
	 * Copy constructor.
	 * 
	 * @param instances the instances to copy.
	 */
	public Instances(Instances instances) {
		this.attributes = instances.attributes;
		this.targetAtt = instances.targetAtt;
		this.instances = new ArrayList<>(instances.instances);
	}

	/**
	 * Adds an instance to the end of the dataset.
	 * 
	 * @param instance the instance to add.
	 */
	public void add(Instance instance) {
		instances.add(instance);
	}

	/**
	 * Returns the instance at given index.
	 * 
	 * @param index the index.
	 * @return the instance at given index.
	 */
	public Instance get(int index) {
		return instances.get(index);
	}

	/**
	 * Returns the target attribute.
	 * 
	 * @return the target attribute.
	 */
	public final Attribute getTargetAttribute() {
		return targetAtt;
	}

	/**
	 * Sets the target attribute.
	 * 
	 * @param targetAtt the target attribute.
	 */
	public final void setTargetAttribute(Attribute targetAtt) {
		this.targetAtt = targetAtt;
	}

	@Override
	public Iterator<Instance> iterator() {
		return instances.iterator();
	}

	/**
	 * Returns the size of this dataset, i.e., the number of instances.
	 * 
	 * @return the size of this dataset.
	 */
	public final int size() {
		return instances.size();
	}

	/**
	 * Returns the dimension of this dataset, i.e., the number of attributes. Note that class attribute does not count.
	 * 
	 * @return the dimension of this dataset.
	 */
	public final int dimension() {
		return attributes.size();
	}

	/**
	 * Returns the list of attributes.
	 * 
	 * @return the list of attributes.
	 */
	public List<Attribute> getAttributes() {
		return attributes;
	}

	/**
	 * Returns the list of attributes at given locations.
	 * 
	 * @param indices the indices.
	 * @return the list of attributes at given locations.
	 */
	public List<Attribute> getAttributes(int... indices) {
		List<Attribute> attributes = new ArrayList<>(indices.length);
		for (int index : indices) {
			attributes.add(this.attributes.get(index));
		}
		return attributes;
	}

	/**
	 * Sets the attributes.
	 * 
	 * @param attributes the attributes to set.
	 */
	public void setAttributes(List<Attribute> attributes) {
		this.attributes = attributes;
	}

	/**
	 * Resets this dataset.
	 */
	public void clear() {
		instances.clear();
	}

	/**
	 * Randomly permutes this dataset.
	 */
	public void shuffle() {
		Collections.shuffle(instances, Random.getInstance().getRandom());
	}

	/**
	 * Randomly permutes this dataset.
	 * 
	 * @param rand the source of randomness to use to shuffle the dataset.
	 */
	public void shuffle(java.util.Random rand) {
		Collections.shuffle(instances, rand);
	}

	@Override
	public Instances copy() {
		List<Attribute> attributes = new ArrayList<>(this.attributes);
		Instances copy = new Instances(attributes, targetAtt, instances.size());
		for (Instance instance : instances) {
			copy.add(instance.copy());
		}
		return copy;
	}

}
