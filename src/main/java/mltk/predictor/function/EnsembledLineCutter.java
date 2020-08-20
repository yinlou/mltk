package mltk.predictor.function;

import mltk.core.Attribute;
import mltk.core.Instances;
import mltk.predictor.BaggedEnsemble;
import mltk.predictor.Learner;

public abstract class EnsembledLineCutter extends Learner {
	
	protected int attIndex;

	protected int numIntervals;
	
	protected int baggingIters;
	
	protected boolean isClassification;

	@Override
	public BaggedEnsemble build(Instances instances) {
		return build(instances, attIndex, numIntervals);
	}
	
	/**
	 * Builds an 1D function ensemble.
	 * 
	 * @param instances the training set.
	 * @param attIndex the attribute index.
	 * @param numIntervals the number of intervals.
	 * @return an 1D function ensemble.
	 */
	public BaggedEnsemble build(Instances instances, int attIndex, int numIntervals) {
		Attribute attribute = instances.getAttributes().get(attIndex);
		return build(instances, attribute, numIntervals);
	}
	
	public abstract BaggedEnsemble build(Instances instances, Attribute attribute, int numIntervals);
	
	/**
	 * Returns the index in the attribute list of the training set.
	 * 
	 * @return the index in the attribute list of the training set.
	 */
	public int getAttributeIndex() {
		return attIndex;
	}
	
	/**
	 * Sets the index in the attribute list of the training set.
	 * 
	 * @param attIndex the attribute index.
	 */
	public void setAttributeIndex(int attIndex) {
		this.attIndex = attIndex;
	}
	
	/**
	 * Returns the number of bagging iterations.
	 * 
	 * @return the number of bagging iterations.
	 */
	public int getBaggingIters() {
		return baggingIters;
	}

	/**
	 * Sets the number of bagging iterations.
	 * 
	 * @param baggingIters the number of bagging iterations.
	 */
	public void setBaggingIters(int baggingIters) {
		this.baggingIters = baggingIters;
	}
	
	/**
	 * Returns {@code true} if it is a classification problem.
	 * 
	 * @return {@code true} if it is a classification problem.
	 */
	public boolean isClassification() {
		return isClassification;
	}
	
	/**
	 * Sets {@code true} if it is a classification problem.
	 * 
	 * @param isClassification {@code true} if it is a classification problem.
	 */
	public void setClassification(boolean isClassification) {
		this.isClassification = isClassification;
	}
	
	/**
	 * Returns the number of intervals.
	 * 
	 * @return the number of intervals.
	 */
	public int getNumIntervals() {
		return numIntervals;
	}

	/**
	 * Sets the number of intervals.
	 * 
	 * @param numIntervals the number of intervals.
	 */
	public void setNumIntervals(int numIntervals) {
		this.numIntervals = numIntervals;
	}

}
