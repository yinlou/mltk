package mltk.predictor.tree.ensemble.brt;

import mltk.core.Instances;
import mltk.predictor.evaluation.Metric;
import mltk.predictor.tree.ensemble.TreeEnsembleLearner;

/**
 * Abstract class for boosted regression tree learner.
 * 
 * @author Yin Lou
 *
 */
public abstract class BRTLearner extends TreeEnsembleLearner {

	protected int maxNumIters;
	protected double alpha;
	protected double learningRate;
	protected Instances validSet;
	protected Metric metric;

	/**
	 * Sets the metric. 
	 * 
	 * @param metric the metric.
	 */
	public void setMetric(Metric metric) {
		this.metric = metric;
	}
	
	/**
	 * Returns the alpha.
	 * 
	 * @return the alpha.
	 */
	public double getAlpha() {
		return alpha;
	}
	
	/**
	 * Sets the alpha.
	 * 
	 * @param alpha the parameter that controls the portion of the features to consider
	 * for each boosting iteration.
	 */
	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}
	
	/**
	 * Returns the learning rate.
	 * 
	 * @return the learning rate.
	 */
	public double getLearningRate() {
		return learningRate;
	}

	/**
	 * Sets the learning rate.
	 * 
	 * @param learningRate the learning rate.
	 */
	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}
	
	/**
	 * Returns the maximum number of iterations.
	 * 
	 * @return the maximum number of iterations.
	 */
	public int getMaxNumIters() {
		return maxNumIters;
	}
	
	/**
	 * Sets the maximum number of iterations.
	 * 
	 * @param maxNumIters the maximum number of iterations.
	 */
	public void setMaxNumIters(int maxNumIters) {
		this.maxNumIters = maxNumIters;
	}
	
	/**
	 * Returns the validation set.
	 * 
	 * @return the validation set.
	 */
	public Instances getValidSet() {
		return validSet;
	}

	/**
	 * Sets the validation set.
	 * 
	 * @param validSet the validation set.
	 */
	public void setValidSet(Instances validSet) {
		this.validSet = validSet;
	}
	
	/**
	 * Returns the metric.
	 * 
	 * @return the metric.
	 */
	public Metric getMetric() {
		return metric;
	}
	
}
