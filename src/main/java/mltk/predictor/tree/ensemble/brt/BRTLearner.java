package mltk.predictor.tree.ensemble.brt;

import mltk.predictor.tree.RegressionTreeLearner.Mode;
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
	
	/**
	 * Constructor.
	 */
	public BRTLearner() {
		verbose = false;
		maxNumIters = 3500;
		learningRate = 0.01;
		alpha = 1;
		
		RobustRegressionTreeLearner rtLearner = new RobustRegressionTreeLearner();
		rtLearner.setConstructionMode(Mode.NUM_LEAVES_LIMITED);
		rtLearner.setMaxNumLeaves(100);
		
		treeLearner = rtLearner;
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
	
}
