package mltk.predictor.glm;

import mltk.core.Instances;
import mltk.predictor.Family;
import mltk.predictor.Learner;
import mltk.util.MathUtils;

/**
 * Abstract class for learning generalized linear models (GLMs).
 * 
 * @author Yin Lou
 *
 */
public abstract class GLMLearner extends Learner {
	
	protected boolean fitIntercept;
	protected int maxNumIters;
	protected double epsilon;
	protected Family family;
	
	/**
	 * Constructor.
	 */
	public GLMLearner() {
		verbose = false;
		fitIntercept = true;
		maxNumIters = -1;
		epsilon = MathUtils.EPSILON;
		family = Family.GAUSSIAN;
	}
	
	/**
	 * Returns <code>true</code> if we fit intercept.
	 * 
	 * @return <code>true</code> if we fit intercept.
	 */
	public boolean fitIntercept() {
		return fitIntercept;
	}

	/**
	 * Sets whether we fit intercept.
	 * 
	 * @param fitIntercept whether we fit intercept.
	 */
	public void fitIntercept(boolean fitIntercept) {
		this.fitIntercept = fitIntercept;
	}

	/**
	 * Returns the convergence threshold epsilon.
	 * 
	 * @return the convergence threshold epsilon.
	 */
	public double getEpsilon() {
		return epsilon;
	}
	
	/**
	 * Sets the convergence threshold epsilon.
	 * 
	 * @param epsilon the convergence threshold epsilon.
	 */
	public void setEpsilon(double epsilon) {
		this.epsilon = epsilon;
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
	 * Returns the response distribution family.
	 */
	public Family getFamily() {
		return family;
	}
	
	/**
	 * Sets the response distribution family.
	 * 
	 * @param family the response distribution family.
	 */
	public void setFamily(Family family) {
		this.family = family;
	}
	
	/**
	 * Builds a generalized linear model given response distribution family.
	 * The default link function for the family will be used.
	 * 
	 * @param trainSet the training set.
	 * @param family the response distribution family.
	 * @return a generalized linear model.
	 */
	public abstract GLM build(Instances trainSet, Family family);
	
}
