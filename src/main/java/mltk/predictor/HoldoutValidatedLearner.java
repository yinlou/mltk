package mltk.predictor;

import mltk.core.Instances;
import mltk.predictor.evaluation.ConvergenceTester;
import mltk.predictor.evaluation.Metric;

/**
 * Class for holdout validated learners.
 * 
 * @author Yin Lou
 *
 */
public abstract class HoldoutValidatedLearner extends Learner {

	protected Instances validSet;
	protected Metric metric;
	protected ConvergenceTester ct;
	
	/**
	 * Constructor.
	 */
	public HoldoutValidatedLearner() {
		ct = new ConvergenceTester(-1, 0, 1.0);
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

	/**
	 * Sets the metric. 
	 * 
	 * @param metric the metric.
	 */
	public void setMetric(Metric metric) {
		this.metric = metric;
	}
	
	/**
	 * Returns the convergence tester.
	 * 
	 * @return the convergence tester.
	 */
	public ConvergenceTester getConvergenceTester() {
		return ct;
	}
	
	/**
	 * Sets the convergence tester.
	 * 
	 * @param ct the convergence tester to set.
	 */
	public void setConvergenceTester(ConvergenceTester ct) {
		this.ct = ct;
	}
	
}
