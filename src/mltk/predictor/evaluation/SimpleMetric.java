package mltk.predictor.evaluation;

/**
 * Class for simple metrics.
 * 
 * @author Yin Lou
 *
 */
public abstract class SimpleMetric extends Metric {

	/**
	 * Constructor.
	 * 
	 * @param isLargerBetter <code>true</code> if larger value is better.
	 */
	public SimpleMetric(boolean isLargerBetter) {
		super(isLargerBetter);
	}
	
	/**
	 * Evaluates predictions given targets.
	 * 
	 * @param preds the predictions.
	 * @param targets the targets.
	 * @return the evaluation measure.
	 */
	public abstract double eval(double[] preds, double[] targets);

}
