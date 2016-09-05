package mltk.predictor.evaluation;

import mltk.core.Instances;
import mltk.util.OptimUtils;

/**
 * Class for evaluating log loss (cross entropy).
 * 
 * @author Yin Lou
 *
 */
public class LogLoss extends SimpleMetric {
	
	protected boolean isRawScore;

	/**
	 * Constructor.
	 */
	public LogLoss() {
		this(false);
	}
	
	public LogLoss(boolean isRawScore) {
		super(false);
		this.isRawScore = isRawScore;
	}

	@Override
	public double eval(double[] preds, double[] targets) {
		return OptimUtils.computeLogLoss(preds, targets, isRawScore);
	}

	@Override
	public double eval(double[] preds, Instances instances) {
		double logLoss = 0;
		for (int i = 0; i < preds.length; i++) {
			logLoss += OptimUtils.computeLogLoss(preds[i], instances.get(i).getTarget(), isRawScore);
		}
		return logLoss;
	}
	
	/**
	 * Returns <code>true</code> if raw score is expected as input.
	 * 
	 * @return <code>true</code> if raw score is expected as input.
	 */
	public boolean isRawScore() {
		return isRawScore;
	}

}
