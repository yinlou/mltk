package mltk.predictor.evaluation;

import mltk.core.Instances;
import mltk.util.OptimUtils;

/**
 * Class for evaluating logistic loss.
 * 
 * @author Yin Lou
 *
 */
public class LogisticLoss extends SimpleMetric {

	/**
	 * Constructor.
	 */
	public LogisticLoss() {
		super(false);
	}

	@Override
	public double eval(double[] preds, double[] targets) {
		return OptimUtils.computeLogisticLoss(preds, targets);
	}

	@Override
	public double eval(double[] preds, Instances instances) {
		double logisticLoss = 0;
		for (int i = 0; i < preds.length; i++) {
			logisticLoss += OptimUtils.computeLogisticLoss(preds[i], instances.get(i).getTarget());
		}
		logisticLoss /= preds.length;
		return logisticLoss;
	}

}
