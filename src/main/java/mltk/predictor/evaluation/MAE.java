package mltk.predictor.evaluation;

import mltk.core.Instances;

/**
 * Class for evaluating mean absolute error (MAE).
 * 
 * @author Yin Lou
 *
 */
public class MAE extends SimpleMetric {

	/**
	 * Constructor.
	 */
	public MAE() {
		super(false);
	}

	@Override
	public double eval(double[] preds, double[] targets) {
		double mae = 0;
		for (int i = 0; i < preds.length; i++) {
			mae += Math.abs(targets[i] - preds[i]);
		}
		mae /= preds.length;
		return mae;
	}

	@Override
	public double eval(double[] preds, Instances instances) {
		double mae = 0;
		for (int i = 0; i < preds.length; i++) {
			mae += Math.abs(instances.get(i).getTarget() - preds[i]);
		}
		mae /= preds.length;
		return mae;
	}

}
