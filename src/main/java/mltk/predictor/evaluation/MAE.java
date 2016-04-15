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
		double lae = 0;
		for (int i = 0; i < preds.length; i++) {
			lae += Math.abs(targets[i] - preds[i]);
		}
		lae /= preds.length;
		return lae;
	}

	@Override
	public double eval(double[] preds, Instances instances) {
		double lae = 0;
		for (int i = 0; i < preds.length; i++) {
			lae += Math.abs(instances.get(i).getTarget() - preds[i]);
		}
		lae /= preds.length;
		return lae;
	}

}
