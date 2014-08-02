package mltk.predictor.evaluation;

import mltk.core.Instances;

/**
 * Class for evaluating root mean squared error (RMSE).
 * 
 * @author Yin Lou
 *
 */
public class RMSE extends Metric {

	/**
	 * Constructor.
	 */
	public RMSE() {
		super(false);
	}

	@Override
	public double eval(double[] preds, double[] targets) {
		double rmse = 0;
		for (int i = 0; i < preds.length; i++) {
			double d = targets[i] - preds[i];
			rmse += d * d;
		}
		rmse = Math.sqrt(rmse / preds.length);
		return rmse;
	}

	@Override
	public double eval(double[] preds, Instances instances) {
		double rmse = 0;
		for (int i = 0; i < preds.length; i++) {
			double d = instances.get(i).getTarget() - preds[i];
			rmse += d * d;
		}
		rmse = Math.sqrt(rmse / preds.length);
		return rmse;
	}

}
