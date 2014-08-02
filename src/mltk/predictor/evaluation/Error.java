package mltk.predictor.evaluation;

import mltk.core.Instances;

/**
 * Class for evaluating error rate.
 * 
 * @author Yin Lou
 *
 */
public class Error extends Metric {

	/**
	 * Constructor.
	 */
	public Error() {
		super(false);
	}

	@Override
	public double eval(double[] preds, double[] targets) {
		double error = 0;
		for (int i = 0; i < preds.length; i++) {
			if (preds[i] != targets[i]) {
				error++;
			}
		}
		return error / preds.length;
	}

	@Override
	public double eval(double[] preds, Instances instances) {
		double error = 0;
		for (int i = 0; i < preds.length; i++) {
			if (preds[i] != instances.get(i).getTarget()) {
				error++;
			}
		}
		return error / preds.length;
	}

}
