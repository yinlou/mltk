package mltk.util;

/**
 * Class for utility functions for optimization.
 * 
 * @author Yin Lou
 * 
 */
public class OptimUtils {

	/**
	 * Returns the residual.
	 * 
	 * @param pred
	 *            the prediction.
	 * @param target
	 *            the target.
	 * @return the residual.
	 */
	public static double getResidual(double pred, double target) {
		return target - pred;
	}

	/**
	 * Returns the pseudo residual.
	 * 
	 * @param pred
	 *            the prediction.
	 * @param cls
	 *            the class label.
	 * @return the pseudo residual.
	 */
	public static double getPseudoResidual(double pred, int cls) {
		if (cls == 1) {
			return 1 / (1 + Math.exp(pred));
		} else {
			return -1 / (1 + Math.exp(-pred));
		}
	}

	/**
	 * Returns the logistic loss.
	 * 
	 * @param cls
	 *            the class label.
	 * @param pred
	 *            the prediction.
	 * @return the logistic loss.
	 */
	public static double getLogisticLoss(int cls, double pred) {
		if (cls == 1) {
			return Math.log(1 + Math.exp(-pred));
		} else {
			return Math.log(1 + Math.exp(pred));
		}
	}

}
