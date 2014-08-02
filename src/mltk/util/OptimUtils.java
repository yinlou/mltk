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
	 * @param pred the prediction.
	 * @param target the target.
	 * @return the residual.
	 */
	public static double getResidual(double pred, double target) {
		return target - pred;
	}

	/**
	 * Returns the pseudo residual.
	 * 
	 * @param pred the prediction.
	 * @param cls the class label.
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
	 * Computes the logistic loss for binary classification problems.
	 * 
	 * @param pred the prediction array.
	 * @param y the target array.
	 * @return the logistic loss for binary classification problems.
	 */
	public static double computeLogisticLoss(double[] pred, int[] y) {
		double loss = 0;
		for (int i = 0; i < pred.length; i++) {
			loss += Math.log(1 + Math.exp(-(y[i] == 1 ? 1 : -1) * pred[i]));
		}
		return loss / y.length;
	}

	/**
	 * Computes the quadratic loss for regression problems.
	 * 
	 * @param residual the residual array.
	 * @return the quadratic loss for regression problems.
	 */
	public static double computeQuadraticLoss(double[] residual) {
		return StatUtils.sumSq(residual) / (2 * residual.length);
	}
	
	/**
	 * Returns gradient on the intercept in regression problems. Residuals will be updated accordingly.
	 * 
	 * @param residual the residual array.
	 * @return the fitted intercept.
	 */
	public static double fitIntercept(double[] residual) {
		double delta = StatUtils.mean(residual);
		VectorUtils.subtract(residual, delta);
		return delta;
	}

	/**
	 * Returns gradient on the intercept in binary classification problems. Predictions will be updated accordingly.
	 * 
	 * @param prediction the prediction array.
	 * @param y the target array.
	 * @return the fitted intercept.
	 */
	public static double fitIntercept(double[] prediction, int[] y) {
		double delta = 0;
		// Use Newton-Raphson's method to approximate
		// 1st derivative
		double eta = 0;
		// 2nd derivative
		double theta = 0;
		for (int i = 0; i < prediction.length; i++) {
			double r = getPseudoResidual(prediction[i], y[i]);
			double t = Math.abs(r);
			eta += r;
			theta += t * (1 - t);
		}

		if (Math.abs(theta) > MathUtils.EPSILON) {
			delta = eta / theta;
			// Update predictions
			VectorUtils.add(prediction, delta);
		}
		return delta;
	}
	
	/**
	 * Returns the error rate for binary classification problems.
	 * 
	 * @param prediction the prediction array.
	 * @param y the target array.
	 * @return the error rate for binary classification problems.
	 */
	public static double evalError(double[] prediction, int[] y) {
		double error = 0;
		for (int i = 0; i < prediction.length; i++) {
			int pred = prediction[i] >= 0 ? 1 : 0;
			if (pred != y[i]) {
				error++;
			}
		}
		error /= y.length;
		return error;
	}

}
