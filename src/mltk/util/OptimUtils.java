package mltk.util;

/**
 * Class for utility functions for optimization.
 * 
 * @author Yin Lou
 * 
 */
public class OptimUtils {
	
	/**
	 * Returns the probability of being in positive class.
	 * 
	 * @param pred the prediction.
	 * @return the probability of being in positive class.
	 */
	public static double getProbability(double pred) {
		return 1.0 / (1.0 + Math.exp(-pred));
	}

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
	public static double getPseudoResidual(double pred, double cls) {
		if (cls == 1) {
			return 1 / (1 + Math.exp(pred));
		} else {
			return -1 / (1 + Math.exp(-pred));
		}
	}
	
	/**
	 * Computes the pseudo residuals.
	 * 
	 * @param prediction the prediction array.
	 * @param y the class label array.
	 * @param residual the residual array.
	 */
	public static void computePseudoResidual(double[] prediction, double[] y, double[] residual) {
		for (int i = 0; i < residual.length; i++) {
			residual[i] = getPseudoResidual(prediction[i], y[i]);
		}
	}

	/**
	 * Computes the logistic loss for binary classification problems.
	 * 
	 * @param pred the prediction.
	 * @param cls the class label.
	 * @return the logistic loss for binary classification problems.
	 */
	public static double computeLogisticLoss(double pred, double cls) {
		if (cls == 1) {
			return Math.log(1 + Math.exp(-pred));
		} else {
			return Math.log(1 + Math.exp(pred));
		}
	}
	
	/**
	 * Computes the logistic loss for binary classification problems.
	 * 
	 * @param pred the prediction array.
	 * @param y the class label array.
	 * @return the logistic loss for binary classification problems.
	 */
	public static double computeLogisticLoss(double[] pred, double[] y) {
		double loss = 0;
		for (int i = 0; i < pred.length; i++) {
			loss += computeLogisticLoss(pred[i], y[i]);
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
	 * Returns gradient on the intercept in binary classification problems. Predictions and residuals will be updated accordingly.
	 * 
	 * @param prediction the prediction array.
	 * @param residual the residual array.
	 * @param y the class label array.
	 * @return the fitted intercept.
	 */
	public static double fitIntercept(double[] prediction, double[] residual, double[] y) {
		double delta = 0;
		// Use Newton-Raphson's method to approximate
		// 1st derivative
		double eta = 0;
		// 2nd derivative
		double theta = 0;
		for (int i = 0; i < prediction.length; i++) {
			double r = residual[i];
			double t = Math.abs(r);
			eta += r;
			theta += t * (1 - t);
		}

		if (Math.abs(theta) > MathUtils.EPSILON) {
			delta = eta / theta;
			// Update predictions
			VectorUtils.add(prediction, delta);
			computePseudoResidual(prediction, y, residual);
		}
		return delta;
	}
	
	/**
	 * Returns <code>true</code> if the relative improvement is less than a threshold.
	 * 
	 * @param prevLoss the previous loss.
	 * @param currLoss the current loss.
	 * @param epsilon the threshold.
	 * @return <code>true</code> if the relative improvement is less than a threshold.
	 */
	public static boolean isConverged(double prevLoss, double currLoss, double epsilon) {
		if (prevLoss < MathUtils.EPSILON) {
			return true;
		} else {
			return (prevLoss - currLoss) / prevLoss < epsilon;
		}
	}

}
