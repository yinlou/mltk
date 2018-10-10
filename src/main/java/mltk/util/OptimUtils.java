package mltk.util;

import java.util.List;

/**
 * Class for utility functions for optimization.
 * 
 * @author Yin Lou
 * 
 */
public class OptimUtils {
	
	/**
	 * Returns the gain for variance reduction. This method is mostly used
	 * in tree learners.
	 * 
	 * @param sum the sum of responses.
	 * @param weight the total weight.
	 * @return the gain for variance reduction.
	 */
	public static double getGain(double sum, double weight) {
		if (weight < MathUtils.EPSILON) {
			return 0;
		} else {
			return sum / weight * sum;
		}
	}
	
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
		return cls - getProbability(pred);
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
	 * Computes the log loss (cross entropy) for binary classification problems.
	 * 
	 * @param prob the probability.
	 * @param y the class label.
	 * @return the log loss.
	 */
	public static double computeLogLoss(double prob, double y) {
		return computeLogLoss(prob, y, false);
	}
	
	/**
	 * Computes the log loss (cross entropy) for binary classification problems.
	 * 
	 * @param p the input.
	 * @param y the class label.
	 * @param isRawScore {@code true} if the input is raw score.
	 * @return the log loss.
	 */
	public static double computeLogLoss(double p, double y, boolean isRawScore) {
		if (isRawScore) {
			p = MathUtils.sigmoid(p);
		}
		if (y == 1) {
			return -Math.log(p);
		} else {
			return -Math.log(1 - p);
		}
	}
	
	/**
	 * Computes the log loss (cross entropy) for binary classification problems.
	 * 
	 * @param prob the probabilities.
	 * @param y the class label array.
	 * @return the log loss.
	 */
	public static double computeLogLoss(double[] prob, double[] y) {
		return computeLogLoss(prob, y, false);
	}
	
	/**
	 * Computes the log loss (cross entropy) for binary classification problems.
	 * 
	 * @param p the input.
	 * @param y the targets
	 * @param isRawScore {@code true} if the input is raw score.
	 * @return the log loss.
	 */
	public static double computeLogLoss(double[] p, double[] y, boolean isRawScore) {
		double logLoss = 0;
		for (int i = 0; i < p.length; i++) {
			logLoss += computeLogLoss(p[i], y[i], isRawScore);
		}
		return logLoss;
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
	 * Returns {@code true} if the relative improvement is less than a threshold.
	 * 
	 * @param prevLoss the previous loss.
	 * @param currLoss the current loss.
	 * @param epsilon the threshold.
	 * @return {@code true} if the relative improvement is less than a threshold.
	 */
	public static boolean isConverged(double prevLoss, double currLoss, double epsilon) {
		if (prevLoss < MathUtils.EPSILON) {
			return true;
		} else {
			return (prevLoss - currLoss) / prevLoss < epsilon;
		}
	}
	
	/**
	 * Returns {@code true} if the array of metric values is converged.
	 * 
	 * @param p an array of metric values.
	 * @param isLargerBetter {@code true} if larger value is better.
	 * @return {@code true} if the list of metric values is converged.
	 */
	public static boolean isConverged(double[] p, boolean isLargerBetter) {
		final int bn = p.length;
		if (p.length <= 20) {
			return false;
		}

		double bestPerf = p[bn - 1];
		double worstPerf = p[bn - 20];
		for (int i = bn - 20; i < bn; i++) {
			if (MathUtils.isFirstBetter(p[i], bestPerf, isLargerBetter)) {
				bestPerf = p[i];
			}
			if (!MathUtils.isFirstBetter(p[i], worstPerf, isLargerBetter)) {
				worstPerf = p[i];
			}
		}
		double relMaxMin = Math.abs(worstPerf - bestPerf) / worstPerf;
		double relImprov;
		if (MathUtils.isFirstBetter(p[bn - 1], p[bn - 21], isLargerBetter)) {
			relImprov = Math.abs(p[bn - 21] - p[bn - 1]) / p[bn - 21];
		} else {
			// Overfitting
			relImprov = Double.NaN;
		}
		return relMaxMin < 0.02 && (Double.isNaN(relImprov) || relImprov < 0.005);
	}
	
	/**
	 * Returns {@code true} if the list of metric values is converged.
	 * 
	 * @param list a list of metric values.
	 * @param isLargerBetter {@code true} if larger value is better.
	 * @return {@code true} if the list of metric values is converged.
	 */
	public static boolean isConverged(List<Double> list, boolean isLargerBetter) {
		if (list.size() <= 20) {
			return false;
		}

		final int bn = list.size();
		double bestPerf = list.get(bn - 1);
		double worstPerf = list.get(bn - 20);
		for (int i = bn - 20; i < bn; i++) {
			if (MathUtils.isFirstBetter(list.get(i), bestPerf, isLargerBetter)) {
				bestPerf = list.get(i);
			}
			if (!MathUtils.isFirstBetter(list.get(i), worstPerf, isLargerBetter)) {
				worstPerf = list.get(i);
			}
		}
		double relMaxMin = Math.abs(worstPerf - bestPerf) / worstPerf;
		double relImprov;
		if (MathUtils.isFirstBetter(list.get(bn - 1), list.get(bn - 21), isLargerBetter)) {
			relImprov = Math.abs(list.get(bn - 21) - list.get(bn - 1)) / list.get(bn - 21);
		} else {
			// Overfitting
			relImprov = Double.NaN;
		}
		return relMaxMin < 0.02 && (Double.isNaN(relImprov) || relImprov < 0.005);
	}

}
