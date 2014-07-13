package mltk.predictor.glm;

import mltk.util.MathUtils;
import mltk.util.OptimUtils;
import mltk.util.StatUtils;
import mltk.util.VectorUtils;

class GLMOptimUtils {

	static GLM getGLM(int[] attrs, double[] w, double intercept) {
		GLM glm = attrs.length == 0 ? new GLM(0) : new GLM(attrs[attrs.length - 1] + 1);
		for (int i = 0; i < attrs.length; i++) {
			glm.w[0][attrs[i]] = w[i];
		}
		glm.intercept[0] = intercept;
		return glm;
	}

	static double fitIntercept(double[] residualTrain) {
		double delta = StatUtils.mean(residualTrain);
		VectorUtils.subtract(residualTrain, delta);
		return delta;
	}

	static double fitIntercept(double[] predictionTrain, int[] y) {
		double delta = 0;
		// Use Newton-Raphson's method to approximate
		// 1st derivative
		double eta = 0;
		// 2nd derivative
		double theta = 0;
		for (int i = 0; i < predictionTrain.length; i++) {
			double r = OptimUtils.getPseudoResidual(predictionTrain[i], y[i]);
			double t = Math.abs(r);
			eta += r;
			theta += t * (1 - t);
		}

		if (Math.abs(theta) > MathUtils.EPSILON) {
			delta = eta / theta;
			// Update predictions
			VectorUtils.add(predictionTrain, delta);
		}
		return delta;
	}

	static double computeLogisticLoss(double[] pred, int[] y) {
		double loss = 0;
		for (int i = 0; i < pred.length; i++) {
			loss += Math.log(1 + Math.exp(-(y[i] == 1 ? 1 : -1) * pred[i]));
		}
		return loss / y.length;
	}

	static double computeQuadraticLoss(double[] residual) {
		return StatUtils.sumSq(residual) / (2 * residual.length);
	}

	static double computeRidgeLoss(double[] residual, double[] w, double lambda) {
		double loss = computeQuadraticLoss(residual);
		loss += lambda / 2 * StatUtils.sumSq(w);
		return loss;
	}

	static double computeRidgeLoss(double[] pred, int[] y, double[] w, double lambda) {
		double loss = computeLogisticLoss(pred, y);
		loss += lambda / 2 * StatUtils.sumSq(w);
		return loss;
	}

	static double computeLassoLoss(double[] residual, double[] w, double lambda) {
		double loss = computeQuadraticLoss(residual);
		loss += lambda * VectorUtils.l1norm(w);
		return loss;
	}

	static double computeLassoLoss(double[] pred, int[] y, double[] w, double lambda) {
		double loss = computeLogisticLoss(pred, y);
		loss += lambda * VectorUtils.l1norm(w);
		return loss;
	}

	static double computeElasticNetLoss(double[] residual, double[] w, double lambda1, double lambda2) {
		double loss = computeQuadraticLoss(residual);
		loss += lambda1 * VectorUtils.l1norm(w) + lambda2 / 2 * StatUtils.sumSq(w);
		return loss;
	}

	static double computeElasticNetLoss(double[] pred, int[] y, double[] w, double lambda1, double lambda2) {
		double loss = computeLogisticLoss(pred, y);
		loss += lambda1 * VectorUtils.l1norm(w) + lambda2 / 2 * StatUtils.sumSq(w);
		return loss;
	}

	static double evalError(int[] y, double[] prediction) {
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
