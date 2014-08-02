package mltk.predictor.glm;

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

	static double computeRidgeLoss(double[] residual, double[] w, double lambda) {
		double loss =  OptimUtils.computeQuadraticLoss(residual);
		loss += lambda / 2 * StatUtils.sumSq(w);
		return loss;
	}

	static double computeRidgeLoss(double[] pred, int[] y, double[] w, double lambda) {
		double loss = OptimUtils.computeLogisticLoss(pred, y);
		loss += lambda / 2 * StatUtils.sumSq(w);
		return loss;
	}

	static double computeLassoLoss(double[] residual, double[] w, double lambda) {
		double loss =  OptimUtils.computeQuadraticLoss(residual);
		loss += lambda * VectorUtils.l1norm(w);
		return loss;
	}

	static double computeLassoLoss(double[] pred, int[] y, double[] w, double lambda) {
		double loss =  OptimUtils.computeLogisticLoss(pred, y);
		loss += lambda * VectorUtils.l1norm(w);
		return loss;
	}

	static double computeElasticNetLoss(double[] residual, double[] w, double lambda1, double lambda2) {
		double loss =  OptimUtils.computeQuadraticLoss(residual);
		loss += lambda1 * VectorUtils.l1norm(w) + lambda2 / 2 * StatUtils.sumSq(w);
		return loss;
	}

	static double computeElasticNetLoss(double[] pred, int[] y, double[] w, double lambda1, double lambda2) {
		double loss =  OptimUtils.computeLogisticLoss(pred, y);
		loss += lambda1 * VectorUtils.l1norm(w) + lambda2 / 2 * StatUtils.sumSq(w);
		return loss;
	}

}
