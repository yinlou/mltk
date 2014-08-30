package mltk.predictor.glm;

import mltk.util.OptimUtils;
import mltk.util.StatUtils;
import mltk.util.VectorUtils;

class GLMOptimUtils {

	static GLM getGLM(int[] attrs, double[] w, double intercept) {
		final int p = attrs.length == 0 ? 0 : (StatUtils.max(attrs) + 1);
		GLM glm = new GLM(p);
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

	static double computeRidgeLoss(double[] pred, double[] y, double[] w, double lambda) {
		double loss = OptimUtils.computeLogisticLoss(pred, y);
		loss += lambda / 2 * StatUtils.sumSq(w);
		return loss;
	}

	static double computeLassoLoss(double[] residual, double[] w, double lambda) {
		double loss =  OptimUtils.computeQuadraticLoss(residual);
		loss += lambda * VectorUtils.l1norm(w);
		return loss;
	}

	static double computeLassoLoss(double[] pred, double[] y, double[] w, double lambda) {
		double loss =  OptimUtils.computeLogisticLoss(pred, y);
		loss += lambda * VectorUtils.l1norm(w);
		return loss;
	}

	static double computeElasticNetLoss(double[] residual, double[] w, double lambda1, double lambda2) {
		double loss =  OptimUtils.computeQuadraticLoss(residual);
		loss += lambda1 * VectorUtils.l1norm(w) + lambda2 / 2 * StatUtils.sumSq(w);
		return loss;
	}

	static double computeElasticNetLoss(double[] pred, double[] y, double[] w, double lambda1, double lambda2) {
		double loss =  OptimUtils.computeLogisticLoss(pred, y);
		loss += lambda1 * VectorUtils.l1norm(w) + lambda2 / 2 * StatUtils.sumSq(w);
		return loss;
	}
	
	static double computeGroupLassoLoss(double[] residual, double[][] w, double[] tl1) {
		double loss =  OptimUtils.computeQuadraticLoss(residual);
		for (int k = 0; k < w.length; k++) {
			loss += tl1[k] * StatUtils.sumSq(w[k]);
		}
		return loss;
	}

	static double computeGroupLassoLoss(double[] pred, double[] y, double[][] w, double[] tl1) {
		double loss =  OptimUtils.computeLogisticLoss(pred, y);
		for (int k = 0; k < w.length; k++) {
			loss += tl1[k] * StatUtils.sumSq(w[k]);
		}
		return loss;
	}

}
