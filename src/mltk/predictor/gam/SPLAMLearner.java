package mltk.predictor.gam;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.cmdline.options.LearnerWithTaskOptions;
import mltk.core.Instances;
import mltk.core.io.InstancesReader;
import mltk.predictor.Learner;
import mltk.predictor.Regressor;
import mltk.predictor.function.CubicSpline;
import mltk.predictor.function.LinearFunction;
import mltk.predictor.glm.GLM;
import mltk.predictor.glm.RidgeLearner;
import mltk.predictor.io.PredictorWriter;
import mltk.util.ArrayUtils;
import mltk.util.MathUtils;
import mltk.util.OptimUtils;
import mltk.util.StatUtils;
import mltk.util.VectorUtils;

/**
 * Class for learning SPLAM models. Currently only cubic spline basis is supported.
 * 
 * @author Yin Lou
 * 
 */
public class SPLAMLearner extends Learner {
	
	static class Options extends LearnerWithTaskOptions {

		@Argument(name = "-d", description = "number of knots (default: 10)")
		int numKnots = 10;

		@Argument(name = "-m", description = "maximum number of iterations (default: 0)")
		int maxNumIters = 0;
		
		@Argument(name = "-l", description = "lambda (default: 0)")
		double lambda = 0;
		
		@Argument(name = "-a", description = "alpha (default: 1, i.e., SPAM model)")
		double alpha = 1;

	}
	
	/**
	 * <p>
	 * 
	 * <pre>
	 * Usage: mltk.predictor.gam.SPLAMLearner
	 * -t	train set path
	 * [-g]	task between classification (c) and regression (r) (default: r)
	 * [-r]	attribute file path
	 * [-o]	output model path
	 * [-V]	verbose (default: true)
	 * [-d]	number of knots (default: 10)
	 * [-m]	maximum number of iterations (default: 0)
	 * [-l]	lambda (default: 0)
	 * [-a]	alpha (default: 1, i.e., SPAM model)
	 * </pre>
	 * 
	 * </p>
	 * 
	 * @param args the command line arguments.
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(SPLAMLearner.class, opts);
		Task task = null;
		try {
			parser.parse(args);
			task = Task.getEnum(opts.task);
			if (opts.numKnots < 0) {
				throw new IllegalArgumentException("Number of knots must be positive.");
			}
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}

		Instances trainSet = InstancesReader.read(opts.attPath, opts.trainPath);

		SPLAMLearner learner = new SPLAMLearner();
		learner.setNumKnots(opts.numKnots);
		learner.setMaxNumIters(opts.maxNumIters);
		learner.setLambda(opts.lambda);
		learner.setAlpha(opts.alpha);
		learner.setTask(task);
		learner.setVerbose(opts.verbose);

		long start = System.currentTimeMillis();
		GAM gam = learner.build(trainSet);
		long end = System.currentTimeMillis();
		System.out.println("Time: " + (end - start) / 1000.0);
		
		if (opts.outputModelPath != null) {
			PredictorWriter.write(gam, opts.outputModelPath);
		}
	}

	static class ModelStructure {

		static final byte ELIMINATED = 0;
		static final byte LINEAR = 1;
		static final byte NONLINEAR = 2;

		byte[] structure;

		ModelStructure(byte[] structure) {
			this.structure = structure;
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			ModelStructure other = (ModelStructure) obj;
			if (!Arrays.equals(structure, other.structure))
				return false;
			return true;
		}

		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + Arrays.hashCode(structure);
			return result;
		}

	}
	
	private boolean fitIntercept;
	private boolean refit;
	private int numKnots;
	private int maxNumIters;
	private Task task;

	private double lambda;

	private double alpha;

	private double epsilon;

	/**
	 * Constructor.
	 */
	public SPLAMLearner() {
		verbose = false;
		fitIntercept = true;
		refit = false;
		numKnots = 10;
		maxNumIters = -1;
		lambda = 0.0;
		alpha = 1;
		epsilon = MathUtils.EPSILON;
		task = Task.REGRESSION;
	}

	@Override
	public GAM build(Instances instances) {
		GAM gam = null;
		if (maxNumIters < 0) {
			maxNumIters = 20;
		}
		if (numKnots < 0) {
			numKnots = 10;
		}
		switch (task) {
			case REGRESSION:
				gam = buildRegressor(instances, maxNumIters, numKnots, lambda, alpha);
				break;
			case CLASSIFICATION:
				gam = buildClassifier(instances, maxNumIters, numKnots, lambda, alpha);
				break;
			default:
				break;
		}
		return gam;
	}

	/**
	 * Returns a binary classifier.
	 * 
	 * @param attrs the attribute list.
	 * @param x the inputs.
	 * @param y the targets.
	 * @param knots the knots.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @param alpha the alpha.
	 * @return a binary classifier.
	 */
	public GAM buildBinaryClassifier(int[] attrs, double[][][] x, double[] y, double[][] knots, int maxNumIters, double lambda,
			double alpha) {
		double[][] w = new double[attrs.length][];
		int m = 0;
		for (int j = 0; j < attrs.length; j++) {
			w[j] = new double[x[j].length];
			if (w[j].length > m) {
				m = w[j].length;
			}
		}
		double[] tl1 = new double[attrs.length];
		double[] tl2 = new double[attrs.length];
		getRegularizationParameters(lambda, alpha, tl1, tl2, y.length);

		double[] pTrain = new double[y.length];
		double[] rTrain = new double[y.length];
		OptimUtils.computePseudoResidual(pTrain, y, rTrain);

		double[] stepSize = new double[attrs.length];
		for (int j = 0; j < stepSize.length; j++) {
			double max = 0;
			double[][] block = x[j];
			for (double[] t : block) {
				double l = StatUtils.sumSq(t) / 4;
				if (l > max) {
					max = l;
				}
			}
			stepSize[j] = 1.0 / max;
		}
		double[] g = new double[m];
		double[] gradient = new double[m];
		double[] gamma1 = new double[m];
		double[] gamma2 = new double[m - 1];

		boolean[] activeSet = new boolean[attrs.length];

		double intercept = 0;
		
		// Block coordinate gradient descent
		int iter = 0;
		while (iter < maxNumIters) {

			if (fitIntercept) {
				intercept += OptimUtils.fitIntercept(pTrain, rTrain, y);
			}

			boolean activeSetChanged = doOnePass(x, y, tl1, tl2, true, activeSet, w, stepSize, 
					g, gradient, gamma1, gamma2, pTrain, rTrain);

			iter++;

			if (!activeSetChanged || iter > maxNumIters) {
				break;
			}

			for (; iter < maxNumIters; iter++) {

				double prevLoss = OptimUtils.computeLogisticLoss(pTrain, y) + getPenalty(w, tl1, tl2);
				
				if (fitIntercept) {
					intercept += OptimUtils.fitIntercept(pTrain, rTrain, y);
				}

				doOnePass(x, y, tl1, tl2, false, activeSet, w, stepSize, g, gradient, gamma1, gamma2, pTrain, rTrain);

				double currLoss = OptimUtils.computeLogisticLoss(pTrain, y) + getPenalty(w, tl1, tl2);
				
				if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
					break;
				}

				if (verbose) {
					System.out.println("Iteration " + iter + ": " + currLoss);
				}
			}
		}

		if (refit) {
			byte[] structure = extractStructure(w);
			GAM gam = refitClassifier(attrs, structure, x, y, knots, w, maxNumIters);
			return gam;
		} else {
			return getGAM(attrs, knots, w, intercept);
		}
	}
	
	/**
	 * Returns a binary classifier.
	 * 
	 * @param attrs the attribute list.
	 * @param indices the indices.
	 * @param values the values.
	 * @param y the targets.
	 * @param knots the knots.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @param alpha the alpha.
	 * @return a binary classifier.
	 */
	public GAM buildBinaryClassifier(int[] attrs, int[][] indices, double[][][] values, double[] y, double[][] knots,
			int maxNumIters, double lambda, double alpha) {
		double[][] w = new double[attrs.length][];
		int m = 0;
		for (int j = 0; j < attrs.length; j++) {
			w[j] = new double[values[j].length];
			if (w[j].length > m) {
				m = w[j].length;
			}
		}
		double[] tl1 = new double[attrs.length];
		double[] tl2 = new double[attrs.length];
		getRegularizationParameters(lambda, alpha, tl1, tl2, y.length);

		double[] pTrain = new double[y.length];
		double[] rTrain = new double[y.length];
		OptimUtils.computePseudoResidual(pTrain, y, rTrain);

		double[] stepSize = new double[attrs.length];
		for (int j = 0; j < stepSize.length; j++) {
			double max = 0;
			double[][] block = values[j];
			for (double[] t : block) {
				double l = StatUtils.sumSq(t) / 4;
				if (l > max) {
					max = l;
				}
			}
			stepSize[j] = 1.0 / max;
		}
		double[] g = new double[m];
		double[] gradient = new double[m];
		double[] gamma1 = new double[m];
		double[] gamma2 = new double[m - 1];
		
		boolean[] activeSet = new boolean[attrs.length];

		double intercept = 0;
		
		// Block coordinate gradient descent
		int iter = 0;
		while (iter < maxNumIters) {

			if (fitIntercept) {
				intercept += OptimUtils.fitIntercept(pTrain, y, rTrain);
			}

			boolean activeSetChanged = doOnePass(indices, values, y, tl1, tl2, true, activeSet, w, stepSize, 
					g, gradient, gamma1, gamma2, pTrain, rTrain);

			iter++;

			if (!activeSetChanged || iter > maxNumIters) {
				break;
			}

			for (; iter < maxNumIters; iter++) {

				double prevLoss = OptimUtils.computeLogisticLoss(pTrain, y) + getPenalty(w, tl1, tl2);
				
				if (fitIntercept) {
					intercept += OptimUtils.fitIntercept(pTrain, rTrain, y);
				}

				doOnePass(indices, values, y, tl1, tl2, false, activeSet, w, stepSize, g, gradient, gamma1, gamma2, pTrain, rTrain);

				double currLoss = OptimUtils.computeLogisticLoss(pTrain, y) + getPenalty(w, tl1, tl2);
				
				if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
					break;
				}

				if (verbose) {
					System.out.println("Iteration " + iter + ": " + currLoss);
				}
			}
		}

		if (refit) {
			byte[] structure = extractStructure(w);
			GAM gam = refitClassifier(attrs, structure, indices, values, y, knots, w, maxNumIters * 10);
			return gam;
		} else {
			return getGAM(attrs, knots, w, intercept);
		}
	}

	/**
	 * Builds a classifier.
	 * 
	 * @param trainSet the training set.
	 * @param isSparse <code>true</code> if the training set is treated as sparse.
	 * @param numKnots the number of knots.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @param alpha the alpha.
	 * @return a classifier.
	 */
	public GAM buildClassifier(Instances trainSet, boolean isSparse, int numKnots, int maxNumIters, double lambda,
			double alpha) {
		if (isSparse) {
			SparseDataset sd = getSparseDataset(trainSet, false);
			SparseDesignMatrix sm = SparseDesignMatrix.createCubicSplineDesignMatrix(trainSet.size(), sd.indices,
					sd.values, sd.stdList, numKnots);
			double[] y = sd.y;
			int[][] indices = sm.indices;
			double[][][] values = sm.values;
			double[][] knots = sm.knots;
			int[] attrs = sd.attrs;

			// Mapping from attribute index to index in design matrix
			Map<Integer, Integer> map = new HashMap<>();
			for (int j = 0; j < sd.attrs.length; j++) {
				map.put(attrs[j], j);
			}
			
			GAM gam = buildBinaryClassifier(attrs, indices, values, y, knots, maxNumIters, lambda, alpha);

			// Rescale weights in gam
			List<Regressor> regressors = gam.getRegressors();
			List<int[]> terms = gam.getTerms();
			double intercept = gam.getIntercept();
			for (int i = 0; i < regressors.size(); i++) {
				Regressor regressor = regressors.get(i);
				int attIndex = terms.get(i)[0];
				int idx = map.get(attIndex);
				double[] std = sm.std[idx];
				if (regressor instanceof LinearFunction) {
					LinearFunction func = (LinearFunction) regressor;
					func.setSlope(func.getSlope() / std[0]);
				} else if (regressor instanceof CubicSpline) {
					CubicSpline spline = (CubicSpline) regressor;
					double[] w = spline.getCoefficients();
					for (int j = 0; j < w.length; j++) {
						w[j] /= std[j];
					}
					double[] k = spline.getKnots();
					for (int j = 0; j < k.length; j++) {
						intercept -= w[j + 3] * CubicSpline.h(0, k[j]);
					}
				}
			}

			if (fitIntercept) {
				gam.setIntercept(intercept);
			}

			return gam;
		} else {
			DenseDataset dd = getDenseDataset(trainSet, false);
			DenseDesignMatrix dm = DenseDesignMatrix.createCubicSplineDesignMatrix(dd.x, dd.stdList, numKnots);
			double[] y = dd.y;
			double[][][] x = dm.x;
			double[][] knots = dm.knots;
			int[] attrs = dd.attrs;
			
			// Mapping from attribute index to index in design matrix
			Map<Integer, Integer> map = new HashMap<>();
			for (int j = 0; j < dd.attrs.length; j++) {
				map.put(dd.attrs[j], j);
			}

			GAM gam = buildBinaryClassifier(attrs, x, y, knots, maxNumIters, lambda, alpha);

			// Rescale weights in gam
			List<Regressor> regressors = gam.getRegressors();
			List<int[]> terms = gam.getTerms();
			for (int i = 0; i < regressors.size(); i++) {
				Regressor regressor = regressors.get(i);
				int attIndex = terms.get(i)[0];
				int idx = map.get(attIndex);
				double[] std = dm.std[idx];
				if (regressor instanceof LinearFunction) {
					LinearFunction func = (LinearFunction) regressor;
					func.setSlope(func.getSlope() / std[0]);
				} else if (regressor instanceof CubicSpline) {
					CubicSpline spline = (CubicSpline) regressor;
					double[] w = spline.getCoefficients();
					for (int j = 0; j < w.length; j++) {
						w[j] /= std[j];
					}
				}
			}

			return gam;
		}
	}

	/**
	 * Builds a classifier.
	 * 
	 * @param trainSet the training set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numKnots the number of knots.
	 * @param lambda the lambda.
	 * @param alpha the alpha.
	 * @param fitIntercept whether the intercept is included.
	 * @return a classifier.
	 */
	public GAM buildClassifier(Instances trainSet, int maxNumIters, int numKnots, double lambda, double alpha) {
		return buildClassifier(trainSet, isSparse(trainSet), maxNumIters, numKnots, lambda, alpha);
	}

	/**
	 * Builds a regressor.
	 * 
	 * @param trainSet the training set.
	 * @param isSparse <code>true</code> if the training set is treated as sparse.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numKnots the number of knots.
	 * @param lambda the lambda.
	 * @param alpha the alpha.
	 * @return a regressor.
	 */
	public GAM buildRegressor(Instances trainSet, boolean isSparse, int maxNumIters, int numKnots, double lambda,
			double alpha) {
		if (isSparse) {
			SparseDataset sd = getSparseDataset(trainSet, false);
			SparseDesignMatrix sm = SparseDesignMatrix.createCubicSplineDesignMatrix(trainSet.size(), sd.indices,
					sd.values, sd.stdList, numKnots);
			double[] y = sd.y;
			int[][] indices = sm.indices;
			double[][][] values = sm.values;
			double[][] knots = sm.knots;
			int[] attrs = sd.attrs;

			// Mapping from attribute index to index in design matrix
			Map<Integer, Integer> map = new HashMap<>();
			for (int j = 0; j < sd.attrs.length; j++) {
				map.put(attrs[j], j);
			}
			
			GAM gam = buildRegressor(attrs, indices, values, y, knots, maxNumIters, lambda, alpha);

			// Rescale weights in gam
			List<Regressor> regressors = gam.getRegressors();
			List<int[]> terms = gam.getTerms();
			double intercept = gam.getIntercept();
			for (int i = 0; i < regressors.size(); i++) {
				Regressor regressor = regressors.get(i);
				int attIndex = terms.get(i)[0];
				int idx = map.get(attIndex);
				double[] std = sm.std[idx];
				if (regressor instanceof LinearFunction) {
					LinearFunction func = (LinearFunction) regressor;
					func.setSlope(func.getSlope() / std[0]);
				} else if (regressor instanceof CubicSpline) {
					CubicSpline spline = (CubicSpline) regressor;
					double[] w = spline.getCoefficients();
					for (int j = 0; j < w.length; j++) {
						w[j] /= std[j];
					}
					double[] k = spline.getKnots();
					for (int j = 0; j < k.length; j++) {
						intercept -= w[j + 3] * CubicSpline.h(0, k[j]);
					}
				}
			}

			if (fitIntercept) {
				gam.setIntercept(intercept);
			}

			return gam;
		} else {
			DenseDataset dd = getDenseDataset(trainSet, false);
			DenseDesignMatrix dm = DenseDesignMatrix.createCubicSplineDesignMatrix(dd.x, dd.stdList, numKnots);
			double[] y = dd.y;
			double[][][] x = dm.x;
			double[][] knots = dm.knots;
			int[] attrs = dd.attrs;
			
			// Mapping from attribute index to index in design matrix
			Map<Integer, Integer> map = new HashMap<>();
			for (int j = 0; j < dd.attrs.length; j++) {
				map.put(dd.attrs[j], j);
			}

			GAM gam = buildRegressor(attrs, x, y, knots, maxNumIters, lambda, alpha);

			// Rescale weights in gam
			List<Regressor> regressors = gam.getRegressors();
			List<int[]> terms = gam.getTerms();
			for (int i = 0; i < regressors.size(); i++) {
				Regressor regressor = regressors.get(i);
				int attIndex = terms.get(i)[0];
				int idx = map.get(attIndex);
				double[] std = dm.std[idx];
				if (regressor instanceof LinearFunction) {
					LinearFunction func = (LinearFunction) regressor;
					func.setSlope(func.getSlope() / std[0]);
				} else if (regressor instanceof CubicSpline) {
					CubicSpline spline = (CubicSpline) regressor;
					double[] w = spline.getCoefficients();
					for (int j = 0; j < w.length; j++) {
						w[j] /= std[j];
					}
				}
			}

			return gam;
		}
	}

	/**
	 * Builds a regressor.
	 * 
	 * @param trainSet the training set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numKnots the number of knots.
	 * @param lambda the lambda.
	 * @param alpha the alpha.
	 * @return a regressor.
	 */
	public GAM buildRegressor(Instances trainSet, int maxNumIters, int numKnots, double lambda,
			double alpha) {
		return buildRegressor(trainSet, isSparse(trainSet), maxNumIters, numKnots, lambda, alpha);
	}

	/**
	 * Returns a regressor.
	 * 
	 * @param attrs the attribute list.
	 * @param x the inputs.
	 * @param y the targets.
	 * @param knots the knots.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @param alpha the alpha.
	 * @return a regressor.
	 */
	public GAM buildRegressor(int[] attrs, double[][][] x, double[] y, double[][] knots, int maxNumIters,
			double lambda, double alpha) {
		// Backup targets
		double[] rTrain = new double[y.length];
		for (int i = 0; i < rTrain.length; i++) {
			rTrain[i] = y[i];
		}

		double[][] w = new double[attrs.length][];
		int m = 0;
		for (int j = 0; j < attrs.length; j++) {
			w[j] = new double[x[j].length];
			if (w[j].length > m) {
				m = w[j].length;
			}
		}
		double[] tl1 = new double[attrs.length];
		double[] tl2 = new double[attrs.length];
		getRegularizationParameters(lambda, alpha, tl1, tl2, y.length);

		double[] stepSize = new double[attrs.length];
		for (int j = 0; j < stepSize.length; j++) {
			double max = 0;
			double[][] block = x[j];
			for (double[] t : block) {
				double l = StatUtils.sumSq(t);
				if (l > max) {
					max = l;
				}
			}
			stepSize[j] = 1.0 / max;
		}
		double[] g = new double[m];
		double[] gradient = new double[m];
		double[] gamma1 = new double[m];
		double[] gamma2 = new double[m - 1];

		boolean[] activeSet = new boolean[attrs.length];

		double intercept = 0;
						
		// Block coordinate gradient descent
		int iter = 0;
		while (iter < maxNumIters) {

			if (fitIntercept) {
				intercept += OptimUtils.fitIntercept(rTrain);
			}

			boolean activeSetChanged = doOnePass(x, tl1, tl2, true, activeSet, w, stepSize, g, gradient, gamma1, gamma2, 
					rTrain);

			iter++;

			if (!activeSetChanged || iter > maxNumIters) {
				break;
			}

			for (; iter < maxNumIters; iter++) {

				double prevLoss = OptimUtils.computeQuadraticLoss(rTrain) + getPenalty(w, tl1, tl2);

				if (fitIntercept) {
					intercept += OptimUtils.fitIntercept(rTrain);
				}
				
				doOnePass(x, tl1, tl2, false, activeSet, w, stepSize, g, gradient, gamma1, gamma2, rTrain);
				
				double currLoss = OptimUtils.computeQuadraticLoss(rTrain) + getPenalty(w, tl1, tl2);

				if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
					break;
				}

				if (verbose) {
					System.out.println("Iteration " + iter + ": " + currLoss);
				}
			}
		}

		if (refit) {
			byte[] struct = extractStructure(w);
			return refitRegressor(attrs, struct, x, y, knots, w, maxNumIters);
		} else {
			return getGAM(attrs, knots, w, intercept);
		}
	}
	
	/**
	 * Returns a regressor.
	 * 
	 * @param attrs the attribute list.
	 * @param indices the indices.
	 * @param values the values.
	 * @param y the targets.
	 * @param knots the knots.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @param alpha the alpha.
	 * @return a regressor.
	 */
	public GAM buildRegressor(int[] attrs, int[][] indices, double[][][] values, double[] y, double[][] knots,
			int maxNumIters, double lambda, double alpha) {
		// Backup targets
		double[] rTrain = new double[y.length];
		for (int i = 0; i < rTrain.length; i++) {
			rTrain[i] = y[i];
		}
				
		double[][] w = new double[attrs.length][];
		int m = 0;
		for (int j = 0; j < attrs.length; j++) {
			w[j] = new double[values[j].length];
			if (w[j].length > m) {
				m = w[j].length;
			}
		}
		double[] tl1 = new double[attrs.length];
		double[] tl2 = new double[attrs.length];
		getRegularizationParameters(lambda, alpha, tl1, tl2, y.length);

		double[] stepSize = new double[attrs.length];
		for (int j = 0; j < stepSize.length; j++) {
			double max = 0;
			int[] index = indices[j];
			double[][] block = values[j];
			for (double[] t : block) {
				double l = 0;
				for (int i = 0; i < index.length; i++) {
					l += y[index[j]] * t[i];
				}
				if (l > max) {
					max = l;
				}
			}
			stepSize[j] = 1.0 / max;
		}
		double[] g = new double[m];
		double[] gradient = new double[m];
		double[] gamma1 = new double[m];
		double[] gamma2 = new double[m - 1];
		
		boolean[] activeSet = new boolean[attrs.length];

		double intercept = 0;
		
		// Block coordinate gradient descent
		int iter = 0;
		while (iter < maxNumIters) {

			if (fitIntercept) {
				intercept += OptimUtils.fitIntercept(rTrain);
			}

			boolean activeSetChanged = doOnePass(indices, values, tl1, tl2, true, activeSet, w, stepSize, 
					g, gradient, gamma1, gamma2, rTrain);

			iter++;

			if (!activeSetChanged || iter > maxNumIters) {
				break;
			}

			for (; iter < maxNumIters; iter++) {

				double prevLoss = OptimUtils.computeQuadraticLoss(rTrain) + getPenalty(w, tl1, tl2);
				
				if (fitIntercept) {
					intercept += OptimUtils.fitIntercept(rTrain);
				}

				doOnePass(indices, values, tl1, tl2, false, activeSet, w, stepSize, g, gradient, gamma1, gamma2, rTrain);

				double currLoss = OptimUtils.computeQuadraticLoss(rTrain) + getPenalty(w, tl1, tl2);
				
				if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
					break;
				}

				if (verbose) {
					System.out.println("Iteration " + iter + ": " + currLoss);
				}
			}
		}

		if (refit) {
			byte[] structure = extractStructure(w);
			GAM gam = refitRegressor(attrs, structure, indices, values, y, knots, w, maxNumIters * 10);
			return gam;
		} else {
			return getGAM(attrs, knots, w, intercept);
		}
	}

	/**
	 * Returns <code>true</code> if we fit intercept.
	 * 
	 * @return <code>true</code> if we fit intercept.
	 */
	public boolean fitIntercept() {
		return fitIntercept;
	}

	/**
	 * Sets whether we fit intercept.
	 * 
	 * @param fitIntercept whether we fit intercept.
	 */
	public void fitIntercept(boolean fitIntercept) {
		this.fitIntercept = fitIntercept;
	}

	/**
	 * Returns the alpha.
	 * 
	 * @return the alpha;
	 */
	public double getAlpha() {
		return alpha;
	}
	
	/**
	 * Returns the convergence threshold epsilon.
	 * 
	 * @return the convergence threshold epsilon.
	 */
	public double getEpsilon() {
		return epsilon;
	}

	/**
	 * Returns the lambda.
	 * 
	 * @return the lambda.
	 */
	public double getLambda() {
		return lambda;
	}
	
	/**
	 * Returns the maximum number of iterations.
	 * 
	 * @return the maximum number of iterations.
	 */
	public int getMaxNumIters() {
		return maxNumIters;
	}

	/**
	 * Returns the number of knots.
	 * 
	 * @return the number of knots.
	 */
	public int getNumKnots() {
		return numKnots;
	}

	/**
	 * Returns the task of this learner.
	 * 
	 * @return the task of this learner.
	 */
	public Task getTask() {
		return task;
	}

	/**
	 * Returns <code>true</code> if we output something during the training.
	 * 
	 * @return <code>true</code> if we output something during the training.
	 */
	public boolean isVerbose() {
		return verbose;
	}
	
	/**
	 * Returns <code>true</code> if we refit the model.
	 * 
	 * @return <code>true</code> if we refit the model.
	 */
	public boolean refit() {
		return refit;
	}

	/**
	 * Sets whether we refit the model.
	 * 
	 * @param refit <code>true</code> if we refit the model.
	 */
	public void refit(boolean refit) {
		this.refit = refit;
	}

	/**
	 * Sets the alpha.
	 * 
	 * @param alpha the alpha.
	 */
	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}

	/**
	 * Sets the convergence threshold epsilon.
	 * 
	 * @param epsilon the convergence threshold epsilon.
	 */
	public void setEpsilon(double epsilon) {
		this.epsilon = epsilon;
	}

	/**
	 * Sets the lambda.
	 * 
	 * @param lambda the lambda.
	 */
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	/**
	 * Sets the maximum number of iterations.
	 * 
	 * @param maxNumIters the maximum number of iterations.
	 */
	public void setMaxNumIters(int maxNumIters) {
		this.maxNumIters = maxNumIters;
	}
	
	/**
	 * Sets the number of knots.
	 * 
	 * @param numKnots the new number of knots.
	 */
	public void setNumKnots(int numKnots) {
		this.numKnots = numKnots;
	}
	
	/**
	 * Sets the task of this learner.
	 * 
	 * @param task the task of this learner.
	 */
	public void setTask(Task task) {
		this.task = task;
	}
	
	/**
	 * Sets whether we output something during the training.
	 * 
	 * @param verbose the switch if we output things during training.
	 */
	public void setVerbose(boolean verbose) {
		this.verbose = verbose;
	}

	protected void computeGradient(double[][] block, double[] rTrain, double[] gradient) {
		for (int i = 0; i < block.length; i++) {
			gradient[i] = VectorUtils.dotProduct(block[i], rTrain);
		}
	}

	protected void computeGradient(int[] index, double[][] block, double[] rTrain, double[] gradient) {
		for (int j = 0; j < block.length; j++) {
			double[] t = block[j];
			gradient[j] = 0;
			for (int i = 0; i < t.length; i++) {
				gradient[j] += rTrain[index[i]] * t[i];
			}
		}
	}

	protected boolean doOnePass(double[][][] x, double[] tl1, double[] tl2, boolean isFullPass, boolean[] activeSet,
			double[][] w, double[] stepSize, double[] g, double[] gradient, double[] gamma1, double[] gamma2, 
			double[] rTrain) {
		boolean activeSetChanged = false;

		for (int k = 0; k < x.length; k++) {
			if (!isFullPass && !activeSet[k]) {
				continue;
			}

			double[][] block = x[k];
			double[] beta = w[k];
			double tk = stepSize[k];
			double lambda1 = tl1[k];
			double lambda2 = tl2[k];

			// Proximal gradient method
			computeGradient(block, rTrain, gradient);

			for (int j = 0; j < beta.length; j++) {
				g[j] = beta[j] + tk * gradient[j];
			}

			// Dual method
			if (beta.length > 1) {
				for (int i = 1; i < beta.length; i++) {
					gamma2[i - 1] = g[i];
				}
				double norm2 = Math.sqrt(StatUtils.sumSq(gamma2, 0, beta.length - 1));
				double t2 = lambda2 * tk;
				if (norm2 > t2) {
					VectorUtils.multiply(gamma2, t2 / norm2);
				}
			}
			gamma1[0] = g[0];
			for (int i = 1; i < beta.length; i++) {
				gamma1[i] = g[i] - gamma2[i - 1];
			}
			double norm1 = Math.sqrt(StatUtils.sumSq(gamma1, 0, beta.length));
			double t1 = lambda1 * tk;
			if (norm1 > t1) {
				VectorUtils.multiply(gamma1, t1 / norm1);
			}
			g[0] -= gamma1[0];
			for (int i = 1; i < beta.length; i++) {
				g[i] -= (gamma1[i] + gamma2[i - 1]);
			}

			// Update residuals
			for (int j = 0; j < beta.length; j++) {
				double[] t = block[j];
				double delta = beta[j] - g[j];
				for (int i = 0; i < rTrain.length; i++) {
					rTrain[i] += delta * t[i];
				}
			}

			// Update weights
			for (int j = 0; j < beta.length; j++) {
				beta[j] = g[j];
			}

			if (isFullPass && !activeSet[k] && !ArrayUtils.isConstant(beta, 0, beta.length, 0)) {
				activeSetChanged = true;
				activeSet[k] = true;
			}
		}

		return activeSetChanged;
	}

	protected boolean doOnePass(double[][][] x, double[] y, double[] tl1, double[] tl2, boolean isFullPass, 
			boolean[] activeSet, double[][] w, double[] stepSize, double[] g, double[] gradient, double[] gamma1, 
			double[] gamma2, double[] pTrain, double[] rTrain) {
		boolean activeSetChanged = false;

		for (int k = 0; k < x.length; k++) {
			if (!isFullPass && !activeSet[k]) {
				continue;
			}

			double[][] block = x[k];
			double[] beta = w[k];
			double tk = stepSize[k];
			double lambda1 = tl1[k];
			double lambda2 = tl2[k];

			// Proximal gradient method
			computeGradient(block, rTrain, gradient);

			for (int j = 0; j < beta.length; j++) {
				g[j] = beta[j] + tk * gradient[j];
			}

			// Dual method
			if (beta.length > 1) {
				for (int i = 1; i < beta.length; i++) {
					gamma2[i - 1] = g[i];
				}
				double norm2 = Math.sqrt(StatUtils.sumSq(gamma2, 0, beta.length - 1));
				double t2 = lambda2 * tk;
				if (norm2 > t2) {
					VectorUtils.multiply(gamma2, t2 / norm2);
				}
			}
			gamma1[0] = g[0];
			for (int i = 1; i < beta.length; i++) {
				gamma1[i] = g[i] - gamma2[i - 1];
			}
			double norm1 = Math.sqrt(StatUtils.sumSq(gamma1, 0, beta.length));
			double t1 = lambda1 * tk;
			if (norm1 > t1) {
				VectorUtils.multiply(gamma1, t1 / norm1);
			}
			g[0] -= gamma1[0];
			for (int i = 1; i < beta.length; i++) {
				g[i] -= (gamma1[i] + gamma2[i - 1]);
			}

			// Update predictions
			for (int j = 0; j < beta.length; j++) {
				double[] t = block[j];
				double delta = g[j] - beta[j];
				for (int i = 0; i < y.length; i++) {
					pTrain[i] += delta * t[i];
				}
			}
			OptimUtils.computePseudoResidual(pTrain, y, rTrain);

			// Update weights
			for (int j = 0; j < beta.length; j++) {
				beta[j] = g[j];
			}

			if (isFullPass && !activeSet[k] && !ArrayUtils.isConstant(beta, 0, beta.length, 0)) {
				activeSetChanged = true;
				activeSet[k] = true;
			}
		}

		return activeSetChanged;
	}

	protected boolean doOnePass(int[][] indices, double[][][] values, double[] tl1, double[] tl2, boolean isFullPass,
			boolean[] activeSet, double[][] w, double[] stepSize, double[] g, double[] gradient, double[] gamma1, 
			double[] gamma2, double[] rTrain) {
		boolean activeSetChanged = false;

		for (int k = 0; k < values.length; k++) {
			if (!isFullPass && !activeSet[k]) {
				continue;
			}

			double[][] block = values[k];
			int[] index = indices[k];
			double[] beta = w[k];
			double tk = stepSize[k];
			double lambda1 = tl1[k];
			double lambda2 = tl2[k];

			// Proximal gradient method
			computeGradient(index, block, rTrain, gradient);

			for (int j = 0; j < beta.length; j++) {
				g[j] = beta[j] + tk * gradient[j];
			}

			// Dual method
			if (beta.length > 1) {
				for (int i = 1; i < beta.length; i++) {
					gamma2[i - 1] = g[i];
				}
				double norm2 = Math.sqrt(StatUtils.sumSq(gamma2, 0, beta.length - 1));
				double t2 = lambda2 * tk;
				if (norm2 > t2) {
					VectorUtils.multiply(gamma2, t2 / norm2);
				}
			}
			gamma1[0] = g[0];
			for (int i = 1; i < beta.length; i++) {
				gamma1[i] = g[i] - gamma2[i - 1];
			}
			double norm1 = Math.sqrt(StatUtils.sumSq(gamma1, 0, beta.length));
			double t1 = lambda1 * tk;
			if (norm1 > t1) {
				VectorUtils.multiply(gamma1, t1 / norm1);
			}
			g[0] -= gamma1[0];
			for (int i = 1; i < beta.length; i++) {
				g[i] -= (gamma1[i] + gamma2[i - 1]);
			}
			
			// Update predictions
			for (int j = 0; j < beta.length; j++) {
				double[] t = block[j];
				double delta = beta[j] - g[j];
				for (int i = 0; i < t.length; i++) {
					rTrain[index[i]] += delta * t[i];
				}
			}

			// Update weights
			for (int j = 0; j < beta.length; j++) {
				beta[j] = g[j];
			}

			if (isFullPass && !activeSet[k] && !ArrayUtils.isConstant(beta, 0, beta.length, 0)) {
				activeSetChanged = true;
				activeSet[k] = true;
			}
		}

		return activeSetChanged;
	}

	protected boolean doOnePass(int[][] indices, double[][][] values, double[] y, double[] tl1, double[] tl2,
			boolean isFullPass, boolean[] activeSet, double[][] w, double[] stepSize, double[] g, double[] gradient,
			double[] gamma1, double[] gamma2, double[] pTrain, double[] rTrain) {
		boolean activeSetChanged = false;

		for (int k = 0; k < values.length; k++) {
			if (!isFullPass && !activeSet[k]) {
				continue;
			}

			int[] index = indices[k];
			double[][] block = values[k];
			double[] beta = w[k];
			double tk = stepSize[k];
			double lambda1 = tl1[k];
			double lambda2 = tl2[k];

			// Proximal gradient method
			computeGradient(index, block, rTrain, gradient);

			for (int j = 0; j < beta.length; j++) {
				g[j] = beta[j] + tk * gradient[j];
			}

			// Dual method
			if (beta.length > 1) {
				for (int j = 1; j < beta.length; j++) {
					gamma2[j - 1] = g[j];
				}
				double norm2 = Math.sqrt(StatUtils.sumSq(gamma2, 0, beta.length - 1));
				double t2 = lambda2 * tk;
				if (norm2 > t2) {
					VectorUtils.multiply(gamma2, t2 / norm2);
				}
			}
			gamma1[0] = g[0];
			for (int j = 1; j < beta.length; j++) {
				gamma1[j] = g[j] - gamma2[j - 1];
			}
			double norm1 = Math.sqrt(StatUtils.sumSq(gamma1, 0, beta.length));
			double t1 = lambda1 * tk;
			if (norm1 > t1) {
				VectorUtils.multiply(gamma1, t1 / norm1);
			}
			g[0] -= gamma1[0];
			for (int i = 1; i < beta.length; i++) {
				g[i] -= (gamma1[i] + gamma2[i - 1]);
			}
			
			// Update predictions
			for (int j = 0; j < beta.length; j++) {
				double[] value = block[j];
				double delta = g[j] - beta[j];
				for (int i = 0; i < value.length; i++) {
					pTrain[index[i]] += delta * value[i];
				}
			}
			
			for (int idx : index) {
				rTrain[idx] = OptimUtils.getPseudoResidual(pTrain[idx], y[idx]);
			}

			// Update weights
			for (int j = 0; j < beta.length; j++) {
				beta[j] = g[j];
			}

			if (isFullPass && !activeSet[k] && !ArrayUtils.isConstant(beta, 0, beta.length, 0)) {
				activeSetChanged = true;
				activeSet[k] = true;
			}
		}

		return activeSetChanged;
	}

	protected byte[] extractStructure(double[][] w) {
		byte[] structure = new byte[w.length];
		for (int i = 0; i < structure.length; i++) {
			double[] beta = w[i];
			boolean isLinear = beta.length == 1 || ArrayUtils.isConstant(beta, 1, beta.length, 0);
			if (isLinear) {
				if (beta[0] != 0) {
					structure[i] = ModelStructure.LINEAR;
				} else {
					structure[i] = ModelStructure.ELIMINATED;
				}
			} else {
				structure[i] = ModelStructure.NONLINEAR;
			}
		}
		return structure;
	}

	protected GAM getGAM(int[] attrs, double[][] knots, double[][] w, double intercept) {
		GAM gam = new GAM();
		for (int j = 0; j < attrs.length; j++) {
			int attIndex = attrs[j];
			double[] beta = w[j];
			boolean isLinear = beta.length == 1 || ArrayUtils.isConstant(beta, 1, beta.length, 0);
			if (isLinear) {
				if (beta[0] != 0) {
					// To rule out a feature, it has to be "linear" and 0 slope.
					gam.add(new int[] { attIndex }, new LinearFunction(attIndex, beta[0]));
				}
			} else {
				double[] coef = Arrays.copyOf(beta, beta.length);
				CubicSpline spline = new CubicSpline(attIndex, 0, knots[j], coef);
				gam.add(new int[] { attIndex }, spline);
			}
		}
		gam.setIntercept(intercept);
		return gam;
	}
	
	protected double getPenalty(double[] w, double lambda1, double lambda2) {
		double penalty = 0;
		double sumSq = StatUtils.sumSq(w);
		double norm1 = Math.sqrt(sumSq);
		penalty += lambda1 * norm1;
		double norm2 = sumSq - w[0] * w[0];
		norm2 = Math.sqrt(norm2);
		penalty += lambda2 * norm2;
		return penalty;
	}

	protected double getPenalty(double[][] coef, double[] lambda1, double[] lambda2) {
		double penalty = 0;
		for (int i = 0; i < coef.length; i++) {
			penalty += getPenalty(coef[i], lambda1[i], lambda2[i]);
		}
		return penalty;
	}

	protected void getRegularizationParameters(double lambda, double alpha, double[] tl1, double[] tl2, int n) {
		for (int j = 0; j < tl1.length; j++) {
			tl1[j] = lambda * alpha * n;
			tl2[j] = lambda * (1 - alpha) * n;
		}
	}

	protected GAM refitClassifier(int[] attrs, byte[] struct, double[][][] x, double[] y, double[][] knots, 
			double[][] w, int maxNumIters) {
		List<double[]> xList = new ArrayList<>();
		for (int i = 0; i < struct.length; i++) {
			if (struct[i] == ModelStructure.NONLINEAR) {
				double[][] t = x[i];
				for (int j = 0; j < t.length; j++) {
					xList.add(t[j]);
				}
			} else if (struct[i] == ModelStructure.LINEAR) {
				xList.add(x[i][0]);
			}
		}

		double[][] xNew = new double[xList.size()][];
		for (int i = 0; i < xNew.length; i++) {
			xNew[i] = xList.get(i);
		}

		int[] attrsNew = new int[xNew.length];
		for (int i = 0; i < attrsNew.length; i++) {
			attrsNew[i] = i;
		}

		RidgeLearner ridgeLearner = new RidgeLearner();
		ridgeLearner.setVerbose(verbose);
		ridgeLearner.setEpsilon(epsilon);
		ridgeLearner.fitIntercept(fitIntercept);
		// A ridge regression with very small regularization parameter
		// This often improves stability a lot
		GLM glm = ridgeLearner.buildBinaryClassifier(attrsNew, xNew, y, maxNumIters, 1e-8);
		
		GAM gam = getGAM(attrs, knots, w, glm.intercept(0));
		List<Regressor> regressors = gam.regressors;
		double[] coef = glm.coefficients(0);
		int k = 0;
		for (Regressor regressor : regressors) {
			if (regressor instanceof LinearFunction) {
				LinearFunction func = (LinearFunction) regressor;
				func.setSlope(coef[k++]);
			} else if (regressor instanceof CubicSpline) {
				CubicSpline spline = (CubicSpline) regressor;
				double[] beta = spline.getCoefficients();
				for (int i = 0; i < beta.length; i++) {
					beta[i] = coef[k++];
				}
			}
		}

		return gam;
	}

	protected GAM refitClassifier(int[] attrs, byte[] struct, int[][] indices, double[][][] values, double[] y, 
			double[][] knots, double[][] w, int maxNumIters) {
		List<int[]> iList = new ArrayList<>();
		List<double[]> vList = new ArrayList<>();
		for (int i = 0; i < struct.length; i++) {
			int[] index = indices[i];
			if (struct[i] == ModelStructure.NONLINEAR) {
				double[][] t = values[i];
				for (int j = 0; j < t.length; j++) {
					iList.add(index);
					vList.add(t[j]);
				}
			} else if (struct[i] == ModelStructure.LINEAR) {
				iList.add(index);
				vList.add(values[i][0]);
			}
		}

		int[][] iNew = new int[iList.size()][];
		for (int i = 0; i < iNew.length; i++) {
			iNew[i] = iList.get(i);
		}
		double[][] vNew = new double[vList.size()][];
		for (int i = 0; i < vNew.length; i++) {
			vNew[i] = vList.get(i);
		}

		int[] attrsNew = new int[iNew.length];
		for (int i = 0; i < attrsNew.length; i++) {
			attrsNew[i] = i;
		}

		RidgeLearner ridgeLearner = new RidgeLearner();
		ridgeLearner.setVerbose(verbose);
		ridgeLearner.setEpsilon(epsilon);
		ridgeLearner.fitIntercept(fitIntercept);
		// A ridge regression with very small regularization parameter
		// This often improves stability a lot
		GLM glm = ridgeLearner.buildBinaryClassifier(attrsNew, iNew, vNew, y, maxNumIters, 1e-8);
		
		GAM gam = getGAM(attrs, knots, w, glm.intercept(0));
		List<Regressor> regressors = gam.regressors;
		double[] coef = glm.coefficients(0);
		int k = 0;
		for (Regressor regressor : regressors) {
			if (regressor instanceof LinearFunction) {
				LinearFunction func = (LinearFunction) regressor;
				func.setSlope(coef[k++]);
			} else if (regressor instanceof CubicSpline) {
				CubicSpline spline = (CubicSpline) regressor;
				double[] beta = spline.getCoefficients();
				for (int j = 0; j < beta.length; j++) {
					beta[j] = coef[k++];
				}
			}
		}

		return gam;
	}
	
	protected GAM refitRegressor(int[] attrs, byte[] struct, double[][][] x, double[] y, double[][] knots,
			double[][] w, int maxNumIters) {
		List<double[]> xList = new ArrayList<>();
		for (int i = 0; i < struct.length; i++) {
			if (struct[i] == ModelStructure.NONLINEAR) {
				double[][] t = x[i];
				for (int j = 0; j < t.length; j++) {
					xList.add(t[j]);
				}
			} else if (struct[i] == ModelStructure.LINEAR) {
				xList.add(x[i][0]);
			}
		}

		if (xList.size() == 0) {
			if (fitIntercept) {
				double intercept = StatUtils.mean(y);
				GAM gam = new GAM();
				gam.setIntercept(intercept);
				return gam;
			} else {
				return new GAM();
			}
		}

		double[][] xNew = new double[xList.size()][];
		for (int i = 0; i < xNew.length; i++) {
			xNew[i] = xList.get(i);
		}

		int[] attrsNew = new int[xNew.length];
		for (int i = 0; i < attrsNew.length; i++) {
			attrsNew[i] = i;
		}

		RidgeLearner ridgeLearner = new RidgeLearner();
		ridgeLearner.setVerbose(verbose);
		ridgeLearner.setEpsilon(epsilon);
		ridgeLearner.fitIntercept(fitIntercept);
		// A ridge regression with very small regularization parameter
		// This often improves stability a lot
		GLM glm = ridgeLearner.buildRegressor(attrsNew, xNew, y, maxNumIters, 1e-8);
		
		GAM gam = getGAM(attrs, knots, w, glm.intercept(0));
		List<Regressor> regressors = gam.regressors;
		double[] coef = glm.coefficients(0);
		int k = 0;
		for (Regressor regressor : regressors) {
			if (regressor instanceof LinearFunction) {
				LinearFunction func = (LinearFunction) regressor;
				func.setSlope(coef[k++]);
			} else if (regressor instanceof CubicSpline) {
				CubicSpline spline = (CubicSpline) regressor;
				double[] beta = spline.getCoefficients();
				for (int i = 0; i < beta.length; i++) {
					beta[i] = coef[k++];
				}
			}
		}

		return gam;
	}

	protected GAM refitRegressor(int[] attrs, byte[] struct, int[][] indices, double[][][] values, double[] y,
			double[][] knots, double[][] w, int maxNumIters) {
		List<int[]> iList = new ArrayList<>();
		List<double[]> vList = new ArrayList<>();
		for (int i = 0; i < struct.length; i++) {
			int[] index = indices[i];
			if (struct[i] == ModelStructure.NONLINEAR) {
				double[][] t = values[i];
				for (int j = 0; j < t.length; j++) {
					iList.add(index);
					vList.add(t[j]);
				}
			} else if (struct[i] == ModelStructure.LINEAR) {
				iList.add(index);
				vList.add(values[i][0]);
			}
		}

		int[][] iNew = new int[iList.size()][];
		for (int i = 0; i < iNew.length; i++) {
			iNew[i] = iList.get(i);
		}
		double[][] vNew = new double[vList.size()][];
		for (int i = 0; i < vNew.length; i++) {
			vNew[i] = vList.get(i);
		}

		int[] attrsNew = new int[iNew.length];
		for (int i = 0; i < attrsNew.length; i++) {
			attrsNew[i] = i;
		}

		RidgeLearner ridgeLearner = new RidgeLearner();
		ridgeLearner.setVerbose(verbose);
		ridgeLearner.setEpsilon(epsilon);
		ridgeLearner.fitIntercept(fitIntercept);
		// A ridge regression with very small regularization parameter
		// This often improves stability a lot
		GLM glm = ridgeLearner.buildRegressor(attrsNew, iNew, vNew, y, maxNumIters, 1e-8);
		
		GAM gam = getGAM(attrs, knots, w, glm.intercept(0));
		List<Regressor> regressors = gam.regressors;
		double[] coef = glm.coefficients(0);
		int k = 0;
		for (Regressor regressor : regressors) {
			if (regressor instanceof LinearFunction) {
				LinearFunction func = (LinearFunction) regressor;
				func.setSlope(coef[k++]);
			} else if (regressor instanceof CubicSpline) {
				CubicSpline spline = (CubicSpline) regressor;
				double[] beta = spline.getCoefficients();
				for (int j = 0; j < beta.length; j++) {
					beta[j] = coef[k++];
				}
			}
		}

		return gam;
	}

}
