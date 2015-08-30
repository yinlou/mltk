package mltk.predictor.glm;

import java.util.Arrays;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.cmdline.options.LearnerWithTaskOptions;
import mltk.core.Attribute;
import mltk.core.Instances;
import mltk.core.NominalAttribute;
import mltk.core.io.InstancesReader;
import mltk.predictor.Learner;
import mltk.predictor.io.PredictorWriter;
import mltk.util.MathUtils;
import mltk.util.OptimUtils;
import mltk.util.StatUtils;
import mltk.util.VectorUtils;

/**
 * Class for learning L2-regularized linear model via coordinate descent.
 * 
 * @author Yin Lou
 * 
 */
public class RidgeLearner extends Learner {

	static class Options extends LearnerWithTaskOptions {

		@Argument(name = "-m", description = "maximum num of iterations (default: 0)")
		int maxIter = 0;

		@Argument(name = "-l", description = "lambda (default: 0)")
		double lambda = 0;

	}

	/**
	 * <p>
	 * 
	 * <pre>
	 * Usage: mltk.predictor.glm.RidgeLearner
	 * -t	train set path
	 * [-g]	task between classification (c) and regression (r) (default: r)
	 * [-r]	attribute file path
	 * [-o]	output model path
	 * [-V]	verbose (default: true)
	 * [-m]	maximum num of iterations (default: 0)
	 * [-l]	lambda (default: 0)
	 * </pre>
	 * </p>
	 * 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(RidgeLearner.class, opts);
		Task task = null;
		try {
			parser.parse(args);
			task = Task.getEnum(opts.task);
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}
		Instances trainSet = InstancesReader.read(opts.attPath, opts.trainPath);

		RidgeLearner learner = new RidgeLearner();
		learner.setVerbose(opts.verbose);
		learner.setTask(task);
		learner.setLambda(opts.lambda);
		learner.setMaxNumIters(opts.maxIter);

		long start = System.currentTimeMillis();
		GLM glm = learner.build(trainSet);
		long end = System.currentTimeMillis();
		System.out.println("Time: " + (end - start) / 1000.0);

		if (opts.outputModelPath != null) {
			PredictorWriter.write(glm, opts.outputModelPath);
		}
	}

	private boolean fitIntercept;
	private int maxNumIters;
	private double epsilon;
	private double lambda;
	private Task task;

	/**
	 * Constructor.
	 */
	public RidgeLearner() {
		verbose = false;
		fitIntercept = true;
		maxNumIters = -1;
		epsilon = MathUtils.EPSILON;
		lambda = 0; // no regularization
		task = Task.REGRESSION;
	}

	@Override
	public GLM build(Instances instances) {
		GLM glm = null;
		if (maxNumIters < 0) {
			maxNumIters = 20;
		}
		switch (task) {
			case REGRESSION:
				glm = buildRegressor(instances, maxNumIters, lambda);
				break;
			case CLASSIFICATION:
				glm = buildClassifier(instances, maxNumIters, lambda);
				break;
			default:
				break;
		}
		return glm;
	}

	/**
	 * Builds an L2-regularized binary classifier. Each row in the input matrix x represents a feature (instead of a
	 * data point). Thus the input matrix is the transpose of the row-oriented data matrix. This procedure does not
	 * assume the data is normalized or centered.
	 * 
	 * @param attrs the attribute list.
	 * @param x the inputs.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @return an L2-regularized binary classifier.
	 */
	public GLM buildBinaryClassifier(int[] attrs, double[][] x, double[] y, int maxNumIters, double lambda) {
		double[] w = new double[attrs.length];
		double intercept = 0;

		double[] pTrain = new double[y.length];
		double[] rTrain = new double[y.length];
		OptimUtils.computePseudoResidual(pTrain, y, rTrain);

		// Calculate theta's
		double[] theta = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			theta[i] = StatUtils.sumSq(x[i]) / 4;
		}

		// Coordinate gradient descent
		final double tl2 = lambda * y.length;
		for (int iter = 0; iter < maxNumIters; iter++) {
			double prevLoss = GLMOptimUtils.computeRidgeLoss(pTrain, y, w, lambda);
			
			if (fitIntercept) {
				intercept += OptimUtils.fitIntercept(pTrain, rTrain, y);
			}

			doOnePass(x, theta, y, tl2, w, pTrain, rTrain);

			double currLoss = GLMOptimUtils.computeRidgeLoss(pTrain, y, w, lambda);

			if (verbose) {
				System.out.println("Iteration " + iter + ": " + " " + currLoss);
			}

			if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
				break;
			}
		}

		return GLMOptimUtils.getGLM(attrs, w, intercept);
	}

	/**
	 * Builds an L2-regularized binary classifier on sparse inputs. Each row of the input represents a feature (instead
	 * of a data point), i.e., in column-oriented format. This procedure does not assume the data is normalized or
	 * centered.
	 * 
	 * @param attrs the attribute list.
	 * @param indices the indices.
	 * @param values the values.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @return an L2-regularized classifier.
	 */
	public GLM buildBinaryClassifier(int[] attrs, int[][] indices, double[][] values, double[] y, int maxNumIters,
			double lambda) {
		double[] w = new double[attrs.length];
		double intercept = 0;

		double[] pTrain = new double[y.length];
		double[] rTrain = new double[y.length];
		OptimUtils.computePseudoResidual(pTrain, y, rTrain);

		// Calculate theta's
		double[] theta = new double[values.length];
		for (int i = 0; i < values.length; i++) {
			theta[i] = StatUtils.sumSq(values[i]) / 4;
		}

		// Coordinate gradient descent
		final double tl2 = lambda * y.length;
		for (int iter = 0; iter < maxNumIters; iter++) {
			double prevLoss = GLMOptimUtils.computeRidgeLoss(pTrain, y, w, lambda);
			
			if (fitIntercept) {
				intercept += OptimUtils.fitIntercept(pTrain, rTrain, y);
			}

			doOnePass(indices, values, theta, y, tl2, w, pTrain, rTrain);

			double currLoss = GLMOptimUtils.computeRidgeLoss(pTrain, y, w, lambda);

			if (verbose) {
				System.out.println("Iteration " + iter + ": " + " " + currLoss);
			}

			if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
				break;
			}
		}

		return GLMOptimUtils.getGLM(attrs, w, intercept);
	}

	/**
	 * Builds L2-regularized binary classifiers for a sequence of regularization parameter lambdas. Each row in the
	 * input matrix x represents a feature (instead of a data point). Thus the input matrix is the transpose of the
	 * row-oriented data matrix. This procedure does not assume the data is normalized or centered.
	 * 
	 * @param attrs the attribute list.
	 * @param x the values.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambdas the lambdas array.
	 * @return L2-regularized classifiers.
	 */
	public GLM[] buildBinaryClassifiers(int[] attrs, double[][] x, double[] y, int maxNumIters, double[] lambdas) {
		double[] w = new double[attrs.length];
		double intercept = 0;

		double[] pTrain = new double[y.length];
		double[] rTrain = new double[y.length];
		OptimUtils.computePseudoResidual(pTrain, y, rTrain);

		// Calculate theta's
		double[] theta = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			theta[i] = StatUtils.sumSq(x[i]) / 4;
		}

		GLM[] glms = new GLM[lambdas.length];
		Arrays.sort(lambdas);

		for (int g = 0; g < glms.length; g++) {
			double lambda = lambdas[g];
			// Coordinate gradient descent
			final double tl2 = lambda * y.length;
			for (int iter = 0; iter < maxNumIters; iter++) {
				double prevLoss = GLMOptimUtils.computeRidgeLoss(pTrain, y, w, lambda);
				
				if (fitIntercept) {
					intercept += OptimUtils.fitIntercept(pTrain, rTrain, y);
				}

				doOnePass(x, theta, y, tl2, w, pTrain, rTrain);

				double currLoss = GLMOptimUtils.computeRidgeLoss(pTrain, y, w, lambda);

				if (verbose) {
					System.out.println("Iteration " + iter + ": " + " " + currLoss);
				}

				if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
					break;
				}
			}

			glms[g] = GLMOptimUtils.getGLM(attrs, w, intercept);
		}

		return glms;
	}

	/**
	 * Builds L2-regularized binary classifiers for a sequence of regularization parameter lambdas on sparse format.
	 * Each row of the input represents a feature (instead of a data point), i.e., in column-oriented format. This
	 * procedure does not assume the data is normalized or centered.
	 * 
	 * @param attrs the attribute list.
	 * @param indices the indices.
	 * @param values the values.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambdas the lambdas array.
	 * @return L2-regularized classifiers.
	 */
	public GLM[] buildBinaryClassifiers(int[] attrs, int[][] indices, double[][] values, double[] y, int maxNumIters,
			double[] lambdas) {
		double[] w = new double[attrs.length];
		double intercept = 0;

		double[] pTrain = new double[y.length];
		double[] rTrain = new double[y.length];
		OptimUtils.computePseudoResidual(pTrain, y, rTrain);

		// Calculate theta's
		double[] theta = new double[values.length];
		for (int i = 0; i < values.length; i++) {
			theta[i] = StatUtils.sumSq(values[i]) / 4;
		}

		GLM[] glms = new GLM[lambdas.length];
		Arrays.sort(lambdas);

		for (int g = 0; g < glms.length; g++) {
			double lambda = lambdas[g];

			// Coordinate gradient descent
			final double tl2 = lambda * y.length;
			for (int iter = 0; iter < maxNumIters; iter++) {
				double prevLoss = GLMOptimUtils.computeRidgeLoss(pTrain, y, w, lambda);
				
				if (fitIntercept) {
					intercept += OptimUtils.fitIntercept(pTrain, rTrain, y);
				}

				doOnePass(indices, values, theta, y, tl2, w, pTrain, rTrain);

				double currLoss = GLMOptimUtils.computeRidgeLoss(pTrain, y, w, lambda);

				if (verbose) {
					System.out.println("Iteration " + iter + ": " + " " + currLoss);
				}

				if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
					break;
				}
			}

			glms[g] = GLMOptimUtils.getGLM(attrs, w, intercept);
		}

		return glms;
	}

	/**
	 * Builds an L2-regularized binary classifier.
	 * 
	 * @param trainSet the training set.
	 * @param isSparse <code>true</code> if the training set is treated as sparse.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @return an L2-regularized binary classifier.
	 */
	public GLM buildClassifier(Instances trainSet, boolean isSparse, int maxNumIters, double lambda) {
		Attribute classAttribute = trainSet.getTargetAttribute();
		if (classAttribute.getType() != Attribute.Type.NOMINAL) {
			throw new IllegalArgumentException("Class attribute must be nominal.");
		}
		NominalAttribute clazz = (NominalAttribute) classAttribute;
		int numClasses = clazz.getStates().length;

		if (isSparse) {
			SparseDataset sd = getSparseDataset(trainSet, true);
			int[] attrs = sd.attrs;
			int[][] indices = sd.indices;
			double[][] values = sd.values;
			double[] y = new double[sd.y.length];
			double[] cList = sd.cList;

			if (numClasses == 2) {
				for (int i = 0; i < y.length; i++) {
					int label = (int) sd.y[i];
					y[i] = label == 0 ? 1 : 0;
				}

				GLM glm = buildBinaryClassifier(attrs, indices, values, y, maxNumIters, lambda);

				double[] w = glm.w[0];
				for (int j = 0; j < cList.length; j++) {
					int attIndex = attrs[j];
					w[attIndex] *= cList[j];
				}

				return glm;
			} else {
				int p = attrs.length == 0 ? 0 : (StatUtils.max(attrs) + 1);
				GLM glm = new GLM(numClasses, p);

				for (int k = 0; k < numClasses; k++) {
					// One-vs-the-rest
					for (int i = 0; i < y.length; i++) {
						int label = (int) sd.y[i];
						y[i] = label == k ? 1 : 0;
					}

					GLM binaryClassifier = buildBinaryClassifier(attrs, indices, values, y, maxNumIters, lambda);

					double[] w = binaryClassifier.w[0];
					for (int j = 0; j < cList.length; j++) {
						int attIndex = attrs[j];
						glm.w[k][attIndex] = w[attIndex] * cList[j];
					}
					glm.intercept[k] = binaryClassifier.intercept[0];
				}

				return glm;
			}
		} else {
			DenseDataset dd = getDenseDataset(trainSet, true);
			int[] attrs = dd.attrs;
			double[][] x = dd.x;
			double[] y = new double[dd.y.length];
			double[] cList = dd.cList;

			if (numClasses == 2) {
				for (int i = 0; i < y.length; i++) {
					int label = (int) dd.y[i];
					y[i] = label == 0 ? 1 : 0;
				}

				GLM glm = buildBinaryClassifier(attrs, x, y, maxNumIters, lambda);

				double[] w = glm.w[0];
				for (int j = 0; j < cList.length; j++) {
					int attIndex = attrs[j];
					w[attIndex] *= cList[j];
				}

				return glm;
			} else {
				int p = attrs.length == 0 ? 0 : attrs[attrs.length - 1] + 1;
				GLM glm = new GLM(numClasses, p);

				for (int k = 0; k < numClasses; k++) {
					// One-vs-the-rest
					for (int i = 0; i < y.length; i++) {
						int label = (int) dd.y[i];
						y[i] = label == k ? 1 : 0;
					}

					GLM binaryClassifier = buildBinaryClassifier(attrs, x, y, maxNumIters, lambda);

					double[] w = binaryClassifier.w[0];
					for (int j = 0; j < cList.length; j++) {
						int attIndex = attrs[j];
						glm.w[k][attIndex] = w[attIndex] * cList[j];
					}
					glm.intercept[k] = binaryClassifier.intercept[0];
				}

				return glm;
			}
		}
	}

	/**
	 * Builds an L2-regularized classifier.
	 * 
	 * @param trainSet the training set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @return an L2-regularized classifier.
	 */
	public GLM buildClassifier(Instances trainSet, int maxNumIters, double lambda) {
		return buildClassifier(trainSet, isSparse(trainSet), maxNumIters, lambda);
	}

	/**
	 * Builds L2-regularized classifiers for a sequence of regularization parameter lambdas.
	 * 
	 * @param trainSet the training set.
	 * @param isSparse <code>true</code> if the training set is treated as sparse.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambdas the lambdas array.
	 * @return L2-regularized binary classifiers.
	 */
	public GLM[] buildClassifiers(Instances trainSet, boolean isSparse, int maxNumIters, double[] lambdas) {
		Attribute classAttribute = trainSet.getTargetAttribute();
		if (classAttribute.getType() != Attribute.Type.NOMINAL) {
			throw new IllegalArgumentException("Class attribute must be nominal.");
		}
		NominalAttribute clazz = (NominalAttribute) classAttribute;
		int numClasses = clazz.getStates().length;

		if (isSparse) {
			SparseDataset sd = getSparseDataset(trainSet, true);
			int[] attrs = sd.attrs;
			int[][] indices = sd.indices;
			double[][] values = sd.values;
			double[] y = new double[sd.y.length];
			double[] cList = sd.cList;

			if (numClasses == 2) {
				for (int i = 0; i < y.length; i++) {
					int label = (int) sd.y[i];
					y[i] = label == 0 ? 1 : 0;
				}

				GLM[] glms = buildBinaryClassifiers(attrs, indices, values, y, maxNumIters, lambdas);

				for (GLM glm : glms) {
					double[] w = glm.w[0];
					for (int j = 0; j < cList.length; j++) {
						int attIndex = attrs[j];
						w[attIndex] *= cList[j];
					}
				}

				return glms;
			} else {
				int p = attrs.length == 0 ? 0 : (StatUtils.max(attrs) + 1);
				GLM[] glms = new GLM[lambdas.length];
				for (int i = 0; i < glms.length; i++) {
					glms[i] = new GLM(numClasses, p);
				}

				for (int k = 0; k < numClasses; k++) {
					// One-vs-the-rest
					for (int i = 0; i < y.length; i++) {
						int label = (int) sd.y[i];
						y[i] = label == k ? 1 : 0;
					}

					GLM[] binaryClassifiers = buildBinaryClassifiers(attrs, indices, values, y, maxNumIters, lambdas);

					for (int l = 0; l < glms.length; l++) {
						GLM binaryClassifier = binaryClassifiers[l];
						GLM glm = glms[l];
						double[] w = binaryClassifier.w[0];
						for (int j = 0; j < cList.length; j++) {
							int attIndex = attrs[j];
							glm.w[k][attIndex] = w[attIndex] * cList[j];
						}
						glm.intercept[k] = binaryClassifier.intercept[0];
					}

				}

				return glms;
			}
		} else {
			DenseDataset dd = getDenseDataset(trainSet, true);
			int[] attrs = dd.attrs;
			double[][] x = dd.x;
			double[] y = new double[dd.y.length];
			double[] cList = dd.cList;

			if (numClasses == 2) {
				for (int i = 0; i < y.length; i++) {
					int label = (int) dd.y[i];
					y[i] = label == 0 ? 1 : 0;
				}

				GLM[] glms = buildBinaryClassifiers(attrs, x, y, maxNumIters, lambdas);

				for (GLM glm : glms) {
					double[] w = glm.w[0];
					for (int j = 0; j < cList.length; j++) {
						int attIndex = attrs[j];
						w[attIndex] *= cList[j];
					}
				}

				return glms;
			} else {
				int p = attrs.length == 0 ? 0 : attrs[attrs.length - 1] + 1;
				GLM[] glms = new GLM[lambdas.length];
				for (int i = 0; i < glms.length; i++) {
					glms[i] = new GLM(numClasses, p);
				}

				for (int k = 0; k < numClasses; k++) {
					// One-vs-the-rest
					for (int i = 0; i < y.length; i++) {
						int label = (int) dd.y[i];
						y[i] = label == k ? 1 : 0;
					}

					GLM[] binaryClassifiers = buildBinaryClassifiers(attrs, x, y, maxNumIters, lambdas);

					for (int l = 0; l < glms.length; l++) {
						GLM binaryClassifier = binaryClassifiers[l];
						GLM glm = glms[l];
						double[] w = binaryClassifier.w[0];
						for (int j = 0; j < cList.length; j++) {
							int attIndex = attrs[j];
							glm.w[k][attIndex] = w[attIndex] * cList[j];
						}
						glm.intercept[k] = binaryClassifier.intercept[0];
					}
				}

				return glms;
			}
		}
	}

	/**
	 * Builds L2-regularized classifiers for a sequence of regularization parameter lambdas.
	 * 
	 * @param trainSet the training set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambdas the lambdas array.
	 * @return L2-regularized binary classifiers.
	 */
	public GLM[] buildClassifiers(Instances trainSet, int maxNumIters, double[] lambdas) {
		return buildClassifiers(trainSet, isSparse(trainSet), maxNumIters, lambdas);
	}

	/**
	 * Builds an L2 regressor.
	 * 
	 * @param trainSet the training set.
	 * @param isSparse <code>true</code> if the training set is treated as sparse.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @return an L2-regularized regressor.
	 */
	public GLM buildRegressor(Instances trainSet, boolean isSparse, int maxNumIters, double lambda) {
		if (isSparse) {
			SparseDataset sd = getSparseDataset(trainSet, true);
			double[] cList = sd.cList;

			GLM glm = buildRegressor(sd.attrs, sd.indices, sd.values, sd.y, maxNumIters, lambda);

			double[] w = glm.w[0];
			for (int j = 0; j < cList.length; j++) {
				int attIndex = sd.attrs[j];
				w[attIndex] *= cList[j];
			}

			return glm;
		} else {
			DenseDataset dd = getDenseDataset(trainSet, true);
			double[] cList = dd.cList;

			GLM glm = buildRegressor(dd.attrs, dd.x, dd.y, maxNumIters, lambda);

			double[] w = glm.w[0];
			for (int j = 0; j < cList.length; j++) {
				int attIndex = dd.attrs[j];
				w[attIndex] *= cList[j];
			}
			return glm;
		}
	}

	/**
	 * Builds an L2-regularized regressor.
	 * 
	 * @param trainSet the training set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @return an L2-regularized regressor.
	 */
	public GLM buildRegressor(Instances trainSet, int maxNumIters, double lambda) {
		return buildRegressor(trainSet, isSparse(trainSet), maxNumIters, lambda);
	}

	/**
	 * Builds an L2-regularized regressor. Each row in the input matrix x represents a feature (instead of a data
	 * point). Thus the input matrix is the transpose of the row-oriented data matrix. This procedure does not assume
	 * the data is normalized or centered.
	 * 
	 * @param attrs the attribute list.
	 * @param x the inputs.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @return an L2-regularized regressor.
	 */
	public GLM buildRegressor(int[] attrs, double[][] x, double[] y, int maxNumIters, double lambda) {
		double[] w = new double[attrs.length];
		double intercept = 0;

		// Initialize residuals
		double[] rTrain = new double[y.length];
		for (int i = 0; i < rTrain.length; i++) {
			rTrain[i] = y[i];
		}

		// Calculate sum of squares
		double[] sq = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			sq[i] = StatUtils.sumSq(x[i]);
		}

		// Coordinate descent
		final double tl2 = lambda * y.length;
		for (int iter = 0; iter < maxNumIters; iter++) {
			double prevLoss = GLMOptimUtils.computeRidgeLoss(rTrain, w, lambda);
			
			if (fitIntercept) {
				intercept += OptimUtils.fitIntercept(rTrain);
			}

			doOnePass(x, sq, tl2, w, rTrain);

			double currLoss = GLMOptimUtils.computeRidgeLoss(rTrain, w, lambda);

			if (verbose) {
				System.out.println("Iteration " + iter + ": " + " " + currLoss);
			}

			if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
				break;
			}
		}

		return GLMOptimUtils.getGLM(attrs, w, intercept);
	}

	/**
	 * Builds an L2-regularized regressor on sparse inputs. Each row of the input represents a feature (instead of a
	 * data point), i.e., in column-oriented format. This procedure does not assume the data is normalized or centered.
	 * 
	 * @param attrs the attribute list.
	 * @param indices the indices.
	 * @param values the values.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @return an L2-regularized regressor.
	 */
	public GLM buildRegressor(int[] attrs, int[][] indices, double[][] values, double[] y, int maxNumIters,
			double lambda) {
		double[] w = new double[attrs.length];
		double intercept = 0;

		// Initialize residuals
		double[] rTrain = new double[y.length];
		for (int i = 0; i < rTrain.length; i++) {
			rTrain[i] = y[i];
		}

		// Calculate sum of squares
		double[] sq = new double[attrs.length];
		for (int i = 0; i < values.length; i++) {
			sq[i] = StatUtils.sumSq(values[i]);
		}

		// Coordinate descent
		final double tl2 = lambda * y.length;
		for (int iter = 0; iter < maxNumIters; iter++) {
			double prevLoss = GLMOptimUtils.computeRidgeLoss(rTrain, w, lambda);
			
			if (fitIntercept) {
				intercept += OptimUtils.fitIntercept(rTrain);
			}

			doOnePass(indices, values, sq, tl2, w, rTrain);

			double currLoss = GLMOptimUtils.computeRidgeLoss(rTrain, w, lambda);

			if (verbose) {
				System.out.println("Iteration " + iter + ": " + " " + currLoss);
			}

			if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
				break;
			}
		}

		return GLMOptimUtils.getGLM(attrs, w, intercept);
	}

	/**
	 * Builds L2-regularized regressors for a sequence of regularization parameter lambdas.
	 * 
	 * @param trainSet the training set.
	 * @param isSparse <code>true</code> if the training set is treated as sparse.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambdas the lambdas array.
	 * @return L2-regularized regressors.
	 */
	public GLM[] buildRegressors(Instances trainSet, boolean isSparse, int maxNumIters, double[] lambdas) {
		if (isSparse) {
			SparseDataset sd = getSparseDataset(trainSet, true);
			double[] cList = sd.cList;

			GLM[] glms = buildRegressors(sd.attrs, sd.indices, sd.values, sd.y, maxNumIters, lambdas);

			for (GLM glm : glms) {
				double[] w = glm.w[0];
				for (int j = 0; j < cList.length; j++) {
					int attIndex = sd.attrs[j];
					w[attIndex] *= cList[j];
				}
			}

			return glms;
		} else {
			DenseDataset dd = getDenseDataset(trainSet, true);
			double[] cList = dd.cList;

			GLM[] glms = buildRegressors(dd.attrs, dd.x, dd.y, maxNumIters, lambdas);

			for (GLM glm : glms) {
				double[] w = glm.w[0];
				for (int j = 0; j < cList.length; j++) {
					int attIndex = dd.attrs[j];
					w[attIndex] *= cList[j];
				}
			}

			return glms;
		}
	}

	/**
	 * Builds L2-regularized regressors for a sequence of regularization parameter lambdas.
	 * 
	 * @param trainSet the training set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambdas the lambdas array.
	 * @return L2-regularized regressors.
	 */
	public GLM[] buildRegressors(Instances trainSet, int maxNumIters, double[] lambdas) {
		return buildRegressors(trainSet, isSparse(trainSet), maxNumIters, lambdas);
	}

	/**
	 * Builds L2-regularized regressors for a sequence of regularization parameter lambdas on dense inputs. Each row in
	 * the input matrix x represents a feature (instead of a data point). Thus the input matrix is the transpose of the
	 * row-oriented data matrix. This procedure does not assume the data is normalized or centered.
	 * 
	 * @param attrs the attribute list.
	 * @param x the inputs.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambdas the lambdas array.
	 * @return L2-regularized regressors.
	 */
	public GLM[] buildRegressors(int[] attrs, double[][] x, double[] y, int maxNumIters, double[] lambdas) {
		double[] w = new double[attrs.length];
		double intercept = 0;

		GLM[] glms = new GLM[lambdas.length];
		Arrays.sort(lambdas);

		// Backup targets
		double[] rTrain = new double[y.length];
		for (int i = 0; i < rTrain.length; i++) {
			rTrain[i] = y[i];
		}

		// Calculate sum of squares
		double[] sq = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			sq[i] = StatUtils.sumSq(x[i]);
		}

		// Compute the regularization path
		for (int g = 0; g < glms.length; g++) {
			double lambda = lambdas[g];

			// Coordinate descent
			final double tl2 = lambda * y.length;
			for (int iter = 0; iter < maxNumIters; iter++) {
				double prevLoss = GLMOptimUtils.computeRidgeLoss(rTrain, w, lambda);
				
				if (fitIntercept) {
					intercept += OptimUtils.fitIntercept(rTrain);
				}

				doOnePass(x, sq, tl2, w, rTrain);

				double currLoss = GLMOptimUtils.computeRidgeLoss(rTrain, w, lambda);
				
				if (verbose) {
					System.out.println("Iteration " + iter + ": " + " " + currLoss);
				}

				if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
					break;
				}
			}

			glms[g] = GLMOptimUtils.getGLM(attrs, w, intercept);
		}

		return glms;
	}

	/**
	 * Builds L2-regularized regressors for a sequence of regularization parameter lambdas on sparse inputs. Each row of
	 * the input represents a feature (instead of a data point), i.e., in column-oriented format. This procedure does
	 * not assume the data is normalized or centered.
	 * 
	 * @param attrs the attribute list.
	 * @param indices the indices.
	 * @param values the values.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambdas the lambdas array.
	 * @return L2-regularized regressors.
	 */
	public GLM[] buildRegressors(int[] attrs, int[][] indices, double[][] values, double[] y, int maxNumIters,
			double[] lambdas) {
		double[] w = new double[attrs.length];
		double intercept = 0;

		GLM[] glms = new GLM[lambdas.length];
		Arrays.sort(lambdas);

		// Backup targets
		double[] rTrain = new double[y.length];
		for (int i = 0; i < rTrain.length; i++) {
			rTrain[i] = y[i];
		}

		// Calculate sum of squares
		double[] sq = new double[attrs.length];
		for (int i = 0; i < values.length; i++) {
			sq[i] = StatUtils.sumSq(values[i]);
		}

		// Compute the regularization path
		for (int g = 0; g < glms.length; g++) {
			double lambda = lambdas[g];

			// Coordinate descent
			final double tl2 = lambda * y.length;
			for (int iter = 0; iter < maxNumIters; iter++) {
				double prevLoss = GLMOptimUtils.computeRidgeLoss(rTrain, w, lambda);
				
				if (fitIntercept) {
					intercept += OptimUtils.fitIntercept(rTrain);
				}

				doOnePass(indices, values, sq, tl2, w, rTrain);

				double currLoss = GLMOptimUtils.computeRidgeLoss(rTrain, w, lambda);
				
				if (verbose) {
					System.out.println("Iteration " + iter + ": " + " " + currLoss);
				}

				if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
					break;
				}
			}

			glms[g] = GLMOptimUtils.getGLM(attrs, w, intercept);
		}

		return glms;
	}

	protected void doOnePass(double[][] x, double[] sq, final double tl2, double[] w, double[] rTrain) {
		for (int j = 0; j < x.length; j++) {
			double[] v = x[j];
			// Calculate weight updates using naive updates
			double eta = VectorUtils.dotProduct(rTrain, v);
			double wNew = (w[j] * sq[j] + eta) / (sq[j] + tl2);

			double delta = wNew - w[j];
			w[j] = wNew;

			// Update residuals
			for (int i = 0; i < rTrain.length; i++) {
				rTrain[i] -= delta * v[i];
			}
		}
	}

	protected void doOnePass(double[][] x, double[] theta, double[] y, final double tl2, double[] w, double[] pTrain,
			double[] rTrain) {
		for (int j = 0; j < x.length; j++) {
			if (Math.abs(theta[j]) <= MathUtils.EPSILON) {
				continue;
			}

			double[] v = x[j];
			double eta = VectorUtils.dotProduct(rTrain, v);

			double newW = (w[j] * theta[j] + eta) / (theta[j] + tl2);
			double delta = newW - w[j];
			w[j] = newW;

			// Update predictions
			for (int i = 0; i < pTrain.length; i++) {
				pTrain[i] += delta * v[i];
				rTrain[i] = OptimUtils.getPseudoResidual(pTrain[i], y[i]);
			}
		}
	}

	protected void doOnePass(int[][] indices, double[][] values, double[] sq, final double tl2, double[] w,
			double[] rTrain) {
		for (int j = 0; j < indices.length; j++) {
			// Calculate weight updates using naive updates
			double wNew = w[j] * sq[j];
			int[] index = indices[j];
			double[] value = values[j];
			for (int i = 0; i < index.length; i++) {
				wNew += rTrain[index[i]] * value[i];
			}
			wNew /= (sq[j] + tl2);

			double delta = wNew - w[j];
			w[j] = wNew;

			// Update residuals
			for (int i = 0; i < index.length; i++) {
				rTrain[index[i]] -= delta * value[i];
			}
		}
	}

	protected void doOnePass(int[][] indices, double[][] values, double[] theta, double[] y, final double tl2, double[] w,
			double[] pTrain, double[] rTrain) {
		for (int j = 0; j < indices.length; j++) {
			if (Math.abs(theta[j]) <= MathUtils.EPSILON) {
				continue;
			}

			double eta = 0;
			int[] index = indices[j];
			double[] value = values[j];
			for (int i = 0; i < index.length; i++) {
				int idx = index[i];
				eta += rTrain[idx] * value[i];
			}

			double newW = (w[j] * theta[j] + eta) / (theta[j] + tl2);
			double delta = newW - w[j];
			w[j] = newW;

			// Update predictions
			for (int i = 0; i < index.length; i++) {
				int idx = index[i];
				pTrain[idx] += delta * value[i];
				rTrain[idx] = OptimUtils.getPseudoResidual(pTrain[idx], y[idx]);
			}
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

}
