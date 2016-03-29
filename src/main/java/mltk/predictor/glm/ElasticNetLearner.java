package mltk.predictor.glm;

import java.util.Arrays;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.cmdline.options.LearnerWithTaskOptions;
import mltk.core.Attribute;
import mltk.core.DenseVector;
import mltk.core.Instances;
import mltk.core.NominalAttribute;
import mltk.core.SparseVector;
import mltk.core.io.InstancesReader;
import mltk.predictor.Family;
import mltk.predictor.LinkFunction;
import mltk.predictor.io.PredictorWriter;
import mltk.util.MathUtils;
import mltk.util.OptimUtils;
import mltk.util.StatUtils;
import mltk.util.VectorUtils;

/**
 * Class for learning elastic-net penalized linear model via coordinate descent.
 * 
 * @author Yin Lou
 * 
 */
public class ElasticNetLearner extends GLMLearner {

	static class Options extends LearnerWithTaskOptions {

		@Argument(name = "-m", description = "maximum number of iterations (default: 0)")
		int maxIter = 0;

		@Argument(name = "-l", description = "lambda (default: 0)")
		double lambda = 0;

		@Argument(name = "-a", description = "L1 ratio (default: 0)")
		double l1Ratio = 0;

	}

	/**
	 * <p>
	 * 
	 * <pre>
	 * Usage: mltk.predictor.glm.ElasticNetLearner
	 * -t	train set path
	 * [-g]	task between classification (c) and regression (r) (default: r)
	 * [-r]	attribute file path
	 * [-o]	output model path
	 * [-V]	verbose (default: true)
	 * [-m]	maximum number of iterations (default: 0)
	 * [-l]	lambda (default: 0)
	 * [-a]	L1 ratio (default: 0)
	 * </pre>
	 * 
	 * </p>
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(ElasticNetLearner.class, opts);
		Task task = null;
		try {
			parser.parse(args);
			task = Task.get(opts.task);
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}
		Instances trainSet = InstancesReader.read(opts.attPath, opts.trainPath);

		ElasticNetLearner learner = new ElasticNetLearner();
		learner.setVerbose(opts.verbose);
		learner.setTask(task);
		learner.setLambda(opts.lambda);
		learner.setL1Ratio(opts.l1Ratio);
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
	private double l1Ratio;
	private Task task;

	/**
	 * Constructor.
	 */
	public ElasticNetLearner() {
		verbose = false;
		fitIntercept = true;
		maxNumIters = -1;
		epsilon = MathUtils.EPSILON;
		lambda = 0; // no regularization
		l1Ratio = 0; // 0: ridge, 1: lasso, (0, 1): elastic net
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
				glm = buildGaussianRegressor(instances, maxNumIters, lambda, l1Ratio);
				break;
			case CLASSIFICATION:
				glm = buildClassifier(instances, maxNumIters, lambda, l1Ratio);
				break;
			default:
				break;
		}
		return glm;
	}
	
	@Override
	public GLM build(Instances trainSet, Family family) {
		GLM glm = null;
		if (maxNumIters < 0) {
			maxNumIters = 20;
		}
		switch (family) {
			case GAUSSIAN:
				glm = buildGaussianRegressor(trainSet, maxNumIters, lambda, l1Ratio);
				break;
			case BINOMIAL:
				glm = buildClassifier(trainSet, maxNumIters, lambda, l1Ratio);
				break;
			default:
				throw new IllegalArgumentException("Unsupported family: " + family);
		}
		return glm;
	}

	/**
	 * Builds an elastic-net penalized binary classifier. Each row in the input matrix x represents a feature (instead
	 * of a data point). Thus the input matrix is the transpose of the row-oriented data matrix. This procedure does not
	 * assume the data is normalized or centered.
	 * 
	 * @param attrs the attribute list.
	 * @param x the inputs.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @param l1Ratio the L1 ratio.
	 * @return an elastic-net penalized binary classifier.
	 */
	public GLM buildBinaryClassifier(int[] attrs, double[][] x, double[] y, int maxNumIters, double lambda, double l1Ratio) {
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

		final double lambda1 = lambda * l1Ratio;
		final double lambda2 = lambda * (1 - l1Ratio);
		final double tl1 = lambda1 * y.length;
		final double tl2 = lambda2 * y.length;

		// Coordinate descent
		for (int iter = 0; iter < maxNumIters; iter++) {
			double prevLoss = GLMOptimUtils.computeElasticNetLoss(pTrain, y, w, lambda1, lambda2);
			
			if (fitIntercept) {
				intercept += OptimUtils.fitIntercept(pTrain, rTrain, y);
			}

			doOnePassBinomial(x, theta, y, tl1, tl2, w, pTrain, rTrain);

			double currLoss = GLMOptimUtils.computeElasticNetLoss(pTrain, y, w, lambda1, lambda2);

			if (verbose) {
				System.out.println("Iteration " + iter + ": " + " " + currLoss);
			}

			if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
				break;
			}
		}

		return GLMOptimUtils.getGLM(attrs, w, intercept, LinkFunction.LOGIT);
	}

	/**
	 * Builds an elastic-net penalized binary classifier on sparse inputs. Each row of the input represents a feature
	 * (instead of a data point), i.e., in column-oriented format. This procedure does not assume the data is normalized
	 * or centered.
	 * 
	 * @param attrs the attribute list.
	 * @param indices the indices.
	 * @param values the values.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @param l1Ratio the L1 ratio.
	 * @return an elastic-net penalized classifier.
	 */
	public GLM buildBinaryClassifier(int[] attrs, int[][] indices, double[][] values, double[] y, int maxNumIters,
			double lambda, double l1Ratio) {
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

		final double lambda1 = lambda * l1Ratio;
		final double lambda2 = lambda * (1 - l1Ratio);
		final double tl1 = lambda1 * y.length;
		final double tl2 = lambda2 * y.length;

		// Coordinate descent
		for (int iter = 0; iter < maxNumIters; iter++) {
			double prevLoss = GLMOptimUtils.computeElasticNetLoss(pTrain, y, w, lambda1, lambda2);
			
			if (fitIntercept) {
				intercept += OptimUtils.fitIntercept(pTrain, rTrain, y);
			}

			doOnePassBinomial(indices, values, theta, y, tl1, tl2, w, pTrain, rTrain);

			double currLoss = GLMOptimUtils.computeElasticNetLoss(pTrain, y, w, lambda1, lambda2);

			if (verbose) {
				System.out.println("Iteration " + iter + ": " + " " + currLoss);
			}

			if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
				break;
			}
		}

		return GLMOptimUtils.getGLM(attrs, w, intercept, LinkFunction.LOGIT);
	}

	/**
	 * Builds elastic-net penalized binary classifiers for a sequence of regularization parameter lambdas. Each row in
	 * the input matrix x represents a feature (instead of a data point). Thus the input matrix is the transpose of the
	 * row-oriented data matrix. This procedure does not assume the data is normalized or centered.
	 * 
	 * @param attrs the attribute list.
	 * @param x the inputs.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numLambdas the number of lambdas.
	 * @param minLambdaRatio the minimum lambda is minLambdaRatio * max lambda.
	 * @param l1Ratio the L1 ratio.
	 * @return elastic-net penalized classifiers.
	 */
	public GLM[] buildBinaryClassifiers(int[] attrs, double[][] x, double[] y, int maxNumIters, int numLambdas,
			double minLambdaRatio, double l1Ratio) {
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

		double maxLambda = findMaxLambdaBinomial(x, y, pTrain, rTrain, l1Ratio);

		// Dampening factor for lambda
		double alpha = Math.pow(minLambdaRatio, 1.0 / numLambdas);

		GLM[] glms = new GLM[numLambdas];
		double lambda = maxLambda;
		for (int g = 0; g < numLambdas; g++) {
			final double lambda1 = lambda * l1Ratio;
			final double lambda2 = lambda * (1 - l1Ratio);
			final double tl1 = lambda1 * y.length;
			final double tl2 = lambda2 * y.length;

			// Coordinate descent
			for (int iter = 0; iter < maxNumIters; iter++) {
				double prevLoss = GLMOptimUtils.computeElasticNetLoss(pTrain, y, w, lambda1, lambda2);
				
				if (fitIntercept) {
					intercept += OptimUtils.fitIntercept(pTrain, rTrain, y);
				}

				doOnePassBinomial(x, theta, y, tl1, tl2, w, pTrain, rTrain);

				double currLoss = GLMOptimUtils.computeElasticNetLoss(pTrain, y, w, lambda1, lambda2);

				if (verbose) {
					System.out.println("Iteration " + iter + ": " + " " + currLoss);
				}

				if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
					break;
				}
			}

			glms[g] = GLMOptimUtils.getGLM(attrs, w, intercept, LinkFunction.LOGIT);
			lambda *= alpha;
		}

		return glms;
	}

	/**
	 * Builds elastic-net penalized binary classifiers on sparse inputs for a sequence of regularization parameter
	 * lambdas. Each row of the input represents a feature (instead of a data point), i.e., in column-oriented format.
	 * This procedure does not assume the data is normalized or centered.
	 * 
	 * @param attrs the attribute list.
	 * @param indices the indices.
	 * @param values the values.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numLambdas the number of lambdas.
	 * @param minLambdaRatio the minimum lambda is minLambdaRatio * max lambda.
	 * @param l1Ratio the L1 ratio.
	 * @return an elastic-net penalized classifier.
	 */
	public GLM[] buildBinaryClassifiers(int[] attrs, int[][] indices, double[][] values, double[] y, int maxNumIters,
			int numLambdas, double minLambdaRatio, double l1Ratio) {
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

		double maxLambda = findMaxLambdaBinomial(indices, values, y, pTrain, rTrain, l1Ratio);

		// Dampening factor for lambda
		double alpha = Math.pow(minLambdaRatio, 1.0 / numLambdas);

		GLM[] glms = new GLM[numLambdas];
		double lambda = maxLambda;
		for (int g = 0; g < glms.length; g++) {
			final double lambda1 = lambda * l1Ratio;
			final double lambda2 = lambda * (1 - l1Ratio);
			final double tl1 = lambda1 * y.length;
			final double tl2 = lambda2 * y.length;

			// Coordinate descent
			for (int iter = 0; iter < maxNumIters; iter++) {
				double prevLoss = GLMOptimUtils.computeElasticNetLoss(pTrain, y, w, lambda1, lambda2);
				
				if (fitIntercept) {
					intercept += OptimUtils.fitIntercept(pTrain, rTrain, y);
				}

				doOnePassBinomial(indices, values, theta, y, tl1, tl2, w, pTrain, rTrain);

				double currLoss = GLMOptimUtils.computeElasticNetLoss(pTrain, y, w, lambda1, lambda2);

				if (verbose) {
					System.out.println("Iteration " + iter + ": " + " " + currLoss);
				}

				if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
					break;
				}
			}

			glms[g] = GLMOptimUtils.getGLM(attrs, w, intercept, LinkFunction.LOGIT);
			lambda *= alpha;
		}

		return glms;
	}

	/**
	 * Builds an elastic-net penalized classifier.
	 * 
	 * @param trainSet the training set.
	 * @param isSparse <code>true</code> if the training set is treated as sparse.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @param l1Ratio the l1 ratio.
	 * @return an elastic-net penalized classifer.
	 */
	public GLM buildClassifier(Instances trainSet, boolean isSparse, int maxNumIters, double lambda, double l1Ratio) {
		Attribute classAttribute = trainSet.getTargetAttribute();
		if (classAttribute.getType() != Attribute.Type.NOMINAL) {
			throw new IllegalArgumentException("Class attribute must be nominal.");
		}
		NominalAttribute clazz = (NominalAttribute) classAttribute;
		int numClasses = clazz.getCardinality();

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

				GLM glm = buildBinaryClassifier(attrs, indices, values, y, maxNumIters, lambda, l1Ratio);

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
						int label = (int) sd.y[i];
						y[i] = label == k ? 1 : 0;
					}

					GLM binaryClassifier = buildBinaryClassifier(attrs, indices, values, y, maxNumIters, lambda,
							l1Ratio);

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

				GLM glm = buildBinaryClassifier(attrs, x, y, maxNumIters, lambda, l1Ratio);

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
						int label = (int) dd.y[i];
						y[i] = label == k ? 1 : 0;
					}

					GLM binaryClassifier = buildBinaryClassifier(attrs, x, y, maxNumIters, lambda, l1Ratio);

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
	 * Builds an elastic-net penalized classifier.
	 * 
	 * @param trainSet the training set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @param l1Ratio the l1 ratio.
	 * @return an elastic-net penalized classifer.
	 */
	public GLM buildClassifier(Instances trainSet, int maxNumIters, double lambda, double l1Ratio) {
		return buildClassifier(trainSet, isSparse(trainSet), maxNumIters, lambda, l1Ratio);
	}

	/**
	 * Builds elastic-net penalized classifiers.
	 * 
	 * @param trainSet the training set.
	 * @param isSparse <code>true</code> if the training set is treated as sparse.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numLambdas the number of lambdas.
	 * @param minLambdaRatio the minimum lambda is minLambdaRatio * max lambda.
	 * @param l1Ratio the l1 ratio.
	 * @return elastic-net penalized classifers.
	 */
	public GLM[] buildClassifiers(Instances trainSet, boolean isSparse, int maxNumIters, int numLambdas,
			double minLambdaRatio, double l1Ratio) {
		Attribute classAttribute = trainSet.getTargetAttribute();
		if (classAttribute.getType() != Attribute.Type.NOMINAL) {
			throw new IllegalArgumentException("Class attribute must be nominal.");
		}
		NominalAttribute clazz = (NominalAttribute) classAttribute;
		int numClasses = clazz.getCardinality();

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

				GLM[] glms = buildBinaryClassifiers(attrs, indices, values, y, maxNumIters, numLambdas, minLambdaRatio,
						l1Ratio);

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
				GLM[] glms = new GLM[numLambdas];
				for (int i = 0; i < glms.length; i++) {
					glms[i] = new GLM(numClasses, p);
				}

				for (int k = 0; k < numClasses; k++) {
					// One-vs-the-rest
					for (int i = 0; i < y.length; i++) {
						int label = (int) sd.y[i];
						y[i] = label == k ? 1 : 0;
					}

					GLM[] binaryClassifiers = buildBinaryClassifiers(attrs, indices, values, y, maxNumIters,
							numLambdas, minLambdaRatio, l1Ratio);

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

				GLM[] glms = buildBinaryClassifiers(attrs, x, y, maxNumIters, numLambdas, minLambdaRatio, l1Ratio);

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
				GLM[] glms = new GLM[numLambdas];
				for (int i = 0; i < glms.length; i++) {
					glms[i] = new GLM(numClasses, p);
				}

				for (int k = 0; k < numClasses; k++) {
					// One-vs-the-rest
					for (int i = 0; i < y.length; i++) {
						int label = (int) dd.y[i];
						y[i] = label == k ? 1 : 0;
					}

					GLM[] binaryClassifiers = buildBinaryClassifiers(attrs, x, y, maxNumIters, numLambdas,
							minLambdaRatio, l1Ratio);

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
	 * Builds elastic-net penalized classifiers.
	 * 
	 * @param trainSet the training set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numLambdas the number of lambdas.
	 * @param minLambdaRatio the minimum lambda is minLambdaRatio * max lambda.
	 * @param l1Ratio the l1 ratio.
	 * @return elastic-net penalized classifers.
	 */
	public GLM[] buildClassifiers(Instances trainSet, int maxNumIters, int numLambdas, double minLambdaRatio,
			double l1Ratio) {
		return buildClassifiers(trainSet, isSparse(trainSet), maxNumIters, numLambdas, minLambdaRatio, l1Ratio);
	}

	/**
	 * Builds an elastic-net penalized regressor.
	 * 
	 * @param trainSet the training set.
	 * @param isSparse <code>true</code> if the training set is treated as sparse.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @param l1Ratio the L1 ratio.
	 * @return an elastic-net penalized regressor.
	 */
	public GLM buildGaussianRegressor(Instances trainSet, boolean isSparse, int maxNumIters, double lambda, double l1Ratio) {
		if (isSparse) {
			SparseDataset sd = getSparseDataset(trainSet, true);
			double[] cList = sd.cList;

			GLM glm = buildGaussianRegressor(sd.attrs, sd.indices, sd.values, sd.y, maxNumIters, lambda, l1Ratio);

			double[] w = glm.w[0];
			for (int j = 0; j < cList.length; j++) {
				int attIndex = sd.attrs[j];
				w[attIndex] *= cList[j];
			}

			return glm;
		} else {
			DenseDataset dd = getDenseDataset(trainSet, true);
			double[] cList = dd.cList;

			GLM glm = buildGaussianRegressor(dd.attrs, dd.x, dd.y, maxNumIters, lambda, l1Ratio);

			double[] w = glm.w[0];
			for (int j = 0; j < cList.length; j++) {
				int attIndex = dd.attrs[j];
				w[attIndex] *= cList[j];
			}
			return glm;
		}
	}

	/**
	 * Builds an elastic-net penalized regressor.
	 * 
	 * @param trainSet the training set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @param l1Ratio the L1 ratio.
	 * @return an elastic-net penalized regressor.
	 */
	public GLM buildGaussianRegressor(Instances trainSet, int maxNumIters, double lambda, double l1Ratio) {
		return buildGaussianRegressor(trainSet, isSparse(trainSet), maxNumIters, lambda, l1Ratio);
	}

	/**
	 * Builds an elastic-net penalized regressor. Each row in the input matrix x represents a feature (instead of a data
	 * point). Thus the input matrix is the transpose of the row-oriented data matrix. This procedure does not assume
	 * the data is normalized or centered.
	 * 
	 * @param attrs the attribute list.
	 * @param x the inputs.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @param l1Ratio the L1 ratio.
	 * @return an elastic-net penalized regressor.
	 */
	public GLM buildGaussianRegressor(int[] attrs, double[][] x, double[] y, int maxNumIters, double lambda, double l1Ratio) {
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

		final double lambda1 = lambda * l1Ratio;
		final double lambda2 = lambda * (1 - l1Ratio);
		final double tl1 = lambda1 * y.length;
		final double tl2 = lambda2 * y.length;

		// Coordinate descent
		for (int iter = 0; iter < maxNumIters; iter++) {
			double prevLoss = GLMOptimUtils.computeElasticNetLoss(rTrain, w, lambda1, lambda2);
			
			if (fitIntercept) {
				intercept += OptimUtils.fitIntercept(rTrain);
			}

			doOnePassGaussian(x, sq, tl1, tl2, w, rTrain);

			double currLoss = GLMOptimUtils.computeElasticNetLoss(rTrain, w, lambda1, lambda2);

			if (verbose) {
				System.out.println("Iteration " + iter + ": " + " " + currLoss);
			}

			if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
				break;
			}
		}

		return GLMOptimUtils.getGLM(attrs, w, intercept, LinkFunction.IDENTITY);
	}

	/**
	 * Builds an elastic-net penalized regressor on sparse inputs. Each row of the input represents a feature (instead
	 * of a data point), i.e., in column-oriented format. This procedure does not assume the data is normalized or
	 * centered.
	 * 
	 * @param attrs the attribute list.
	 * @param indices the indices.
	 * @param values the values.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @param l1Ratio the L1 ratio.
	 * @return an elastic-net penalized regressor.
	 */
	public GLM buildGaussianRegressor(int[] attrs, int[][] indices, double[][] values, double[] y, int maxNumIters,
			double lambda, double l1Ratio) {
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

		final double lambda1 = lambda * l1Ratio;
		final double lambda2 = lambda * (1 - l1Ratio);
		final double tl1 = lambda1 * y.length;
		final double tl2 = lambda2 * y.length;

		// Coordinate descent
		for (int iter = 0; iter < maxNumIters; iter++) {
			double prevLoss = GLMOptimUtils.computeElasticNetLoss(rTrain, w, lambda1, lambda2);
			
			if (fitIntercept) {
				intercept += OptimUtils.fitIntercept(rTrain);
			}

			doOnePassGaussian(indices, values, sq, tl1, tl2, w, rTrain);

			double currLoss = GLMOptimUtils.computeElasticNetLoss(rTrain, w, lambda1, lambda2);

			if (verbose) {
				System.out.println("Iteration " + iter + ": " + " " + currLoss);
			}

			if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
				break;
			}
		}

		return GLMOptimUtils.getGLM(attrs, w, intercept, LinkFunction.IDENTITY);
	}

	/**
	 * Builds elastic-net penalized regressors for a sequence of regularization parameter lambdas.
	 * 
	 * @param trainSet the training set.
	 * @param isSparse <code>true</code> if the training set is treated as sparse.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numLambdas the number of lambdas.
	 * @param minLambdaRatio the minimum lambda is minLambdaRatio * max lambda.
	 * @param l1Ratio the L1 ratio.
	 * @return elastic-net penalized regressors.
	 */
	public GLM[] buildGaussianRegressors(Instances trainSet, boolean isSparse, int maxNumIters, int numLambdas,
			double minLambdaRatio, double l1Ratio) {
		if (isSparse) {
			SparseDataset sd = getSparseDataset(trainSet, true);
			double[] cList = sd.cList;

			GLM[] glms = buildGaussianRegressors(sd.attrs, sd.indices, sd.values, sd.y, maxNumIters, numLambdas,
					minLambdaRatio, l1Ratio);

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

			GLM[] glms = buildGaussianRegressors(dd.attrs, dd.x, dd.y, maxNumIters, numLambdas, minLambdaRatio, l1Ratio);

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
	 * Builds elastic-net penalized regressors for a sequence of regularization parameter lambdas.
	 * 
	 * @param trainSet the training set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numLambdas the number of lambdas.
	 * @param minLambdaRatio the minimum lambda is minLambdaRatio * max lambda.
	 * @param l1Ratio the L1 ratio.
	 * @return elastic-net penalized regressors.
	 */
	public GLM[] buildGaussianRegressors(Instances trainSet, int maxNumIters, int numLambdas, double minLambdaRatio,
			double l1Ratio) {
		return buildGaussianRegressors(trainSet, isSparse(trainSet), maxNumIters, numLambdas, minLambdaRatio, l1Ratio);
	}

	/**
	 * Builds elastic-net penalized regressors for a sequence of regularization parameter lambdas. Each row in the input
	 * matrix x represents a feature (instead of a data point). Thus the input matrix is the transpose of the
	 * row-oriented data matrix. This procedure does not assume the data is normalized or centered.
	 * 
	 * @param attrs the attribute list.
	 * @param x the inputs.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numLambdas the number of lambdas.
	 * @param minLambdaRatio the minimum lambda is minLambdaRatio * max lambda.
	 * @param l1Ratio the L1 ratio.
	 * @return elastic-net penalized regressors.
	 */
	public GLM[] buildGaussianRegressors(int[] attrs, double[][] x, double[] y, int maxNumIters, int numLambdas,
			double minLambdaRatio, double l1Ratio) {
		double[] w = new double[attrs.length];
		double intercept = 0;

		GLM[] glms = new GLM[numLambdas];

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

		// Determine max lambda
		double maxLambda = findMaxLambdaGaussian(x, y, l1Ratio);

		// Dampening factor for lambda
		double alpha = Math.pow(minLambdaRatio, 1.0 / numLambdas);

		// Compute the regularization path
		double lambda = maxLambda;
		for (int g = 0; g < glms.length; g++) {

			final double lambda1 = lambda * l1Ratio;
			final double lambda2 = lambda * (1 - l1Ratio);
			final double tl1 = lambda1 * y.length;
			final double tl2 = lambda2 * y.length;

			// Coordinate descent
			for (int iter = 0; iter < maxNumIters; iter++) {
				double prevLoss = GLMOptimUtils.computeElasticNetLoss(rTrain, w, lambda1, lambda2);
				
				if (fitIntercept) {
					intercept += OptimUtils.fitIntercept(rTrain);
				}

				doOnePassGaussian(x, sq, tl1, tl2, w, rTrain);

				double currLoss = GLMOptimUtils.computeElasticNetLoss(rTrain, w, lambda1, lambda2);

				if (verbose) {
					System.out.println("Iteration " + iter + ": " + currLoss);
				}

				if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
					break;
				}
			}

			lambda *= alpha;
			glms[g] = GLMOptimUtils.getGLM(attrs, w, intercept, LinkFunction.IDENTITY);
		}

		return glms;
	}

	/**
	 * Builds elastic-net penalized regressors on sparse inputs for a sequence of regularization parameter lambdas. Each row of the input
	 * represents a feature (instead of a data point), i.e., in column-oriented format. This procedure does not assume
	 * the data is normalized or centered.
	 * 
	 * @param attrs the attribute list.
	 * @param indices the indices.
	 * @param values the values.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numLambdas the number of lambdas.
	 * @param minLambdaRatio the minimum lambda is minLambdaRatio * max lambda.
	 * @param l1Ratio the L1 ratio.
	 * @return elastic-net penalized regressors.
	 */
	public GLM[] buildGaussianRegressors(int[] attrs, int[][] indices, double[][] values, double[] y, int maxNumIters,
			int numLambdas, double minLambdaRatio, double l1Ratio) {
		double[] w = new double[attrs.length];
		double intercept = 0;

		GLM[] glms = new GLM[numLambdas];

		// Backup targets
		double[] rTrain = new double[y.length];
		for (int i = 0; i < rTrain.length; i++) {
			rTrain[i] = y[i];
		}

		// Calculate sum of squares
		double[] sq = new double[values.length];
		for (int i = 0; i < values.length; i++) {
			sq[i] = StatUtils.sumSq(values[i]);
		}

		// Determine max lambda
		double maxLambda = findMaxLambdaGaussian(indices, values, y, l1Ratio);

		// Dampening factor for lambda
		double alpha = Math.pow(minLambdaRatio, 1.0 / numLambdas);

		// Compute the regularization path
		double lambda = maxLambda;
		for (int g = 0; g < glms.length; g++) {

			final double lambda1 = lambda * l1Ratio;
			final double lambda2 = lambda * (1 - l1Ratio);
			final double tl1 = lambda1 * y.length;
			final double tl2 = lambda2 * y.length;

			// Coordinate descent
			for (int iter = 0; iter < maxNumIters; iter++) {
				double prevLoss = GLMOptimUtils.computeElasticNetLoss(rTrain, w, lambda1, lambda2);
				
				if (fitIntercept) {
					intercept += OptimUtils.fitIntercept(rTrain);
				}

				doOnePassGaussian(indices, values, sq, tl1, tl2, w, rTrain);

				double currLoss = GLMOptimUtils.computeElasticNetLoss(rTrain, w, lambda1, lambda2);

				if (verbose) {
					System.out.println("Iteration " + iter + ": " + currLoss);
				}

				if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
					break;
				}
			}
			
			lambda *= alpha;
			glms[g] = GLMOptimUtils.getGLM(attrs, w, intercept, LinkFunction.IDENTITY);
		}

		return glms;
	}

	protected void doOnePassGaussian(double[][] x, double[] sq, final double tl1, final double tl2, double[] w, double[] rTrain) {
		for (int j = 0; j < x.length; j++) {
			double[] v = x[j];

			// Calculate weight updates using naive updates
			double wNew = w[j] * sq[j] + VectorUtils.dotProduct(v, rTrain);
			if (Math.abs(wNew) <= tl1) {
				wNew = 0;
			} else if (wNew > 0) {
				wNew -= tl1;
			} else {
				wNew += tl1;
			}
			wNew /= (sq[j] + tl2);

			double delta = wNew - w[j];
			w[j] = wNew;

			// Update residuals
			for (int i = 0; i < rTrain.length; i++) {
				rTrain[i] -= delta * v[i];
			}
		}
	}
	
	protected void doOnePassGaussian(int[][] indices, double[][] values, double[] sq, final double tl1, final double tl2,
			double[] w, double[] rTrain) {
		for (int j = 0; j < indices.length; j++) {
			// Calculate weight updates using naive updates
			double wNew = w[j] * sq[j];
			int[] index = indices[j];
			double[] value = values[j];
			for (int i = 0; i < index.length; i++) {
				wNew += rTrain[index[i]] * value[i];
			}

			if (Math.abs(wNew) <= tl1) {
				wNew = 0;
			} else if (wNew > 0) {
				wNew -= tl1;
			} else {
				wNew += tl1;
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

	protected void doOnePassBinomial(double[][] x, double[] theta, double[] y, final double tl1, final double tl2, double[] w,
			double[] pTrain, double[] rTrain) {
		for (int j = 0; j < x.length; j++) {
			if (Math.abs(theta[j]) <= MathUtils.EPSILON) {
				continue;
			}

			double[] v = x[j];
			double eta = VectorUtils.dotProduct(rTrain, v);

			double newW = w[j] * theta[j] + eta;
			if (newW > tl1) {
				newW -= tl1;
			} else if (newW < -tl1) {
				newW += tl1;
			} else {
				newW = 0;
			}
			newW /= (theta[j] + tl2);

			double delta = newW - w[j];
			w[j] = newW;

			// Update predictions
			for (int i = 0; i < pTrain.length; i++) {
				pTrain[i] += delta * v[i];
				rTrain[i] = OptimUtils.getPseudoResidual(pTrain[i], y[i]);
			}
		}
	}
	
	protected void doOnePassBinomial(int[][] indices, double[][] values, double[] theta, double[] y, final double tl1,
			final double tl2, double[] w, double[] pTrain, double[] rTrain) {
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

			double newW = w[j] * theta[j] + eta;
			if (newW > tl1) {
				newW -= tl1;
			} else if (newW < -tl1) {
				newW += tl1;
			} else {
				newW = 0;
			}
			newW /= (theta[j] + tl2);

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

	protected double findMaxLambdaGaussian(double[][] x, double[] y, double l1Ratio) {
		double mean = 0;
		if (fitIntercept) {
			mean = OptimUtils.fitIntercept(y);
		}
		// Determine max lambda
		double maxLambda = 0;
		for (double[] col : x) {
			double dot = Math.abs(VectorUtils.dotProduct(col, y));
			if (dot > maxLambda) {
				maxLambda = dot;
			}
		}
		maxLambda /= y.length;
		maxLambda /= l1Ratio;
		if (fitIntercept) {
			VectorUtils.add(y, mean);
		}
		return maxLambda;
	}
	
	protected double findMaxLambdaGaussian(int[][] indices, double[][] values, double[] y, double l1Ratio) {
		double mean = 0;
		if (fitIntercept) {
			mean = OptimUtils.fitIntercept(y);
		}

		DenseVector v = new DenseVector(y);
		// Determine max lambda
		double maxLambda = 0;
		for (int i = 0; i < indices.length; i++) {
			int[] index = indices[i];
			double[] value = values[i];
			double dot = Math.abs(VectorUtils.dotProduct(new SparseVector(index, value), v));
			if (dot > maxLambda) {
				maxLambda = dot;
			}
		}
		maxLambda /= y.length;
		maxLambda /= l1Ratio;
		if (fitIntercept) {
			VectorUtils.add(y, mean);
		}
		return maxLambda;
	}

	protected double findMaxLambdaBinomial(double[][] x, double[] y, double[] pTrain, double[] rTrain, double l1Ratio) {
		if (fitIntercept) {
			OptimUtils.fitIntercept(pTrain, rTrain, y);
		}
		double maxLambda = 0;
		for (double[] col : x) {
			double eta = 0;
			for (int i = 0; i < col.length; i++) {
				eta += rTrain[i] * col[i];
			}

			double t = Math.abs(eta);
			if (t > maxLambda) {
				maxLambda = t;
			}
		}
		maxLambda /= y.length;
		maxLambda /= l1Ratio;
		if (fitIntercept) {
			Arrays.fill(pTrain, 0);
			OptimUtils.computePseudoResidual(pTrain, y, rTrain);
		}
		return maxLambda;
	}

	protected double findMaxLambdaBinomial(int[][] indices, double[][] values, double[] y, double[] pTrain, double[] rTrain, double l1Ratio) {
		if (fitIntercept) {
			OptimUtils.fitIntercept(pTrain, rTrain, y);
		}
		double maxLambda = 0;
		for (int k = 0; k < values.length; k++) {
			double eta = 0;
			int[] index = indices[k];
			double[] value = values[k];
			for (int i = 0; i < index.length; i++) {
				int idx = index[i];
				double r = OptimUtils.getPseudoResidual(pTrain[idx], y[idx]);
				r *= value[i];
				eta += r;
			}
			double t = Math.abs(eta);
			if (t > maxLambda) {
				maxLambda = t;
			}
		}
		maxLambda /= y.length;
		maxLambda /= l1Ratio;
		if (fitIntercept) {
			Arrays.fill(pTrain, 0);
			OptimUtils.computePseudoResidual(pTrain, y, rTrain);
		}
		return maxLambda;
	}

	/**
	 * Returns the l1 ratio.
	 * 
	 * @return the l1 ratio.
	 */
	public double getL1Ratio() {
		return l1Ratio;
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
	 * Returns the task of this learner.
	 * 
	 * @return the task of this learner.
	 */
	public Task getTask() {
		return task;
	}

	/**
	 * Sets the l1 ratio.
	 * 
	 * @param l1Ratio the l1 ratio.
	 */
	public void setL1Ratio(double l1Ratio) {
		this.l1Ratio = l1Ratio;
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
	 * Sets the task of this learner.
	 * 
	 * @param task the task of this learner.
	 */
	public void setTask(Task task) {
		this.task = task;
	}

}
