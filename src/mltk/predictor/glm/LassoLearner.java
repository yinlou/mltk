package mltk.predictor.glm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.cmdline.options.LearnerWithTaskOptions;
import mltk.core.Attribute;
import mltk.core.DenseVector;
import mltk.core.Instances;
import mltk.core.NominalAttribute;
import mltk.core.SparseVector;
import mltk.core.io.InstancesReader;
import mltk.predictor.Learner;
import mltk.predictor.io.PredictorWriter;
import mltk.util.MathUtils;
import mltk.util.OptimUtils;
import mltk.util.StatUtils;
import mltk.util.VectorUtils;

/**
 * Class for learning L1-regularized linear model via coordinate descent.
 * 
 * @author Yin Lou
 * 
 */
public class LassoLearner extends Learner {
	
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
	 * Usage: mltk.predictor.glm.LassoLearner
	 * -t	train set path
	 * [-g]	task between classification (c) and regression (r) (default: r)
	 * [-r]	attribute file path
	 * [-o]	output model path
	 * [-V]	verbose (default: true)
	 * [-m]	maximum num of iterations (default: 0)
	 * [-l]	lambda (default: 0)
	 * </pre>
	 * 
	 * </p>
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(LassoLearner.class, opts);
		Task task = null;
		try {
			parser.parse(args);
			task = Task.getEnum(opts.task);
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}
		Instances trainSet = InstancesReader.read(opts.attPath, opts.trainPath);

		LassoLearner learner = new LassoLearner();
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

	static class ModelStructure {

		boolean[] structure;

		ModelStructure(boolean[] structure) {
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
	private int maxNumIters;
	private int numLambdas;
	private double epsilon;
	private double lambda;
	private Task task;

	/**
	 * Constructor.
	 */
	public LassoLearner() {
		verbose = false;
		fitIntercept = true;
		refit = false;
		maxNumIters = -1;
		epsilon = MathUtils.EPSILON;
		lambda = 0; // no regularization
		numLambdas = -1; // no regularization path
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
	 * Builds an L1-regularized binary classifier. Each row in the input matrix x represents a feature (instead of a
	 * data point). Thus the input matrix is the transpose of the row-oriented data matrix. This procedure does not
	 * assume the data is normalized or centered.
	 * 
	 * @param attrs the attribute list.
	 * @param x the inputs.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @return an L1-regularized classifier.
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

		// Coordinate descent
		final double tl1 = lambda * y.length;
		for (int iter = 0; iter < maxNumIters; iter++) {
			double prevLoss = GLMOptimUtils.computeLassoLoss(pTrain, y, w, lambda);
			
			if (fitIntercept) {
				intercept += OptimUtils.fitIntercept(pTrain, rTrain, y);
			}

			doOnePass(x, theta, y, tl1, w, pTrain, rTrain);

			double currLoss = GLMOptimUtils.computeLassoLoss(pTrain, y, w, lambda);

			if (verbose) {
				System.out.println("Iteration " + iter + ": " + " " + currLoss);
			}

			if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
				break;
			}
		}
		
		if (refit) {
			boolean[] selected = new boolean[attrs.length];
			for (int i = 0; i < selected.length; i++) {
				selected[i] = w[i] != 0;
			}
			return refitClassifier(attrs, selected, x, y, maxNumIters);
		} else {
			return GLMOptimUtils.getGLM(attrs, w, intercept);
		}
	}

	/**
	 * Builds an L1-regularized binary classifier on sparse inputs. Each row of the input represents a feature (instead
	 * of a data point), i.e., in column-oriented format. This procedure does not assume the data is normalized or
	 * centered.
	 * 
	 * @param attrs the attribute list.
	 * @param indices the indices.
	 * @param values the values.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @return an L1-regularized classifier.
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

		// Coordinate descent
		final double tl1 = lambda * y.length;
		for (int iter = 0; iter < maxNumIters; iter++) {
			double prevLoss = GLMOptimUtils.computeLassoLoss(pTrain, y, w, lambda);
			
			if (fitIntercept) {
				intercept += OptimUtils.fitIntercept(pTrain, rTrain, y);
			}

			doOnePass(indices, values, theta, y, tl1, w, pTrain, rTrain);

			double currLoss = GLMOptimUtils.computeLassoLoss(pTrain, y, w, lambda);

			if (verbose) {
				System.out.println("Iteration " + iter + ": " + " " + currLoss);
			}

			if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
				break;
			}
		}
		
		if (refit) {
			boolean[] selected = new boolean[attrs.length];
			for (int i = 0; i < selected.length; i++) {
				selected[i] = w[i] != 0;
			}
			return refitClassifier(attrs, selected, indices, values, y, maxNumIters);
		} else {
			return GLMOptimUtils.getGLM(attrs, w, intercept);
		}
	}

	/**
	 * Builds L1-regularized binary classifiers for a sequence of regularization parameter lambdas. Each row in the
	 * input matrix x represents a feature (instead of a data point). Thus the input matrix is the transpose of the
	 * row-oriented data matrix. This procedure does not assume the data is normalized or centered.
	 * 
	 * @param attrs the attribute list.
	 * @param x the values.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numLambdas the number of lambdas.
	 * @param minLambdaRatio the minimum lambda is minLambdaRatio * max lambda.
	 * @return L1-regularized classifiers.
	 */
	public List<GLM> buildBinaryClassifiers(int[] attrs, double[][] x, double[] y, int maxNumIters, int numLambdas,
			double minLambdaRatio) {
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

		double maxLambda = findMaxLambda(x, y, pTrain, rTrain);

		// Dampening factor for lambda
		double alpha = Math.pow(minLambdaRatio, 1.0 / numLambdas);

		// Compute the regularization path
		List<GLM> glms = new ArrayList<>(numLambdas);
		Set<ModelStructure> structures = new HashSet<>();
		double lambda = maxLambda;
		for (int g = 0; g < numLambdas; g++) {
			// Coordinate descent
			final double tl1 = lambda * y.length;
			for (int iter = 0; iter < maxNumIters; iter++) {
				double prevLoss = GLMOptimUtils.computeLassoLoss(pTrain, y, w, lambda);
				
				if (fitIntercept) {
					intercept += OptimUtils.fitIntercept(pTrain, rTrain, y);
				}

				doOnePass(x, theta, y, tl1, w, pTrain, rTrain);

				double currLoss = GLMOptimUtils.computeLassoLoss(pTrain, y, w, lambda);

				if (verbose) {
					System.out.println("Iteration " + iter + ": " + " " + currLoss);
				}

				if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
					break;
				}
			}

			lambda *= alpha;
			if (refit) {
				boolean[] selected = new boolean[attrs.length];
				for (int i = 0; i < selected.length; i++) {
					selected[i] = w[i] != 0;
				}
				ModelStructure structure = new ModelStructure(selected);
				if (!structures.contains(structure)) {
					GLM glm = refitClassifier(attrs, selected, x, y, maxNumIters);
					glms.add(glm);
					structures.add(structure);
				}
			} else {
				GLM glm = GLMOptimUtils.getGLM(attrs, w, intercept);
				glms.add(glm);
			}
		}

		return glms;
	}

	/**
	 * Builds L1-regularized binary classifiers for a sequence of regularization parameter lambdas on sparse format.
	 * Each row of the input represents a feature (instead of a data point), i.e., in column-oriented format. This
	 * procedure does not assume the data is normalized or centered.
	 * 
	 * @param attrs the attribute list.
	 * @param indices the indices.
	 * @param values the values.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numLambdas the number of lambdas.
	 * @param minLambdaRatio the minimum lambda is minLambdaRatio * max lambda.
	 * @return L1-regularized classifiers.
	 */
	public List<GLM> buildBinaryClassifiers(int[] attrs, int[][] indices, double[][] values, double[] y, int maxNumIters,
			int numLambdas, double minLambdaRatio) {
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

		double maxLambda = findMaxLambda(indices, values, y, pTrain, rTrain);

		// Dampening factor for lambda
		double alpha = Math.pow(minLambdaRatio, 1.0 / numLambdas);

		// Compute the regularization path
		List<GLM> glms = new ArrayList<>(numLambdas);
		Set<ModelStructure> structures = new HashSet<>();
		double lambda = maxLambda;
		for (int g = 0; g < numLambdas; g++) {
			// Coordinate descent
			final double tl1 = lambda * y.length;
			for (int iter = 0; iter < maxNumIters; iter++) {
				double prevLoss = GLMOptimUtils.computeLassoLoss(pTrain, y, w, lambda);
				
				if (fitIntercept) {
					intercept += OptimUtils.fitIntercept(pTrain, rTrain, y);
				}

				doOnePass(indices, values, theta, y, tl1, w, pTrain, rTrain);

				double currLoss = GLMOptimUtils.computeLassoLoss(pTrain, y, w, lambda);

				if (verbose) {
					System.out.println("Iteration " + iter + ": " + " " + currLoss);
				}

				if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
					break;
				}
			}

			lambda *= alpha;
			if (refit) {
				boolean[] selected = new boolean[attrs.length];
				for (int i = 0; i < selected.length; i++) {
					selected[i] = w[i] != 0;
				}
				ModelStructure structure = new ModelStructure(selected);
				if (!structures.contains(structure)) {
					GLM glm = refitClassifier(attrs, selected, indices, values, y, maxNumIters);
					glms.add(glm);
					structures.add(structure);
				}
			} else {
				GLM glm = GLMOptimUtils.getGLM(attrs, w, intercept);
				glms.add(glm);
			}
		}

		return glms;
	}

	/**
	 * Builds an L1-regularized classifier.
	 * 
	 * @param trainSet the training set.
	 * @param isSparse <code>true</code> if the training set is treated as sparse.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @return an L1-regularized classifer.
	 */
	public GLM buildClassifier(Instances trainSet, boolean isSparse, int maxNumIters, double lambda) {
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

				GLM glm = buildBinaryClassifier(attrs, indices, values, y, maxNumIters, lambda);

				double[] w = glm.w[0];
				for (int j = 0; j < cList.length; j++) {
					int attIndex = sd.attrs[j];
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
				int p = attrs.length == 0 ? 0 : (StatUtils.max(attrs) + 1);
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
	 * Builds an L1-regularized classifier.
	 * 
	 * @param trainSet the training set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @return an L1-regularized classifer.
	 */
	public GLM buildClassifier(Instances trainSet, int maxNumIters, double lambda) {
		return buildClassifier(trainSet, isSparse(trainSet), maxNumIters, lambda);
	}

	/**
	 * Builds L1-regularized classifiers for a sequence of regularization parameter lambdas.
	 * 
	 * @param trainSet the training set.
	 * @param isSparse <code>true</code> if the training set is treated as sparse.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numLambdas the number of lambdas.
	 * @param minLambdaRatio the minimum lambda is minLambdaRatio * max lambda.
	 * @return L1-regularized classifiers.
	 */
	public List<GLM> buildClassifiers(Instances trainSet, boolean isSparse, int maxNumIters, int numLambdas,
			double minLambdaRatio) {
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

				List<GLM> glms = buildBinaryClassifiers(attrs, indices, values, y, maxNumIters, numLambdas,
						minLambdaRatio);

				for (GLM glm : glms) {
					double[] w = glm.w[0];
					for (int j = 0; j < cList.length; j++) {
						int attIndex = sd.attrs[j];
						w[attIndex] *= cList[j];
					}
				}

				return glms;
			} else {
				boolean refit = this.refit;
				this.refit = false; // Not supported in multiclass
									// classification

				int p = attrs.length == 0 ? 0 : (StatUtils.max(attrs) + 1);
				List<GLM> glms = new ArrayList<>();
				for (int i = 0; i < numLambdas; i++) {
					GLM glm = new GLM(numClasses, p);
					glms.add(glm);
				}

				for (int k = 0; k < numClasses; k++) {
					// One-vs-the-rest
					for (int i = 0; i < y.length; i++) {
						int label = (int) sd.y[i];
						y[i] = label == k ? 1 : 0;
					}

					List<GLM> binaryClassifiers = buildBinaryClassifiers(attrs, indices, values, y, maxNumIters,
							numLambdas, minLambdaRatio);

					for (int l = 0; l < numLambdas; l++) {
						GLM binaryClassifier = binaryClassifiers.get(l);
						GLM glm = glms.get(l);
						double[] w = binaryClassifier.w[0];
						for (int j = 0; j < cList.length; j++) {
							int attIndex = attrs[j];
							glm.w[k][attIndex] = w[attIndex] * cList[j];
						}
						glm.intercept[k] = binaryClassifier.intercept[0];
					}

				}

				this.refit = refit;

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

				List<GLM> glms = buildBinaryClassifiers(attrs, x, y, maxNumIters, numLambdas, minLambdaRatio);

				for (GLM glm : glms) {
					double[] w = glm.w[0];
					for (int j = 0; j < cList.length; j++) {
						int attIndex = dd.attrs[j];
						w[attIndex] *= cList[j];
					}
				}

				return glms;
			} else {
				boolean refit = this.refit;
				this.refit = false; // Not supported in multiclass
									// classification

				int p = attrs.length == 0 ? 0 : (StatUtils.max(attrs) + 1);
				List<GLM> glms = new ArrayList<>();
				for (int i = 0; i < numLambdas; i++) {
					GLM glm = new GLM(numClasses, p);
					glms.add(glm);
				}

				for (int k = 0; k < numClasses; k++) {
					// One-vs-the-rest
					for (int i = 0; i < y.length; i++) {
						int label = (int) dd.y[i];
						y[i] = label == k ? 1 : 0;
					}

					List<GLM> binaryClassifiers = buildBinaryClassifiers(attrs, x, y, maxNumIters, numLambdas,
							minLambdaRatio);

					for (int l = 0; l < numLambdas; l++) {
						GLM binaryClassifier = binaryClassifiers.get(l);
						GLM glm = glms.get(l);
						double[] w = binaryClassifier.w[0];
						for (int j = 0; j < cList.length; j++) {
							int attIndex = attrs[j];
							glm.w[k][attIndex] = w[attIndex] * cList[j];
						}
						glm.intercept[k] = binaryClassifier.intercept[0];
					}

				}

				this.refit = refit;

				return glms;
			}
		}
	}

	/**
	 * Builds L1-regularized classifiers for a sequence of regularization parameter lambdas.
	 * 
	 * @param trainSet the training set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numLambdas the number of lambdas.
	 * @param minLambdaRatio the minimum lambda is minLambdaRatio * max lambda.
	 * @return L1-regularized classifiers.
	 */
	public List<GLM> buildClassifiers(Instances trainSet, int maxNumIters, int numLambdas, double minLambdaRatio) {
		return buildClassifiers(trainSet, isSparse(trainSet), maxNumIters, numLambdas, minLambdaRatio);
	}

	/**
	 * Builds an L1-regularized regressor.
	 * 
	 * @param trainSet the training set.
	 * @param isSparse <code>true</code> if the training set is treated as sparse.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @return an L1-regularized regressor.
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
	 * Builds an L1-regularized regressor.
	 * 
	 * @param trainSet the training set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @return an L1-regularized regressor.
	 */
	public GLM buildRegressor(Instances trainSet, int maxNumIters, double lambda) {
		return buildRegressor(trainSet, isSparse(trainSet), maxNumIters, lambda);
	}

	/**
	 * Builds an L1-regularized regressor. Each row in the input matrix x represents a feature (instead of a data
	 * point). Thus the input matrix is the transpose of the row-oriented data matrix. This procedure does not assume
	 * the data is normalized or centered.
	 * 
	 * @param attrs the attribute list.
	 * @param x the inputs.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @return an L1-regularized penalized regressor.
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
		for (int j = 0; j < x.length; j++) {
			sq[j] = StatUtils.sumSq(x[j]);
		}

		// Coordinate descent
		final double tl1 = lambda * y.length;
		for (int iter = 0; iter < maxNumIters; iter++) {
			double prevLoss = GLMOptimUtils.computeLassoLoss(rTrain, w, lambda);
			
			if (fitIntercept) {
				intercept += OptimUtils.fitIntercept(rTrain);
			}

			doOnePass(x, sq, tl1, w, rTrain);

			double currLoss = GLMOptimUtils.computeLassoLoss(rTrain, w, lambda);

			if (verbose) {
				System.out.println("Iteration " + iter + ": " + " " + currLoss);
			}

			if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
				break;
			}
		}
		
		if (refit) {
			boolean[] selected = new boolean[attrs.length];
			for (int i = 0; i < selected.length; i++) {
				selected[i] = w[i] != 0;
			}
			return refitRegressor(attrs, selected, x, y, maxNumIters);
		} else {
			return GLMOptimUtils.getGLM(attrs, w, intercept);
		}
	}

	/**
	 * Builds an L1-regularized regressor on sparse inputs. Each row of the input represents a feature (instead of a
	 * data point), i.e., in column-oriented format. This procedure does not assume the data is normalized or centered.
	 * 
	 * @param attrs the attribute list.
	 * @param indices the indices.
	 * @param values the values.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @return an L1-regularized regressor.
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
		double[] sq = new double[values.length];
		for (int j = 0; j < values.length; j++) {
			sq[j] = StatUtils.sumSq(values[j]);
		}

		// Coordinate descent
		final double tl1 = lambda * y.length;
		for (int iter = 0; iter < maxNumIters; iter++) {
			double prevLoss = GLMOptimUtils.computeLassoLoss(rTrain, w, lambda);
			
			if (fitIntercept) {
				intercept += OptimUtils.fitIntercept(rTrain);
			}

			doOnePass(indices, values, sq, tl1, w, rTrain);

			double currLoss = GLMOptimUtils.computeLassoLoss(rTrain, w, lambda);

			if (verbose) {
				System.out.println("Iteration " + iter + ": " + " " + currLoss);
			}

			if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
				break;
			}
		}
		
		if (refit) {
			boolean[] selected = new boolean[attrs.length];
			for (int i = 0; i < selected.length; i++) {
				selected[i] = w[i] != 0;
			}
			return refitRegressor(attrs, selected, indices, values, y, maxNumIters);
		} else {
			return GLMOptimUtils.getGLM(attrs, w, intercept);
		}
	}

	/**
	 * Builds L1-regularized regressors for a sequence of regularization parameter lambdas.
	 * 
	 * @param trainSet the training set.
	 * @param isSparse <code>true</code> if the training set is treated as sparse.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numLambdas the number of lambdas.
	 * @param minLambdaRatio the minimum lambda is minLambdaRatio * max lambda.
	 * @return L1-regularized regressors.
	 */
	public List<GLM> buildRegressors(Instances trainSet, boolean isSparse, int maxNumIters, int numLambdas,
			double minLambdaRatio) {
		if (isSparse) {
			SparseDataset sd = getSparseDataset(trainSet, true);
			double[] cList = sd.cList;

			List<GLM> glms = buildRegressors(sd.attrs, sd.indices, sd.values, sd.y, maxNumIters, numLambdas,
					minLambdaRatio);

			for (GLM glm : glms) {
				double[] w = glm.w[0];
				for (int i = 0; i < cList.length; i++) {
					int attIndex = sd.attrs[i];
					w[attIndex] *= cList[i];
				}
			}

			return glms;
		} else {
			DenseDataset dd = getDenseDataset(trainSet, true);
			double[] cList = dd.cList;

			List<GLM> glms = buildRegressors(dd.attrs, dd.x, dd.y, maxNumIters, numLambdas, minLambdaRatio);

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
	 * Builds L1-regularized regressors for a sequence of regularization parameter lambdas.
	 * 
	 * @param trainSet the training set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numLambdas the number of lambdas.
	 * @param minLambdaRatio the minimum lambda is minLambdaRatio * max lambda.
	 * @return L1-regularized regressors.
	 */
	public List<GLM> buildRegressors(Instances trainSet, int maxNumIters, int numLambdas, double minLambdaRatio) {
		return buildRegressors(trainSet, isSparse(trainSet), maxNumIters, numLambdas, minLambdaRatio);
	}

	/**
	 * Builds L1-regularized regressors for a sequence of regularization parameter lambdas. Each row in the input matrix
	 * x represents a feature (instead of a data point). Thus the input matrix is the transpose of the row-oriented data
	 * matrix. This procedure does not assume the data is normalized or centered.
	 * 
	 * @param attrs the attribute list.
	 * @param x the inputs.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numLambdas the number of lambdas.
	 * @param minLambdaRatio the minimum lambda is minLambdaRatio * max lambda.
	 * @return L1-regularized regressors.
	 */
	public List<GLM> buildRegressors(int[] attrs, double[][] x, double[] y, int maxNumIters, int numLambdas,
			double minLambdaRatio) {
		double[] w = new double[attrs.length];
		double intercept = 0;

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
		double maxLambda = findMaxLambda(x, rTrain);

		// Dampening factor for lambda
		double alpha = Math.pow(minLambdaRatio, 1.0 / numLambdas);

		// Compute the regularization path
		List<GLM> glms = new ArrayList<>(numLambdas);
		Set<ModelStructure> structures = new HashSet<>();
		double lambda = maxLambda;
		for (int g = 0; g < numLambdas; g++) {
			final double tl1 = lambda * y.length;
			// Coordinate descent
			for (int iter = 0; iter < maxNumIters; iter++) {
				double prevLoss = GLMOptimUtils.computeLassoLoss(rTrain, w, lambda);
				
				if (fitIntercept) {
					intercept += OptimUtils.fitIntercept(rTrain);
				}

				doOnePass(x, sq, tl1, w, rTrain);

				double currLoss = GLMOptimUtils.computeLassoLoss(rTrain, w, lambda);
				
				if (verbose) {
					System.out.println("Iteration " + iter + ": " + " " + currLoss);
				}

				if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
					break;
				}
			}

			lambda *= alpha;
			if (refit) {
				boolean[] selected = new boolean[attrs.length];
				for (int i = 0; i < selected.length; i++) {
					selected[i] = w[i] != 0;
				}
				ModelStructure structure = new ModelStructure(selected);
				if (!structures.contains(structure)) {
					GLM glm = refitRegressor(attrs, selected, x, y, maxNumIters);
					glms.add(glm);
					structures.add(structure);
				}
			} else {
				GLM glm = GLMOptimUtils.getGLM(attrs, w, intercept);
				glms.add(glm);
			}
		}

		return glms;
	}

	/**
	 * Builds L1-regularized regressors for a sequence of regularization parameter lambdas on sparse format. Each row of
	 * the input represents a feature (instead of a data point), i.e., in column-oriented format. This procedure does
	 * not assume the data is normalized or centered.
	 * 
	 * @param attrs the attribute list.
	 * @param indices the indices.
	 * @param values the values.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numLambdas the number of lambdas.
	 * @param minLambdaRatio the minimum lambda is minLambdaRatio * max lambda.
	 * @return L1-regularized regressors.
	 */
	public List<GLM> buildRegressors(int[] attrs, int[][] indices, double[][] values, double[] y, int maxNumIters,
			int numLambdas, double minLambdaRatio) {
		double[] w = new double[attrs.length];
		double intercept = 0;

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
		double maxLambda = findMaxLambda(indices, values, y);

		// Dampening factor for lambda
		double alpha = Math.pow(minLambdaRatio, 1.0 / numLambdas);

		// Compute the regularization path
		List<GLM> glms = new ArrayList<>(numLambdas);
		Set<ModelStructure> structures = new HashSet<>();
		double lambda = maxLambda;
		for (int g = 0; g < numLambdas; g++) {
			final double tl1 = lambda * y.length;
			// Coordinate descent
			for (int iter = 0; iter < maxNumIters; iter++) {
				double prevLoss = GLMOptimUtils.computeLassoLoss(rTrain, w, lambda);
				
				if (fitIntercept) {
					intercept += OptimUtils.fitIntercept(rTrain);
				}

				doOnePass(indices, values, sq, tl1, w, rTrain);

				double currLoss = GLMOptimUtils.computeLassoLoss(rTrain, w, lambda);
				
				if (verbose) {
					System.out.println("Iteration " + iter + ": " + " " + currLoss);
				}

				if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
					break;
				}
			}

			lambda *= alpha;
			if (refit) {
				boolean[] selected = new boolean[attrs.length];
				for (int i = 0; i < selected.length; i++) {
					selected[i] = w[i] != 0;
				}
				ModelStructure structure = new ModelStructure(selected);
				if (!structures.contains(structure)) {
					GLM glm = refitRegressor(attrs, selected, indices, values, y, maxNumIters);
					glms.add(glm);
					structures.add(structure);
				}
			} else {
				GLM glm = GLMOptimUtils.getGLM(attrs, w, intercept);
				glms.add(glm);
			}
		}

		return glms;
	}

	protected void doOnePass(double[][] x, double[] sq, final double tl1, double[] w, double[] rTrain) {
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
			wNew /= sq[j];

			double delta = wNew - w[j];
			w[j] = wNew;

			// Update residuals
			for (int i = 0; i < rTrain.length; i++) {
				rTrain[i] -= delta * v[i];
			}
		}
	}

	protected void doOnePass(double[][] x, double[] theta, double[] y, final double tl1, double[] w, double[] pTrain,
			double[] rTrain) {
		for (int j = 0; j < x.length; j++) {
			if (Math.abs(theta[j]) <= MathUtils.EPSILON) {
				continue;
			}

			double[] v = x[j];
			double eta = VectorUtils.dotProduct(rTrain, v);

			double newW = w[j] + eta / theta[j];
			double t = tl1 / theta[j];
			if (newW > t) {
				newW -= t;
			} else if (newW < -t) {
				newW += t;
			} else {
				newW = 0;
			}

			double delta = newW - w[j];
			w[j] = newW;

			// Update predictions
			for (int i = 0; i < pTrain.length; i++) {
				pTrain[i] += delta * v[i];
				rTrain[i] = OptimUtils.getPseudoResidual(pTrain[i], y[i]);
			}
		}
	}

	protected void doOnePass(int[][] indices, double[][] values, double[] sq, final double tl1, double[] w,
			double[] rTrain) {
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
			wNew /= sq[j];

			double delta = wNew - w[j];
			w[j] = wNew;

			// Update residuals
			for (int i = 0; i < index.length; i++) {
				rTrain[index[i]] -= delta * value[i];
			}
		}
	}

	protected void doOnePass(int[][] indices, double[][] values, double[] theta, double[] y, final double tl1, double[] w,
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

			double newW = w[j] + eta / theta[j];
			double t = tl1 / theta[j];
			if (newW > t) {
				newW -= t;
			} else if (newW < -t) {
				newW += t;
			} else {
				newW = 0;
			}

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

	protected double findMaxLambda(double[][] x, double[] y) {
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
		if (fitIntercept) {
			VectorUtils.add(y, mean);
		}
		return maxLambda;
	}

	protected double findMaxLambda(double[][] x, double[] y, double[] pTrain, double[] rTrain) {
		if (fitIntercept) {
			OptimUtils.fitIntercept(pTrain, rTrain, y);
		}
		double maxLambda = 0;
		for (double[] col : x) {
			double eta = 0;
			for (int i = 0; i < col.length; i++) {
				double r = OptimUtils.getPseudoResidual(pTrain[i], y[i]);
				r *= col[i];
				eta += r;
			}

			double t = Math.abs(eta);
			if (t > maxLambda) {
				maxLambda = t;
			}
		}
		maxLambda /= y.length;
		if (fitIntercept) {
			Arrays.fill(pTrain, 0);
			OptimUtils.computePseudoResidual(pTrain, y, rTrain);
		}
		return maxLambda;
	}

	protected double findMaxLambda(int[][] indices, double[][] values, double[] y) {
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
		if (fitIntercept) {
			VectorUtils.add(y, mean);
		}
		return maxLambda;
	}

	protected double findMaxLambda(int[][] indices, double[][] values, double[] y, double[] pTrain, double[] rTrain) {
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
		if (fitIntercept) {
			Arrays.fill(pTrain, 0);
			OptimUtils.computePseudoResidual(pTrain, y, rTrain);
		}
		return maxLambda;
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
	 * Returns the number of lambdas.
	 * 
	 * @return the number of lambdas.
	 */
	public int getNumLambdas() {
		return numLambdas;
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

	protected GLM refitRegressor(int[] attrs, boolean[] selected, double[][] x, double[] y, int maxNumIters) {
		List<double[]> xList = new ArrayList<>();
		for (int j = 0; j < attrs.length; j++) {
			if (selected[j]) {
				xList.add(x[j]);
			}
		}
		if (xList.size() == 0) {
			GLM glm = new GLM(0);
			if (fitIntercept) {
				glm.intercept[0] = StatUtils.mean(y);
			}
			return glm;
		}
		double[][] xNew = new double[xList.size()][];
		for (int i = 0; i < xNew.length; i++) {
			xNew[i] = xList.get(i);
		}
		int[] attrsNew = new int[xNew.length];
		for (int i = 0; i < attrsNew.length; i++) {
			attrsNew[i] = i;
		}

		RidgeLearner learner = new RidgeLearner();
		learner.setVerbose(verbose);
		learner.setEpsilon(epsilon);
		learner.fitIntercept(fitIntercept);
		// A ridge regression with very small regularization parameter
		// This often improves stability a lot
		GLM glm = learner.buildRegressor(attrsNew, xNew, y, maxNumIters, 1e-8);
		double[] w = new double[attrs.length];
		double[] coef = glm.coefficients(0);
		int k = 0;
		for (int j = 0; j < w.length; j++) {
			if (selected[j]) {
				w[j] = coef[k++];
			}
		}

		return GLMOptimUtils.getGLM(attrs, w, glm.intercept(0));
	}
	

	protected GLM refitRegressor(int[] attrs, boolean[] selected, int[][] indices, double[][] values, double[] y, int maxNumIters) {
		List<int[]> indicesList = new ArrayList<>();
		List<double[]> valuesList = new ArrayList<>();
		for (int j = 0; j < attrs.length; j++) {
			if (selected[j]) {
				indicesList.add(indices[j]);
				valuesList.add(values[j]);
			}
		}
		if (indicesList.size() == 0) {
			GLM glm = new GLM(0);
			if (fitIntercept) {
				glm.intercept[0] = StatUtils.mean(y);
			}
			return glm;
		}
		int[][] indicesNew = new int[indicesList.size()][];
		for (int i = 0; i < indicesNew.length; i++) {
			indicesNew[i] = indicesList.get(i);
		}
		double[][] valuesNew = new double[valuesList.size()][];
		for (int i = 0; i < indicesNew.length; i++) {
			valuesNew[i] = valuesList.get(i);
		}
		int[] attrsNew = new int[indicesNew.length];
		for (int i = 0; i < attrsNew.length; i++) {
			attrsNew[i] = i;
		}

		RidgeLearner learner = new RidgeLearner();
		learner.setVerbose(verbose);
		learner.setEpsilon(epsilon);
		learner.fitIntercept(fitIntercept);
		// A ridge regression with very small regularization parameter
		// This often improves stability a lot
		GLM glm = learner.buildRegressor(attrsNew, indicesNew, valuesNew, y, maxNumIters, 1e-8);
		double[] w = new double[attrs.length];
		double[] coef = glm.coefficients(0);
		int k = 0;
		for (int j = 0; j < w.length; j++) {
			if (selected[j]) {
				w[j] = coef[k++];
			}
		}

		return GLMOptimUtils.getGLM(attrs, w, glm.intercept(0));
	}

	protected GLM refitClassifier(int[] attrs, boolean[] selected, double[][] x, double[] y, int maxNumIters) {
		List<double[]> xList = new ArrayList<>();
		for (int j = 0; j < attrs.length; j++) {
			if (selected[j]) {
				xList.add(x[j]);
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

		RidgeLearner learner = new RidgeLearner();
		learner.setVerbose(verbose);
		learner.setEpsilon(epsilon);
		learner.fitIntercept(fitIntercept);
		// A ridge regression with very small regularization parameter
		// This often improves stability a lot
		GLM glm = learner.buildBinaryClassifier(attrsNew, xNew, y, maxNumIters, 1e-8);
		double[] w = new double[attrs.length];
		double[] coef = glm.coefficients(0);
		int k = 0;
		for (int j = 0; j < w.length; j++) {
			if (selected[j]) {
				w[j] = coef[k++];
			}
		}

		return GLMOptimUtils.getGLM(attrs, w, glm.intercept(0));
	}

	protected GLM refitClassifier(int[] attrs, boolean[] selected, int[][] indices, double[][] values, double[] y, int maxNumIters) {
		List<int[]> indicesList = new ArrayList<>();
		List<double[]> valuesList = new ArrayList<>();
		for (int j = 0; j < attrs.length; j++) {
			if (selected[j]) {
				indicesList.add(indices[j]);
				valuesList.add(values[j]);
			}
		}

		int[][] indicesNew = new int[indicesList.size()][];
		for (int i = 0; i < indicesNew.length; i++) {
			indicesNew[i] = indicesList.get(i);
		}
		double[][] valuesNew = new double[valuesList.size()][];
		for (int i = 0; i < indicesNew.length; i++) {
			valuesNew[i] = valuesList.get(i);
		}
		int[] attrsNew = new int[indicesNew.length];
		for (int i = 0; i < attrsNew.length; i++) {
			attrsNew[i] = i;
		}

		RidgeLearner learner = new RidgeLearner();
		learner.setVerbose(verbose);
		learner.setEpsilon(epsilon);
		learner.fitIntercept(fitIntercept);
		// A ridge regression with very small regularization parameter
		// This often improves stability a lot
		GLM glm = learner.buildBinaryClassifier(attrsNew, indicesNew, valuesNew, y, maxNumIters, 1e-8);
		double[] w = new double[attrs.length];
		double[] coef = glm.coefficients(0);
		int k = 0;
		for (int j = 0; j < w.length; j++) {
			if (selected[j]) {
				w[j] = coef[k++];
			}
		}

		return GLMOptimUtils.getGLM(attrs, w, glm.intercept(0));
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
	 * Sets the number of lambdas.
	 * 
	 * @param numLambdas the number of lambdas.
	 */
	public void setNumLambdas(int numLambdas) {
		this.numLambdas = numLambdas;
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
