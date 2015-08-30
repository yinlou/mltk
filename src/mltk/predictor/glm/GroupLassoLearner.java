package mltk.predictor.glm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import mltk.core.Attribute;
import mltk.core.Instances;
import mltk.core.NominalAttribute;
import mltk.predictor.Learner;
import mltk.predictor.glm.GLM;
import mltk.util.ArrayUtils;
import mltk.util.MathUtils;
import mltk.util.OptimUtils;
import mltk.util.StatUtils;
import mltk.util.VectorUtils;

/**
 * Class for learning group-lasso models via block coordinate gradient descent.
 * 
 * <p>
 * Reference:<br>
 * M Yuan and Y Lin. Model selection and estimation in regression with grouped variables. In 
 * <i>Journal of the Royal Statistical Society: Series B (Statistical Methodology)</i>, 
 * 68(1):49-67, 2006.
 * </p>
 * 
 * @author Yin Lou
 * 
 */
public class GroupLassoLearner extends Learner {
	
	class DenseDesignMatrix {

		int[][] groups;
		double[][][] x;

		DenseDesignMatrix(int[][] groups, double[][][] x) {
			this.groups = groups;
			this.x = x;
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
	
	class SparseDesignMatrix {

		int[][] group;
		int[][][] indices;
		double[][][] values;

		SparseDesignMatrix(int[][] groups, int[][][] indices, double[][][] values) {
			this.indices = indices;
			this.values = values;
		}
	}

	private boolean fitIntercept;
	private boolean refit;
	private int maxNumIters;
	private int numLambdas;
	private double lambda;
	private double epsilon;
	private Task task;
	private List<int[]> groups;
	
	/**
	 * Constructor.
	 */
	public GroupLassoLearner() {
		verbose = false;
		fitIntercept = true;
		refit = false;
		maxNumIters = -1;
		lambda = 0.0;
		epsilon = MathUtils.EPSILON;
		task = Task.REGRESSION;
		groups = null;
	}

	@Override
	public GLM build(Instances instances) {
		if (groups == null) {
			throw new IllegalArgumentException("Groups are not set.");
		}
		GLM glm = null;
		if (maxNumIters < 0) {
			maxNumIters = 20;
		}
		switch (task) {
			case REGRESSION:
				glm = buildRegressor(instances, groups, maxNumIters, lambda);
				break;
			case CLASSIFICATION:
				glm = buildClassifier(instances, groups, maxNumIters, lambda);
				break;
			default:
				break;
		}
		return glm;
	}

	/**
	 * Builds a group-lasso penalized binary classifier. The input matrix is grouped by groups. This procedure does not
	 * assume the data is normalized or centered.
	 * 
	 * @param attrs the groups of variables.
	 * @param x the inputs.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @return a group-lasso penalized classifier.
	 */
	public GLM buildBinaryClassifier(int[][] attrs, double[][][] x, double[] y, int maxNumIters, double lambda) {
		int p = 0;
		if (attrs.length > 0) {
			for (int[] attr : attrs) {
				p = Math.max(p, StatUtils.max(attr));
			}
			p += 1;
		}
		
		double[][] w = new double[attrs.length][];
		double[] tl1 = new double[attrs.length];
		int m = 0;
		for (int j = 0; j < attrs.length; j++) {
			w[j] = new double[attrs[j].length];
			tl1[j] = lambda * Math.sqrt(w[j].length);
			if (w[j].length > m) {
				m = w[j].length;
			}
		}

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

		boolean[] activeSet = new boolean[attrs.length];

		double intercept = 0;
		
		// Block coordinate gradient descent
		int iter = 0;
		while (iter < maxNumIters) {

			if (fitIntercept) {
				intercept += OptimUtils.fitIntercept(pTrain, rTrain, y);
			}

			boolean activeSetChanged = doOnePass(x, y, tl1, true, activeSet, w, stepSize, 
					g, gradient, pTrain, rTrain);

			iter++;

			if (!activeSetChanged || iter > maxNumIters) {
				break;
			}

			for (; iter < maxNumIters; iter++) {

				double prevLoss = GLMOptimUtils.computeGroupLassoLoss(pTrain, y, w, tl1);
				
				if (fitIntercept) {
					intercept += OptimUtils.fitIntercept(pTrain, rTrain, y);
				}

				doOnePass(x, y, tl1, false, activeSet, w, stepSize, g, gradient, pTrain, rTrain);

				double currLoss = GLMOptimUtils.computeGroupLassoLoss(pTrain, y, w, tl1);
				
				if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
					break;
				}

				if (verbose) {
					System.out.println("Iteration " + iter + ": " + currLoss);
				}
			}
		}

		if (refit) {
			boolean[] selected = new boolean[attrs.length];
			for (int i = 0; i < selected.length; i++) {
				selected[i] = !ArrayUtils.isConstant(w[i], 0, w[i].length, 0);
			}
			return refitRegressor(p, attrs, selected, x, y, w, maxNumIters);
		} else {
			return getGLM(p, attrs, w, intercept);
		}
	}

	/**
	 * Builds a group-lasso penalized binary classifier on sparse inputs. The input matrix is grouped by groups. This procedure does not
	 * assume the data is normalized or centered.
	 * 
	 * @param attrs the groups of variables.
	 * @param indices the indices.
	 * @param values the values.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @return a group-lasso penalized classifier.
	 */
	public GLM buildBinaryClassifier(int[][] attrs, int[][][] indices, double[][][] values, double[] y, int maxNumIters, double lambda) {
		int p = 0;
		if (attrs.length > 0) {
			for (int[] attr : attrs) {
				p = Math.max(p, StatUtils.max(attr));
			}
			p += 1;
		}
		
		int[][] indexUnion = new int[attrs.length][];
		for (int g = 0; g < attrs.length; g++) {
			int[][] index = indices[g];
			Set<Integer> set = new HashSet<>();
			for (int[] idx : index) {
				for (int i : idx) {
					set.add(i);
				}
			}
			int[] idxUnion = new int[set.size()];
			int k = 0;
			for (int idx : set) {
				idxUnion[k++] = idx;
			}
			indexUnion[g] = idxUnion;
		}
		
		double[][] w = new double[attrs.length][];
		double[] tl1 = new double[attrs.length];
		int m = 0;
		for (int j = 0; j < attrs.length; j++) {
			w[j] = new double[attrs[j].length];
			tl1[j] = lambda * Math.sqrt(w[j].length);
			if (w[j].length > m) {
				m = w[j].length;
			}
		}

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

		boolean[] activeSet = new boolean[values.length];

		double intercept = 0;
		
		// Block coordinate gradient descent
		int iter = 0;
		while (iter < maxNumIters) {

			if (fitIntercept) {
				intercept += OptimUtils.fitIntercept(pTrain, rTrain, y);
			}

			boolean activeSetChanged = doOnePass(indices, indexUnion, values, y, tl1, true, 
					activeSet, w, stepSize, g, gradient, pTrain, rTrain);

			iter++;

			if (!activeSetChanged || iter > maxNumIters) {
				break;
			}

			for (; iter < maxNumIters; iter++) {

				double prevLoss = GLMOptimUtils.computeGroupLassoLoss(pTrain, y, w, tl1);
				
				if (fitIntercept) {
					intercept += OptimUtils.fitIntercept(pTrain, rTrain, y);
				}

				doOnePass(indices, indexUnion, values, y, tl1, true, activeSet, w, stepSize, 
						g, gradient, pTrain, rTrain);

				double currLoss = GLMOptimUtils.computeGroupLassoLoss(pTrain, y, w, tl1);
				
				if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
					break;
				}

				if (verbose) {
					System.out.println("Iteration " + iter + ": " + currLoss);
				}
			}
		}

		if (refit) {
			boolean[] selected = new boolean[attrs.length];
			for (int i = 0; i < selected.length; i++) {
				selected[i] = !ArrayUtils.isConstant(w[i], 0, w[i].length, 0);
			}
			return refitClassifier(p, attrs, selected, indices, values, y, w, maxNumIters);
		} else {
			return getGLM(p, attrs, w, intercept);
		}
	}

	/**
	 * Builds group-lasso penalized binary classifiers for a sequence of regularization parameter lambdas. The input matrix is grouped by groups. This procedure does not
	 * assume the data is normalized or centered.
	 * 
	 * @param attrs the groups of variables.
	 * @param x the inputs.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numLambdas the number of lambdas.
	 * @param minLambdaRatio the minimum lambda is minLambdaRatio * max lambda.
	 * @return group-lasso penalized classifiers.
	 */
	public List<GLM> buildBinaryClassifiers(int[][] attrs, double[][][] x, double[] y, int maxNumIters, int numLambdas, double minLambdaRatio) {
		int p = 0;
		if (attrs.length > 0) {
			for (int[] attr : attrs) {
				p = Math.max(p, StatUtils.max(attr));
			}
			p += 1;
		}
		
		double[][] w = new double[attrs.length][];
		int m = 0;
		for (int j = 0; j < attrs.length; j++) {
			w[j] = new double[attrs[j].length];
			if (w[j].length > m) {
				m = w[j].length;
			}
		}

		double[] pTrain = new double[y.length];
		double[] rTrain = new double[y.length];
		OptimUtils.computePseudoResidual(pTrain, y, rTrain);

		double[] g = new double[m];
		double[] gradient = new double[m];

		double[] tl1 = new double[x.length];

		double[] stepSize = new double[x.length];
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

		boolean[] activeSet = new boolean[x.length];

		double maxLambda = findMaxLambda(x, y, pTrain, rTrain, gradient);

		// Dampening factor for lambda
		double alpha = Math.pow(minLambdaRatio, 1.0 / numLambdas);

		// Compute the regularization path
		List<GLM> glms = new ArrayList<>(numLambdas);
		double lambda = maxLambda;
		double intercept = 0;
		Set<ModelStructure> structures = new HashSet<>();
		for (int l = 0; l < numLambdas; l++) {
			// Initialize regularization parameters
			for (int j = 0; j < tl1.length; j++) {
				tl1[j] = lambda * Math.sqrt(w[j].length);
			}

			// Block coordinate gradient descent
			int iter = 0;
			while (iter < maxNumIters) {

				if (fitIntercept) {
					intercept += OptimUtils.fitIntercept(pTrain, rTrain, y);
				}

				boolean activeSetChanged = doOnePass(x, y, tl1, true, activeSet, w,
						stepSize, g, gradient, pTrain, rTrain);

				iter++;

				if (!activeSetChanged || iter > maxNumIters) {
					break;
				}

				for (; iter < maxNumIters; iter++) {

					double prevLoss = GLMOptimUtils.computeGroupLassoLoss(pTrain, y, w, tl1);
					
					if (fitIntercept) {
						intercept += OptimUtils.fitIntercept(pTrain, rTrain, y);
					}
					
					doOnePass(x, y, tl1, false, activeSet, w, stepSize, g, gradient, pTrain, rTrain);

					double currLoss = GLMOptimUtils.computeGroupLassoLoss(pTrain, y, w, tl1);

					if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
						break;
					}

					if (verbose) {
						System.out.println("Iteration " + iter + ": " + currLoss);
					}
				}
			}

			lambda *= alpha;
			if (refit) {
				boolean[] selected = new boolean[attrs.length];
				for (int i = 0; i < selected.length; i++) {
					selected[i] = !ArrayUtils.isConstant(w[i], 0, w[i].length, 0);
				}
				ModelStructure structure = new ModelStructure(selected);
				if (!structures.contains(structure)) {
					GLM glm = refitClassifier(p, attrs, selected, x, y, w, maxNumIters);
					glms.add(glm);
					structures.add(structure);
				}
			} else {
				GLM glm = getGLM(p, attrs, w, intercept);
				glms.add(glm);
			}
		}

		return glms;
	}

	/**
	 * Builds group-lasso penalized binary classifiers on sparse inputs for a sequence of regularization parameter lambdas. The input matrix is grouped by groups. This procedure does not
	 * assume the data is normalized or centered.
	 * 
	 * @param attrs the groups of variables.
	 * @param indices the indices.
	 * @param values the values.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numLambdas the number of lambdas.
	 * @param minLambdaRatio the minimum lambda is minLambdaRatio * max lambda.
	 * @return group-lasso penalized classifiers.
	 */
	public List<GLM> buildBinaryClassifiers(int[][] attrs, int[][][] indices, double[][][] values, double[] y, 
			int maxNumIters, int numLambdas, double minLambdaRatio) {
		int p = 0;
		if (attrs.length > 0) {
			for (int[] attr : attrs) {
				p = Math.max(p, StatUtils.max(attr));
			}
			p += 1;
		}
		
		int[][] indexUnion = new int[indices.length][];
		for (int g = 0; g < indices.length; g++) {
			int[][] index = indices[g];
			Set<Integer> set = new HashSet<>();
			for (int[] idx : index) {
				for (int i : idx) {
					set.add(i);
				}
			}
			int[] idxUnion = new int[set.size()];
			int k = 0;
			for (int idx : set) {
				idxUnion[k++] = idx;
			}
			indexUnion[g] = idxUnion;
		}
		
		double[][] w = new double[attrs.length][];
		double[] tl1 = new double[attrs.length];
		int m = 0;
		for (int j = 0; j < values.length; j++) {
			w[j] = new double[values[j].length];
			if (w[j].length > m) {
				m = w[j].length;
			}
		}

		double[] pTrain = new double[y.length];
		double[] rTrain = new double[y.length];
		OptimUtils.computePseudoResidual(pTrain, y, rTrain);
		
		double[] stepSize = new double[values.length];
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

		boolean[] activeSet = new boolean[values.length];

		double maxLambda = findMaxLambda(indices, values, y, pTrain, rTrain, gradient);

		// Dampening factor for lambda
		double alpha = Math.pow(minLambdaRatio, 1.0 / numLambdas);

		List<GLM> glms = new ArrayList<>(numLambdas);

		double lambda = maxLambda;
		double intercept = 0;
		Set<ModelStructure> structures = new HashSet<>();
		for (int l = 0; l < numLambdas; l++) {
			// Initialize regularization parameters
			for (int j = 0; j < tl1.length; j++) {
				tl1[j] = lambda * Math.sqrt(w[j].length);
			}

			// Block coordinate gradient descent
			int iter = 0;
			while (iter < maxNumIters) {

				if (fitIntercept) {
					intercept += OptimUtils.fitIntercept(pTrain, rTrain, y);
				}

				boolean activeSetChanged = doOnePass(indices, indexUnion, values, y, tl1, true, 
						activeSet, w, stepSize, g, gradient, pTrain, rTrain);

				iter++;

				if (!activeSetChanged || iter > maxNumIters) {
					break;
				}

				for (; iter < maxNumIters; iter++) {

					double prevLoss = GLMOptimUtils.computeGroupLassoLoss(pTrain, y, w, tl1);
					
					if (fitIntercept) {
						intercept += OptimUtils.fitIntercept(pTrain, rTrain, y);
					}

					doOnePass(indices, indexUnion, values, y, tl1, false, activeSet, w, stepSize,
							g, gradient, pTrain, rTrain);

					double currLoss = GLMOptimUtils.computeGroupLassoLoss(pTrain, y, w, tl1);
					
					if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
						break;
					}

					if (verbose) {
						System.out.println("Iteration " + iter + ": " + currLoss);
					}
				}
			}

			lambda *= alpha;
			if (refit) {
				boolean[] selected = new boolean[attrs.length];
				for (int i = 0; i < selected.length; i++) {
					selected[i] = !ArrayUtils.isConstant(w[i], 0, w[i].length, 0);
				}
				ModelStructure structure = new ModelStructure(selected);
				if (!structures.contains(structure)) {
					GLM glm = refitClassifier(p, attrs, selected, indices, values, y, w, maxNumIters);
					glms.add(glm);
				}
			} else {
				GLM glm = getGLM(p, attrs, w, intercept);
				glms.add(glm);
			}
		}

		return glms;
	}
	
	/**
	 * Builds a group-lasso penalized classifier.
	 * 
	 * @param trainSet the training set.
	 * @param isSparse <code>true</code> if the training set is treated as sparse.
	 * @param groups the groups of variables.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @return a group-lasso penalized classifier.
	 */
	public GLM buildClassifier(Instances trainSet, boolean isSparse, List<int[]> groups, int maxNumIters, double lambda) {
		Attribute classAttribute = trainSet.getTargetAttribute();
		if (classAttribute.getType() != Attribute.Type.NOMINAL) {
			throw new IllegalArgumentException("Class attribute must be nominal.");
		}
		NominalAttribute clazz = (NominalAttribute) classAttribute;
		int numClasses = clazz.getCardinality();
		
		if (isSparse) {
			SparseDataset sd = getSparseDataset(trainSet, true);
			SparseDesignMatrix sm = createDesignMatrix(sd, groups);
			int[] attrs = sd.attrs;
			int[][] group = sm.group;
			int[][][] indices = sm.indices;
			double[][][] values = sm.values;
			double[] y = new double[sd.y.length];
			double[] cList = sd.cList;
			
			if (numClasses == 2) {
				for (int i = 0; i < y.length; i++) {
					int label = (int) sd.y[i];
					y[i] = label == 0 ? 1 : 0;
				}
				
				GLM glm = buildBinaryClassifier(group, indices, values, y, maxNumIters, lambda);
				
				double[] w = glm.coefficients(0);
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

					GLM binaryClassifier = buildBinaryClassifier(group, indices, values, y, maxNumIters, lambda);

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
			DenseDesignMatrix dm = createDesignMatrix(dd, groups);
			int[] attrs = dd.attrs;
			int[][] group = dm.groups;
			double[][][] x = dm.x;
			double[] y = new double[dd.y.length];
			double[] cList = dd.cList;
			
			if (numClasses == 2) {
				for (int i = 0; i < y.length; i++) {
					int label = (int) dd.y[i];
					y[i] = label == 0 ? 1 : 0;
				}
				
				GLM glm = buildBinaryClassifier(group, x, y, maxNumIters, lambda);

				double[] w = glm.coefficients(0);
				for (int j = 0; j < cList.length; j++) {
					int attIndex = dd.attrs[j];
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

					GLM binaryClassifier = buildBinaryClassifier(group, x, y, maxNumIters, lambda);

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
	 * Builds a group-lasso penalized classifier.
	 * 
	 * @param trainSet the training set.
	 * @param groups the groups of variables.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @return a group-lasso penalized classifier.
	 */
	public GLM buildClassifier(Instances trainSet, List<int[]> groups, int maxNumIters, double lambda) {
		return buildClassifier(trainSet, isSparse(trainSet), groups, maxNumIters, lambda);
	}
	
	/**
	 * Builds group-lasso penalized classifiers.
	 * 
	 * @param trainSet the training set.
	 * @param isSparse <code>true</code> if the training set is treated as sparse.
	 * @param groups the groups of variables.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numLambdas the number of lambdas.
	 * @param minLambdaRatio the minimum lambda is minLambdaRatio * max lambda.
	 * @return group-lasso penalized classifiers.
	 */
	public List<GLM> buildClassifiers(Instances trainSet, boolean isSparse, List<int[]> groups, int maxNumIters,
			int numLambdas, double minLambdaRatio) {
		Attribute classAttribute = trainSet.getTargetAttribute();
		if (classAttribute.getType() != Attribute.Type.NOMINAL) {
			throw new IllegalArgumentException("Class attribute must be nominal.");
		}
		NominalAttribute clazz = (NominalAttribute) classAttribute;
		int numClasses = clazz.getCardinality();
		
		if (isSparse) {
			SparseDataset sd = getSparseDataset(trainSet, true);
			SparseDesignMatrix sm = createDesignMatrix(sd, groups);
			int[] attrs = sd.attrs;
			int[][] group = sm.group;
			int[][][] indices = sm.indices;
			double[][][] values = sm.values;
			double[] y = new double[sd.y.length];
			double[] cList = sd.cList;

			if (numClasses == 2) {
				for (int i = 0; i < y.length; i++) {
					int label = (int) sd.y[i];
					y[i] = label == 0 ? 1 : 0;
				}
				
				List<GLM> glms = buildBinaryClassifiers(sm.group, sm.indices, sm.values, y, maxNumIters, numLambdas,
						minLambdaRatio);

				for (GLM glm : glms) {
					double[] w = glm.coefficients(0);
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

					List<GLM> binaryClassifiers = buildBinaryClassifiers(group, indices, values, y, maxNumIters,
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
			DenseDesignMatrix dm = createDesignMatrix(dd, groups);
			int[] attrs = dd.attrs;
			int[][] group = dm.groups;
			double[][][] x = dm.x;
			double[] y = new double[dd.y.length];
			double[] cList = dd.cList;
			
			if (numClasses == 2) {
				for (int i = 0; i < y.length; i++) {
					int label = (int) dd.y[i];
					y[i] = label == 0 ? 1 : 0;
				}

				List<GLM> glms = buildBinaryClassifiers(dm.groups, dm.x, y, maxNumIters, numLambdas, minLambdaRatio);

				for (GLM glm : glms) {
					double[] w = glm.coefficients(0);
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

					List<GLM> binaryClassifiers = buildBinaryClassifiers(group, x, y, maxNumIters, numLambdas,
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
	 * Builds group-lasso penalized classifiers.
	 * 
	 * @param trainSet the training set.
	 * @param groups the groups of variables.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numLambdas the number of lambdas.
	 * @param minLambdaRatio the minimum lambda is minLambdaRatio * max lambda.
	 * @return group-lasso penalized classifiers.
	 */
	public List<GLM> buildClassifiers(Instances trainSet, List<int[]> groups, int maxNumIters,
			int numLambdas, double minLambdaRatio) {
		return buildClassifiers(trainSet, isSparse(trainSet), groups, maxNumIters, numLambdas, minLambdaRatio);
	}
	
	/**
	 * Builds a group-lasso penalized regressor.
	 * 
	 * @param trainSet the training set.
	 * @param isSparse <code>true</code> if the training set is treated as sparse.
	 * @param groups the groups of variables.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @return a group-lasso penalized regressor.
	 */
	public GLM buildRegressor(Instances trainSet, boolean isSparse, List<int[]> groups, int maxNumIters, double lambda) {
		if (isSparse) {
			SparseDataset sd = getSparseDataset(trainSet, true);
			SparseDesignMatrix sm = createDesignMatrix(sd, groups);
			double[] cList = sd.cList;

			GLM glm = buildRegressor(sm.group, sm.indices, sm.values, sd.y, maxNumIters, lambda);
			
			double[] w = glm.coefficients(0);
			for (int j = 0; j < cList.length; j++) {
				int attIndex = sd.attrs[j];
				w[attIndex] *= cList[j];
			}

			return glm;
		} else {
			DenseDataset dd = getDenseDataset(trainSet, true);
			DenseDesignMatrix dm = createDesignMatrix(dd, groups);
			double[] cList = dd.cList;

			GLM glm = buildRegressor(dm.groups, dm.x, dd.y, maxNumIters, lambda);

			double[] w = glm.coefficients(0);
			for (int j = 0; j < cList.length; j++) {
				int attIndex = dd.attrs[j];
				w[attIndex] *= cList[j];
			}

			return glm;
		}
	}

	/**
	 * Builds a group-lasso penalized regressor.
	 * 
	 * @param trainSet the training set.
	 * @param groups the groups.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @return a group-lasso penalized regressor.
	 */
	public GLM buildRegressor(Instances trainSet, List<int[]> groups, int maxNumIters, double lambda) {
		return buildRegressor(trainSet, isSparse(trainSet), groups, maxNumIters, lambda);
	}

	/**
	 * Builds a group-lasso penalized regressor. The input matrix is grouped by groups. This procedure does not
	 * assume the data is normalized or centered.
	 * 
	 * @param attrs the groups of variables.
	 * @param x the inputs.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @return a group-lasso penalized regressor.
	 */
	public GLM buildRegressor(int[][] attrs, double[][][] x, double[] y, int maxNumIters, double lambda) {
		int p = 0;
		if (attrs.length > 0) {
			for (int[] attr : attrs) {
				p = Math.max(p, StatUtils.max(attr));
			}
			p += 1;
		}
		
		// Backup targets
		double[] rTrain = new double[y.length];
		for (int i = 0; i < rTrain.length; i++) {
			rTrain[i] = y[i];
		}

		double[][] w = new double[attrs.length][];
		double[] tl1 = new double[attrs.length];
		int m = 0;
		for (int j = 0; j < attrs.length; j++) {
			w[j] = new double[x[j].length];
			tl1[j] = lambda * Math.sqrt(w[j].length);
			if (w[j].length > m) {
				m = w[j].length;
			}
		}

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

		boolean[] activeSet = new boolean[x.length];

		double intercept = 0;
		
		// Block coordinate gradient descent
		int iter = 0;
		while (iter < maxNumIters) {

			if (fitIntercept) {
				intercept += OptimUtils.fitIntercept(rTrain);
			}

			boolean activeSetChanged = doOnePass(x, tl1, true, activeSet, w, stepSize, g, gradient, rTrain);

			iter++;

			if (!activeSetChanged || iter > maxNumIters) {
				break;
			}

			for (; iter < maxNumIters; iter++) {

				double prevLoss = GLMOptimUtils.computeGroupLassoLoss(rTrain, w, tl1);
				
				if (fitIntercept) {
					intercept += OptimUtils.fitIntercept(rTrain);
				}
				
				doOnePass(x, tl1, false, activeSet, w, stepSize, g, gradient, rTrain);
				
				double currLoss = GLMOptimUtils.computeGroupLassoLoss(rTrain, w, tl1);

				if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
					break;
				}

				if (verbose) {
					System.out.println("Iteration " + iter + ": " + currLoss);
				}
			}
		}

		if (refit) {
			boolean[] selected = new boolean[attrs.length];
			for (int i = 0; i < selected.length; i++) {
				selected[i] = !ArrayUtils.isConstant(w[i], 0, w[i].length, 0);
			}
			return refitRegressor(p, attrs, selected, x, y, w, maxNumIters);
		} else {
			return getGLM(p, attrs, w, intercept);
		}
	}
	
	/**
	 * Builds a group-lasso penalized regressor on sparse inputs. The input matrix is grouped by groups. This procedure does not
	 * assume the data is normalized or centered.
	 * 
	 * @param groups the groups of variables.
	 * @param indices the indices.
	 * @param values the values.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the lambda.
	 * @return a group-lasso penalized regressor.
	 */
	public GLM buildRegressor(int[][] groups, int[][][] indices, double[][][] values, double[] y, int maxNumIters, double lambda) {
		int p = 0;
		if (groups.length > 0) {
			for (int[] group : groups) {
				p = Math.max(p, StatUtils.max(group));
			}
			p += 1;
		}
		
		// Backup targets
		double[] rTrain = new double[y.length];
		for (int i = 0; i < rTrain.length; i++) {
			rTrain[i] = y[i];
		}

		double[][] w = new double[groups.length][];
		double[] tl1 = new double[groups.length];
		int m = 0;
		for (int j = 0; j < groups.length; j++) {
			w[j] = new double[groups[j].length];
			tl1[j] = lambda * Math.sqrt(w[j].length);
			if (w[j].length > m) {
				m = w[j].length;
			}
		}

		double[] stepSize = new double[groups.length];
		for (int j = 0; j < stepSize.length; j++) {
			double max = 0;
			double[][] block = values[j];
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

		boolean[] activeSet = new boolean[groups.length];

		double intercept = 0;
		int iter = 0;
		while (iter < maxNumIters) {

			if (fitIntercept) {
				intercept += OptimUtils.fitIntercept(rTrain);
			}

			boolean activeSetChanged = doOnePass(indices, values, tl1, true, activeSet, w, stepSize, g, gradient, rTrain);

			iter++;

			if (!activeSetChanged || iter > maxNumIters) {
				break;
			}

			for (; iter < maxNumIters; iter++) {

				double prevLoss = GLMOptimUtils.computeGroupLassoLoss(rTrain, w, tl1);
				
				if (fitIntercept) {
					intercept += OptimUtils.fitIntercept(rTrain);
				}
				
				doOnePass(indices, values, tl1, false, activeSet, w, stepSize, g, gradient, rTrain);
				
				double currLoss = GLMOptimUtils.computeGroupLassoLoss(rTrain, w, tl1);

				if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
					break;
				}

				if (verbose) {
					System.out.println("Iteration " + iter + ": " + currLoss);
				}
			}
		}

		if (refit) {
			boolean[] selected = new boolean[groups.length];
			for (int i = 0; i < selected.length; i++) {
				selected[i] = !ArrayUtils.isConstant(w[i], 0, w[i].length, 0);
			}
			return refitRegressor(p, groups, selected, indices, values, y, w, maxNumIters);
		} else {
			return getGLM(p, groups, w, intercept);
		}
	}
	
	/**
	 * Builds group-lasso penalized regressors.
	 * 
	 * @param trainSet the training set.
	 * @param isSparse <code>true</code> if the training set is treated as sparse.
	 * @param groups the groups of variables.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numLambdas the number of lambdas.
	 * @param minLambdaRatio the minimum lambda is minLambdaRatio * max lambda.
	 * @return group-lasso penalized regressors.
	 */
	public List<GLM> buildRegressors(Instances trainSet, boolean isSparse, List<int[]> groups, int maxNumIters,
			int numLambdas, double minLambdaRatio) {
		if (isSparse) {
			SparseDataset sd = getSparseDataset(trainSet, true);
			SparseDesignMatrix sm = createDesignMatrix(sd, groups);
			double[] cList = sd.cList;
			
			List<GLM> glms = buildRegressors(sm.group, sm.indices, sm.values, sd.y, maxNumIters, numLambdas, minLambdaRatio);
			
			for (GLM glm : glms) {
				double[] w = glm.coefficients(0);
				for (int j = 0; j < cList.length; j++) {
					int attIndex = sd.attrs[j];
					w[attIndex] *= cList[j];
				}
			}
			
			return glms;
		} else {
			DenseDataset dd = getDenseDataset(trainSet, false);
			DenseDesignMatrix dm = createDesignMatrix(dd, groups);
			double[] stdList = dd.stdList;

			List<GLM> glms = buildRegressors(dm.groups, dm.x, dd.y, maxNumIters, numLambdas, minLambdaRatio);

			for (GLM glm : glms) {
				double[] w = glm.coefficients(0);
				for (int j = 0; j < stdList.length; j++) {
					int attIndex = dd.attrs[j];
					w[attIndex] *= stdList[j];
				}
			}
			
			return glms;
		}
	}

	/**
	 * Builds group-lasso penalized regressors.
	 * 
	 * @param trainSet the training set.
	 * @param groups the groups of variables.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numLambdas the number of lambdas.
	 * @param minLambdaRatio the minimum lambda is minLambdaRatio * max lambda.
	 * @return group-lasso penalized regressors.
	 */
	public List<GLM> buildRegressors(Instances trainSet, List<int[]> groups, int maxNumIters,
			int numLambdas, double minLambdaRatio) {
		return buildRegressors(trainSet, isSparse(trainSet), groups, maxNumIters, numLambdas, minLambdaRatio);
	}

	/**
	 * Builds group-lasso penalized regressors for a sequence of regularization parameter lambdas. The input matrix is grouped by groups. This procedure does not
	 * assume the data is normalized or centered.
	 * 
	 * @param groups the groups of variables.
	 * @param x the inputs.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numLambdas the number of lambdas.
	 * @param minLambdaRatio the minimum lambda is minLambdaRatio * max lambda.
	 * @return group-lasso penalized regressors.
	 */
	public List<GLM> buildRegressors(int[][] groups, double[][][] x, double[] y, int maxNumIters,
			int numLambdas, double minLambdaRatio) {
		int p = 0;
		if (groups.length > 0) {
			for (int[] group : groups) {
				p = Math.max(p, StatUtils.max(group));
			}
			p += 1;
		}
		
		// Backup targets
		double[] rTrain = new double[y.length];
		for (int i = 0; i < rTrain.length; i++) {
			rTrain[i] = y[i];
		}

		// Allocate coefficients
		double[][] w = new double[x.length][];
		int m = 0;
		for (int j = 0; j < x.length; j++) {
			w[j] = new double[x[j].length];
			if (w[j].length > m) {
				m = w[j].length;
			}
		}

		double[] g = new double[m];
		double[] gradient = new double[m];

		double[] tl1 = new double[x.length];

		double[] stepSize = new double[x.length];
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

		boolean[] activeSet = new boolean[x.length];

		// Determine max lambda
		double maxLambda = findMaxLambda(x, rTrain, gradient);

		// Dampening factor for lambda
		double alpha = Math.pow(minLambdaRatio, 1.0 / numLambdas);

		// Compute the regularization path
		List<GLM> glms = new ArrayList<>(numLambdas);
		double lambda = maxLambda;
		double intercept = 0;
		Set<ModelStructure> structures = new HashSet<>();
		for (int i = 0; i < numLambdas; i++) {
			// Initialize regularization parameters
			for (int j = 0; j < tl1.length; j++) {
				tl1[j] = lambda * Math.sqrt(w[j].length);
			}

			// Block coordinate gradient descent
			int iter = 0;
			while (iter < maxNumIters) {

				if (fitIntercept) {
					intercept += OptimUtils.fitIntercept(rTrain);
				}

				boolean activeSetChanged = doOnePass(x, tl1, true, activeSet, w, stepSize, g, gradient, rTrain);

				iter++;

				if (!activeSetChanged || iter > maxNumIters) {
					break;
				}

				for (; iter < maxNumIters; iter++) {

					double prevLoss = GLMOptimUtils.computeGroupLassoLoss(rTrain, w, tl1);
					
					if (fitIntercept) {
						intercept += OptimUtils.fitIntercept(rTrain);
					}
					
					doOnePass(x, tl1, false, activeSet, w, stepSize, g, gradient, rTrain);

					double currLoss = GLMOptimUtils.computeGroupLassoLoss(rTrain, w, tl1);

					if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
						break;
					}

					if (verbose) {
						System.out.println("Iteration " + iter + ": " + currLoss);
					}
				}
			}

			lambda *= alpha;
			if (refit) {
				boolean allActivated = true;
				boolean[] selected = new boolean[groups.length];
				for (int j = 0; j < selected.length; j++) {
					selected[j] = !ArrayUtils.isConstant(w[j], 0, w[j].length, 0);
					allActivated = allActivated & selected[j];
				}
				ModelStructure structure = new ModelStructure(selected);
				if (!structures.contains(structure)) {
					GLM glm = refitRegressor(p, groups, selected, x, y, w, maxNumIters);
					glms.add(glm);
					structures.add(structure);
				}
				if (allActivated) {
					break;
				}
			} else {
				GLM glm = getGLM(p, groups, w, intercept);
				glms.add(glm);
			}
		}

		return glms;
	}
	
	/**
	 * Builds group-lasso penalized regressors on sparse inputs for a sequence of regularization parameter lambdas. The input matrix is grouped by groups. This procedure does not
	 * assume the data is normalized or centered.
	 * 
	 * @param groups the groups of variables.
	 * @param indices the indices.
	 * @param values the values.
	 * @param y the targets.
	 * @param maxNumIters the maximum number of iterations.
	 * @param numLambdas the number of lambdas.
	 * @param minLambdaRatio the minimum lambda is minLambdaRatio * max lambda.
	 * @return group-lasso penalized regressors.
	 */
	public List<GLM> buildRegressors(int[][] groups, int[][][] indices, double[][][] values, double[] y, int maxNumIters,
			int numLambdas, double minLambdaRatio) {
		int p = 0;
		if (groups.length > 0) {
			for (int[] group : groups) {
				p = Math.max(p, StatUtils.max(group));
			}
			p += 1;
		}
		
		// Backup targets
		double[] rTrain = new double[y.length];
		for (int i = 0; i < rTrain.length; i++) {
			rTrain[i] = y[i];
		}

		// Allocate coefficients
		double[][] w = new double[groups.length][];
		int m = 0;
		for (int j = 0; j < groups.length; j++) {
			w[j] = new double[groups[j].length];
			if (w[j].length > m) {
				m = w[j].length;
			}
		}

		double[] g = new double[m];
		double[] gradient = new double[m];

		double[] tl1 = new double[groups.length];

		double[] stepSize = new double[groups.length];
		for (int j = 0; j < stepSize.length; j++) {
			double max = 0;
			double[][] block = values[j];
			for (double[] t : block) {
				double l = StatUtils.sumSq(t);
				if (l > max) {
					max = l;
				}
			}
			stepSize[j] = 1.0 / max;
		}

		boolean[] activeSet = new boolean[groups.length];

		// Determine max lambda
		double maxLambda = findMaxLambda(indices, values, rTrain, gradient);

		// Dampening factor for lambda
		double alpha = Math.pow(minLambdaRatio, 1.0 / numLambdas);

		// Compute the regularization path
		List<GLM> glms = new ArrayList<>(numLambdas);
		double lambda = maxLambda;
		double intercept = 0;
		Set<ModelStructure> structures = new HashSet<>();
		for (int i = 0; i < numLambdas; i++) {
			// Initialize regularization parameters
			for (int j = 0; j < tl1.length; j++) {
				tl1[j] = lambda * Math.sqrt(w[j].length);
			}

			// Block coordinate gradient descent
			int iter = 0;
			while (iter < maxNumIters) {

				if (fitIntercept) {
					intercept += OptimUtils.fitIntercept(rTrain);
				}

				boolean activeSetChanged = doOnePass(indices, values, tl1, true, activeSet, w, stepSize, g, gradient, rTrain);

				iter++;

				if (!activeSetChanged || iter > maxNumIters) {
					break;
				}

				for (; iter < maxNumIters; iter++) {

					double prevLoss = GLMOptimUtils.computeGroupLassoLoss(rTrain, w, tl1);
					
					if (fitIntercept) {
						intercept += OptimUtils.fitIntercept(rTrain);
					}
					
					doOnePass(indices, values, tl1, false, activeSet, w, stepSize, g, gradient, rTrain);

					double currLoss = GLMOptimUtils.computeGroupLassoLoss(rTrain, w, tl1);

					if (OptimUtils.isConverged(prevLoss, currLoss, epsilon)) {
						break;
					}

					if (verbose) {
						System.out.println("Iteration " + iter + ": " + currLoss);
					}
				}
			}

			lambda *= alpha;
			if (refit) {
				boolean allActivated = true;
				boolean[] selected = new boolean[groups.length];
				for (int j = 0; j < selected.length; j++) {
					selected[j] = !ArrayUtils.isConstant(w[j], 0, w[j].length, 0);
					allActivated = allActivated & selected[j];
				}
				ModelStructure structure = new ModelStructure(selected);
				if (!structures.contains(structure)) {
					GLM glm = refitRegressor(p, groups, selected, indices, values, y, w, maxNumIters);
					glms.add(glm);
					structures.add(structure);
				}
				if (allActivated) {
					break;
				}
			} else {
				GLM glm = getGLM(p, groups, w, intercept);
				glms.add(glm);
			}
		}

		return glms;
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
	
	protected void computeGradient(double[][] block, double[] rTrain, double[] gradient) {
		for (int i = 0; i < block.length; i++) {
			gradient[i] = VectorUtils.dotProduct(block[i], rTrain) / rTrain.length;
		}
	}

	protected void computeGradient(int[][] index, double[][] block, double[] rTrain, double[] gradient) {
		for (int j = 0; j < block.length; j++) {
			double[] t = block[j];
			int[] idx = index[j];
			gradient[j] = 0;
			for (int i = 0; i < t.length; i++) {
				gradient[j] += rTrain[idx[i]] * t[i];
			}
			gradient[j] /= rTrain.length;
		}
	}

	protected double computePenalty(double[] w, double lambda) {
		return lambda * VectorUtils.l2norm(w);
	}

	protected double computePenalty(double[][] w, double[] lambdas) {
		double penalty = 0;
		for (int i = 0; i < w.length; i++) {
			penalty += computePenalty(w[i], lambdas[i]);
		}
		return penalty;
	}
	
	protected DenseDesignMatrix createDesignMatrix(DenseDataset dd, List<int[]> groupList) {
		int[] attrs = dd.attrs;
		Map<Integer, Integer> attrSet = new HashMap<>();
		for (int j = 0; j < attrs.length; j++) {
			attrSet.put(attrs[j], j);
		}
		List<int[]> gList = new ArrayList<>();
		for (int[] group : groupList) {
			List<Integer> list = new ArrayList<>();
			for (int idx : group) {
				if (attrSet.containsKey(idx)) {
					list.add(attrSet.get(idx));
				}
			}
			if (list.size() > 0) {
				int[] a = new int[list.size()];
				for (int i = 0; i < a.length; i++) {
					a[i] = list.get(i);
				}
				gList.add(a);
			}
		}
		int[][] groups = new int[gList.size()][];
		double[][][] x = new double[gList.size()][][];
		for (int g = 0; g < groups.length; g++) {
			int[] group = gList.get(g);
			groups[g] = group;
			double[][] v = new double[group.length][];
			for (int j = 0; j < group.length; j++) {
				int idx = group[j];
				v[j] = dd.x[idx];
			}
			x[g] = v;
		}
		return new DenseDesignMatrix(groups, x);
	}
	
	protected SparseDesignMatrix createDesignMatrix(SparseDataset sd, List<int[]> groupList) {
		int[] attrs = sd.attrs;
		Map<Integer, Integer> attrSet = new HashMap<>();
		for (int j = 0; j < attrs.length; j++) {
			attrSet.put(attrs[j], j);
		}
		List<int[]> gList = new ArrayList<>();
		for (int[] group : groupList) {
			List<Integer> list = new ArrayList<>();
			for (int idx : group) {
				if (attrSet.containsKey(idx)) {
					list.add(attrSet.get(idx));
				}
			}
			if (list.size() > 0) {
				int[] a = new int[list.size()];
				for (int i = 0; i < a.length; i++) {
					a[i] = list.get(i);
				}
				gList.add(a);
			}
		}
		
		int[][] groups = new int[gList.size()][];
		int[][][] indices = new int[gList.size()][][];
		double[][][] values = new double[gList.size()][][];
		for (int g = 0; g < groups.length; g++) {
			int[] group = gList.get(g);
			groups[g] = group;
			int[][] idx = new int[group.length][];
			double[][] val = new double[group.length][];
			for (int j = 0; j < group.length; j++) {
				int index = group[j];
				idx[j] = sd.indices[index];
				val[j] = sd.values[index];
			}
			indices[g] = idx;
			values[g] = val;
		}
		return new SparseDesignMatrix(groups, indices, values);
	}

	protected boolean doOnePass(double[][][] x, double[] tl1, boolean isFullPass, boolean[] activeSet,
			double[][] w, double[] stepSize, double[] g, double[] gradient, double[] rTrain) {
		boolean activeSetChanged = false;

		for (int k = 0; k < x.length; k++) {
			if (!isFullPass && !activeSet[k]) {
				continue;
			}

			double[][] block = x[k];
			double[] beta = w[k];
			double tk = stepSize[k];

			// Proximal gradient method
			computeGradient(block, rTrain, gradient);

			for (int j = 0; j < beta.length; j++) {
				g[j] = beta[j] + tk * gradient[j];
			}

			double norm = Math.sqrt(StatUtils.sumSq(g, 0, beta.length));
			double lambda = tl1[k] * tk;
			if (norm > lambda) {
				VectorUtils.multiply(g, (1 - lambda / norm));
			} else {
				Arrays.fill(g, 0);
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

	protected boolean doOnePass(double[][][] x, double[] y, double[] tl1, boolean isFullPass, boolean[] activeSet,
			double[][] w, double[] stepSize, double[] g, double[] gradient, double[] pTrain, double[] rTrain) {
		boolean activeSetChanged = false;

		for (int k = 0; k < x.length; k++) {
			if (!isFullPass && !activeSet[k]) {
				continue;
			}

			double[][] block = x[k];
			double[] beta = w[k];
			double tk = stepSize[k];

			// Proximal gradient method
			computeGradient(block, rTrain, gradient);

			for (int j = 0; j < beta.length; j++) {
				g[j] = beta[j] + tk * gradient[j];
			}

			double norm = Math.sqrt(StatUtils.sumSq(g, 0, beta.length));
			double lambda = tl1[k] * tk;
			if (norm > lambda) {
				VectorUtils.multiply(g, (1 - lambda / norm));
			} else {
				Arrays.fill(g, 0);
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
	
	protected boolean doOnePass(int[][][] indices, double[][][] values, double[] tl1, boolean isFullPass,
			boolean[] activeSet, double[][] w, double[] stepSize, double[] g, double[] gradient, double[] rTrain) {
		boolean activeSetChanged = false;

		for (int k = 0; k < values.length; k++) {
			if (!isFullPass && !activeSet[k]) {
				continue;
			}

			double[][] block = values[k];
			int[][] index = indices[k];
			double[] beta = w[k];
			double tk = tl1[k];

			// Proximal gradient method
			computeGradient(index, block, rTrain, gradient);

			for (int j = 0; j < beta.length; j++) {
				g[j] = beta[j] + tk * gradient[j];
			}

			double norm = Math.sqrt(StatUtils.sumSq(g, 0, beta.length));
			double lambda = tl1[k] * tk;
			if (norm > lambda) {
				VectorUtils.multiply(g, (1 - lambda / norm));
			} else {
				VectorUtils.multiply(g, 0);
			}
			
			// Update predictions
			for (int j = 0; j < beta.length; j++) {
				int[] idx = index[j];
				double[] t = block[j];
				double delta = beta[j] - g[j];
				for (int i = 0; i < t.length; i++) {
					rTrain[idx[i]] += delta * t[i];
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

	protected boolean doOnePass(int[][][] indices, int[][] indexUnion, double[][][] values, double[] y, double[] tl1, 
			boolean isFullPass, boolean[] activeSet, double[][] w, double[] stepSize, double[] g, double[] gradient,
			double[] pTrain, double[] rTrain) {
		boolean activeSetChanged = false;

		for (int k = 0; k < values.length; k++) {
			if (!isFullPass && !activeSet[k]) {
				continue;
			}

			int[][] index = indices[k];
			double[][] block = values[k];
			double[] beta = w[k];
			double tk = stepSize[k];

			// Proximal gradient method
			computeGradient(index, block, rTrain, gradient);

			for (int j = 0; j < beta.length; j++) {
				g[j] = beta[j] + tk * gradient[j];
			}

			double norm = Math.sqrt(StatUtils.sumSq(g, 0, beta.length));
			double lambda = tl1[k] * tk;
			if (norm > lambda) {
				VectorUtils.multiply(g, (1 - lambda / norm));
			} else {
				VectorUtils.multiply(g, 0);
			}
			
			// Update predictions
			for (int j = 0; j < beta.length; j++) {
				int[] idx = index[j];
				double[] value = block[j];
				double delta = g[j] - beta[j];
				for (int i = 0; i < value.length; i++) {
					pTrain[idx[i]] += delta * value[i];
				}
			}
			
			int[] idxUnion = indexUnion[k];
			for (int idx : idxUnion) {
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

	protected double findMaxLambda(double[][][] x, double[] rTrain, double[] gradient) {
		double mean = 0;
		if (fitIntercept) {
			mean = OptimUtils.fitIntercept(rTrain);
		}
		double maxLambda = 0;
		for (double[][] block : x) {
			computeGradient(block, rTrain, gradient);
			double t = Math.sqrt(StatUtils.sumSq(gradient, 0, block.length)) / Math.sqrt(block.length);
			if (t > maxLambda) {
				maxLambda = t;
			}
		}
		if (fitIntercept) {
			VectorUtils.add(rTrain, mean);
		}
		return maxLambda;
	}

	protected double findMaxLambda(double[][][] x, double[] y, double[] pTrain, double[] rTrain, double[] gradient) {
		if (fitIntercept) {
			OptimUtils.fitIntercept(pTrain, rTrain, y);
		}
		double maxLambda = 0;
		for (double[][] block : x) {
			computeGradient(block, rTrain, gradient);
			double t = Math.sqrt(StatUtils.sumSq(gradient, 0, block.length)) / Math.sqrt(block.length);
			if (t > maxLambda) {
				maxLambda = t;
			}
		}
		if (fitIntercept) {
			Arrays.fill(pTrain, 0);
			OptimUtils.computePseudoResidual(pTrain, y, rTrain);
		}
		return maxLambda;
	}
	
	protected double findMaxLambda(int[][][] indices, double[][][] values, double[] rTrain, double[] gradient) {
		double mean = 0;
		if (fitIntercept) {
			mean = OptimUtils.fitIntercept(rTrain);
		}
		double maxLambda = 0;
		for (int g = 0; g < values.length; g++) {
			int[][] index = indices[g];
			double[][] block = values[g];
			computeGradient(index, block, rTrain, gradient);
			double t = Math.sqrt(StatUtils.sumSq(gradient, 0, block.length)) / Math.sqrt(block.length);
			if (t > maxLambda) {
				maxLambda = t;
			}
		}
		if (fitIntercept) {
			VectorUtils.add(rTrain, mean);
		}
		return maxLambda;
	}
	
	protected double findMaxLambda(int[][][] indices, double[][][] values, double[] y, double[] pTrain,
			double[] rTrain, double[] gradient) {
		if (fitIntercept) {
			OptimUtils.fitIntercept(pTrain, rTrain, y);
		}
		double maxLambda = 0;
		for (int j = 0; j < values.length; j++) {
			int[][] index = indices[j];
			double[][] block = values[j];
			computeGradient(index, block, rTrain, gradient);
			double t = Math.sqrt(StatUtils.sumSq(gradient, 0, block.length)) / Math.sqrt(block.length);
			if (t > maxLambda) {
				maxLambda = t;
			}
		}
		if (fitIntercept) {
			Arrays.fill(pTrain, 0);
			OptimUtils.computePseudoResidual(pTrain, y, rTrain);
		}
		return maxLambda;
	}
	
	protected GLM getGLM(int p, int[][] attrs, boolean[] selected, double[] coef, double intercept) {
		GLM glm = new GLM(p);
		int k = 0;
		double[] w = glm.coefficients(0);
		for (int g = 0; g < attrs.length; g++) {
			if (selected[g]) {
				int[] attr = attrs[g];
				for (int attIndex : attr) {
					w[attIndex] = coef[k++];
				}
			}
		}
		glm.intercept[0] = intercept;
		return glm;
	}
	
	protected GLM getGLM(int p, int[][] attrs, double[][] coef, double intercept) {
		GLM glm = new GLM(p);
		double[] w = glm.coefficients(0);
		for (int g = 0; g < attrs.length; g++) {
			int[] attr = attrs[g];
			double[] beta = coef[g];
			for (int j = 0; j < attr.length; j++) {
				w[attr[j]] = beta[j];
			}
		}
		glm.intercept[0] = intercept;
		return glm;
	}
	
	protected GLM refitClassifier(int p, int[][] groups, boolean[] selected, double[][][] x, double[] y, double[][] w, int maxNumIters) {
		List<double[]> xList = new ArrayList<>();
		for (int g = 0; g < selected.length; g++) {
			if (selected[g]) {
				double[][] t = x[g];
				for (int j = 0; j < t.length; j++) {
					xList.add(t[j]);
				}
			}
		}

		double[][] xNew = new double[xList.size()][];
		for (int i = 0; i < xNew.length; i++) {
			xNew[i] = xList.get(i);
		}

		int[] attrs = new int[xNew.length];
		for (int i = 0; i < attrs.length; i++) {
			attrs[i] = i;
		}

		RidgeLearner ridgeLearner = new RidgeLearner();
		ridgeLearner.setVerbose(verbose);
		ridgeLearner.setEpsilon(epsilon);
		ridgeLearner.fitIntercept(fitIntercept);
		// A ridge regression with very small regularization parameter
		// This often improves stability a lot
		GLM glm = ridgeLearner.buildBinaryClassifier(attrs, xNew, y, maxNumIters, 1e-8);
		return getGLM(p, groups, selected, glm.coefficients(0), glm.intercept(0));
	}
	
	protected GLM refitClassifier(int p, int[][] groups, boolean[] selected, int[][][] indices, double[][][] values, double[] y, double[][] w, int maxNumIters) {
		List<int[]> iList = new ArrayList<>();
		List<double[]> vList = new ArrayList<>();
		for (int g = 0; g < selected.length; g++) {
			if (selected[g]) {
				int[][] iBlock = indices[g];
				double[][] vBlock = values[g];
				for (int j = 0; j < vBlock.length; j++) {
					iList.add(iBlock[j]);
					vList.add(vBlock[j]);
				}
			}
		}

		int[][] idxNew = new int[iList.size()][];
		for (int i = 0; i < idxNew.length; i++) {
			idxNew[i] = iList.get(i);
		}
		double[][] valNew = new double[vList.size()][];
		for (int i = 0; i < valNew.length; i++) {
			valNew[i] = vList.get(i);
		}

		int[] attrs = new int[valNew.length];
		for (int i = 0; i < attrs.length; i++) {
			attrs[i] = i;
		}

		RidgeLearner ridgeLearner = new RidgeLearner();
		ridgeLearner.setVerbose(verbose);
		ridgeLearner.setEpsilon(epsilon);
		ridgeLearner.fitIntercept(fitIntercept);
		// A ridge regression with very small regularization parameter
		// This often improves stability a lot
		GLM glm = ridgeLearner.buildBinaryClassifier(attrs, idxNew, valNew, y, maxNumIters, 1e-8);
		return getGLM(p, groups, selected, glm.coefficients(0), glm.intercept(0));
	}
	
	protected GLM refitRegressor(int p, int[][] attrs, boolean[] selected, double[][][] x, double[] y, double[][] w, int maxNumIters) {
		List<double[]> xList = new ArrayList<>();
		for (int g = 0; g < selected.length; g++) {
			if (selected[g]) {
				double[][] t = x[g];
				for (int j = 0; j < t.length; j++) {
					xList.add(t[j]);
				}
			}
		}

		if (xList.size() == 0) {
			if (fitIntercept) {
				double intercept = StatUtils.mean(y);
				GLM glm = new GLM();
				glm.intercept[0] = intercept;
				return glm;
			} else {
				return new GLM();
			}
		}

		double[][] xNew = new double[xList.size()][];
		for (int i = 0; i < xNew.length; i++) {
			xNew[i] = xList.get(i);
		}

		int[] attrsNew = new int[xNew.length];
		for (int j = 0; j < attrsNew.length; j++) {
			attrsNew[j] = j;
		}

		RidgeLearner ridgeLearner = new RidgeLearner();
		ridgeLearner.setVerbose(verbose);
		ridgeLearner.setEpsilon(epsilon);
		ridgeLearner.fitIntercept(fitIntercept);
		// A ridge regression with very small regularization parameter
		// This often improves stability a lot
		GLM glm = ridgeLearner.buildRegressor(attrsNew, xNew, y, maxNumIters, 1e-8);
		return getGLM(p, attrs, selected, glm.coefficients(0), glm.intercept(0));
	}

	protected GLM refitRegressor(int p, int[][] groups, boolean[] selected, int[][][] indices, double[][][] values, double[] y, double[][] w, int maxNumIters) {
		List<int[]> iList = new ArrayList<>();
		List<double[]> vList = new ArrayList<>();
		for (int g = 0; g < selected.length; g++) {
			if (selected[g]) {
				int[][] iBlock = indices[g];
				double[][] vBlock = values[g];
				for (int j = 0; j < vBlock.length; j++) {
					iList.add(iBlock[j]);
					vList.add(vBlock[j]);
				}
			}
		}

		if (vList.size() == 0) {
			if (fitIntercept) {
				double intercept = StatUtils.mean(y);
				GLM glm = new GLM();
				glm.intercept[0] = intercept;
				return glm;
			} else {
				return new GLM();
			}
		}

		int[][] idxNew = new int[iList.size()][];
		for (int i = 0; i < idxNew.length; i++) {
			idxNew[i] = iList.get(i);
		}
		double[][] valNew = new double[vList.size()][];
		for (int i = 0; i < valNew.length; i++) {
			valNew[i] = vList.get(i);
		}

		int[] attrs = new int[valNew.length];
		for (int i = 0; i < attrs.length; i++) {
			attrs[i] = i;
		}

		RidgeLearner ridgeLearner = new RidgeLearner();
		ridgeLearner.setVerbose(verbose);
		ridgeLearner.setEpsilon(epsilon);
		ridgeLearner.fitIntercept(fitIntercept);
		// A ridge regression with very small regularization parameter
		// This often improves stability a lot
		GLM glm = ridgeLearner.buildRegressor(attrs, idxNew, valNew, y, maxNumIters, 1e-8);
		return getGLM(p, groups, selected, glm.coefficients(0), glm.intercept(0));
	}

}
