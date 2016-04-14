package mltk.predictor.tree.ensemble.brt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.cmdline.options.LearnerOptions;
import mltk.core.Attribute;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.NominalAttribute;
import mltk.core.io.InstancesReader;
import mltk.predictor.evaluation.Metric;
import mltk.predictor.io.PredictorWriter;
import mltk.predictor.tree.RTree;
import mltk.predictor.tree.TreeLearner;
import mltk.predictor.tree.RegressionTreeLearner.Mode;
import mltk.util.MathUtils;
import mltk.util.OptimUtils;
import mltk.util.Permutation;
import mltk.util.Random;

/**
 * Class for logit boost learner.
 * 
 * <p>
 * Reference:<br>
 * P. Li. Robust logitboost and adaptive base class (abc) logitboost. In <i>Proceedings of the 26th Conference on
 * Uncertainty in Artificial Intelligence (UAI)</i>, Catalina Island, CA, USA, 2010.
 * </p>
 * 
 * @author Yin Lou
 * 
 */
public class LogitBoostLearner extends BRTLearner {
	
	static class Options extends LearnerOptions {

		@Argument(name = "-c", description = "max number of leaves (default: 100)")
		int maxNumLeaves = 100;

		@Argument(name = "-m", description = "maximum number of iterations", required = true)
		int maxNumIters = -1;

		@Argument(name = "-s", description = "seed of the random number generator (default: 0)")
		long seed = 0L;

		@Argument(name = "-l", description = "learning rate (default: 0.01)")
		double learningRate = 0.01;

	}

	/**
	 * Trains an additive logistic regression.
	 * 
	 * <pre>
	 * Usage: mltk.predictor.tree.ensemble.brt.LogitBoostLearner
	 * -t	train set path
	 * -m	maximum number of iterations
	 * [-r]	attribute file path
	 * [-o]	output model path
	 * [-V]	verbose (default: true)
	 * [-c]	max number of leaves (default: 100)
	 * [-s]	seed of the random number generator (default: 0)
	 * [-l]	learning rate (default: 0.01)
	 * </pre>
	 * 
	 * @param args the command line arguments.
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(LogitBoostLearner.class, opts);
		try {
			parser.parse(args);
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}

		Random.getInstance().setSeed(opts.seed);

		Instances trainSet = InstancesReader.read(opts.attPath, opts.trainPath);
		
		RobustRegressionTreeLearner rtLearner = new RobustRegressionTreeLearner();
		rtLearner.setConstructionMode(Mode.NUM_LEAVES_LIMITED);
		rtLearner.setMaxNumLeaves(opts.maxNumLeaves);

		LogitBoostLearner learner = new LogitBoostLearner();
		learner.setLearningRate(opts.learningRate);
		learner.setMaxNumIters(opts.maxNumIters);
		learner.setVerbose(opts.verbose);
		learner.setTreeLearner(rtLearner);

		long start = System.currentTimeMillis();
		BRT brt = learner.build(trainSet);
		long end = System.currentTimeMillis();
		System.out.println("Time: " + (end - start) / 1000.0 + " (s).");

		if (opts.outputModelPath != null) {
			PredictorWriter.write(brt, opts.outputModelPath);
		}
	}

	/**
	 * Constructor.
	 */
	public LogitBoostLearner() {
		verbose = false;
		maxNumIters = 3500;
		learningRate = 0.01;
		alpha = 1;
		
		RobustRegressionTreeLearner rtLearner = new RobustRegressionTreeLearner();
		rtLearner.setConstructionMode(Mode.NUM_LEAVES_LIMITED);
		rtLearner.setMaxNumLeaves(100);
		
		treeLearner = rtLearner;
	}
	
	@Override
	public BRT build(Instances instances) {
		return buildClassifier(instances, maxNumIters);
	}

	/**
	 * Builds a classifier.
	 * 
	 * @param trainSet the training set.
	 * @param validSet the validation set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param metric the metric to optimize for on the validation set.
	 * @return a classifier.
	 */
	public BRT buildBinaryClassifier(Instances trainSet, Instances validSet, int maxNumIters, Metric metric) {
		Attribute classAttribute = trainSet.getTargetAttribute();
		if (classAttribute.getType() != Attribute.Type.NOMINAL) {
			throw new IllegalArgumentException("Class attribute must be nominal.");
		}
		NominalAttribute clazz = (NominalAttribute) classAttribute;
		if (clazz.getCardinality() != 2) {
			throw new IllegalArgumentException("Only binary classification is accepted.");
		}

		BRT brt = new BRT(1);

		List<Attribute> attributes = trainSet.getAttributes();
		int limit = (int) (attributes.size() * alpha);
		int[] indices = new int[limit];
		Permutation perm = new Permutation(attributes.size());
		if (alpha < 1) {
			perm.permute();
		}

		// Backup targets and weights
		double[] targetTrain = new double[trainSet.size()];
		double[] weightTrain = new double[targetTrain.length];
		for (int i = 0; i < targetTrain.length; i++) {
			Instance instance = trainSet.get(i);
			targetTrain[i] = instance.getTarget();
			weightTrain[i] = instance.getWeight();
		}

		// Initialization
		double[] predTrain = new double[targetTrain.length];
		double[] probTrain = new double[targetTrain.length];
		computeProbabilities(predTrain, probTrain);
		double[] rTrain = new double[targetTrain.length];
		OptimUtils.computePseudoResidual(predTrain, targetTrain, rTrain);
		double[] predValid = new double[validSet.size()];

		List<Double> measureList = new ArrayList<>(maxNumIters);
		for (int iter = 0; iter < maxNumIters; iter++) {
			// Prepare attributes
			if (alpha < 1) {
				int[] a = perm.getPermutation();
				for (int i = 0; i < indices.length; i++) {
					indices[i] = a[i];
				}
				Arrays.sort(indices);
				List<Attribute> attList = trainSet.getAttributes(indices);
				trainSet.setAttributes(attList);
			}
			
			// Prepare training set
			for (int i = 0; i < targetTrain.length; i++) {
				Instance instance = trainSet.get(i);
				double prob = probTrain[i];
				double w = prob * (1 - prob);
				instance.setTarget(rTrain[i] * weightTrain[i]);
				instance.setWeight(w * weightTrain[i]);
			}
			
			RTree rt = (RTree) treeLearner.build(trainSet);
			if (learningRate != 1) {
				rt.multiply(learningRate);
			}
			brt.trees[0].add(rt);
			
			for (int i = 0; i < predTrain.length; i++) {
				double pred = rt.regress(trainSet.get(i));
				predTrain[i] += pred;
			}
			for (int i = 0; i < predValid.length; i++) {
				double pred = rt.regress(validSet.get(i));
				predValid[i] += pred;
			}

			if (alpha < 1) {
				// Restore attributes
				trainSet.setAttributes(attributes);
			}

			// Update residuals and probabilities
			OptimUtils.computePseudoResidual(predTrain, targetTrain, rTrain);
			computeProbabilities(predTrain, probTrain);
			
			double measure = metric.eval(predValid, validSet);
			measureList.add(measure);
			if (verbose) {
				System.out.println("Iteration " + iter + ": " + measure);
			}
		}
		
		// Search the best model on validation set
		int idx = metric.searchBestMetricValueIndex(measureList);
		for (int i = brt.trees[0].size() - 1; i > idx; i--) {
			brt.trees[0].removeLast();
		}

		// Restore targets and weights
		for (int i = 0; i < targetTrain.length; i++) {
			Instance instance = trainSet.get(i);
			instance.setTarget(targetTrain[i]);
			instance.setWeight(weightTrain[i]);
		}

		return brt;
	}

	/**
	 * Builds a classifier.
	 * 
	 * @param trainSet the training set.
	 * @param validSet the validation set.
	 * @param maxNumIters the maximum number of iterations.
	 * @return a classifier.
	 */
	public BRT buildClassifier(Instances trainSet, Instances validSet, int maxNumIters) {
		Attribute classAttribute = trainSet.getTargetAttribute();
		if (classAttribute.getType() != Attribute.Type.NOMINAL) {
			throw new IllegalArgumentException("Class attribute must be nominal.");
		}
		NominalAttribute clazz = (NominalAttribute) classAttribute;
		final int numClasses = clazz.getStates().length;
		final double l = learningRate * (numClasses - 1.0) / numClasses;

		BRT brt = new BRT(numClasses);

		List<Attribute> attributes = trainSet.getAttributes();
		int limit = (int) (attributes.size() * alpha);
		int[] indices = new int[limit];
		Permutation perm = new Permutation(attributes.size());
		if (alpha < 1) {
			perm.permute();
		}

		// Backup targets and weights
		double[] targetTrain = new double[trainSet.size()];
		double[] weightTrain = new double[targetTrain.length];
		for (int i = 0; i < targetTrain.length; i++) {
			Instance instance = trainSet.get(i);
			targetTrain[i] = instance.getTarget();
			weightTrain[i] = instance.getWeight();
		}
		double[] targetValid = new double[validSet.size()];
		for (int i = 0; i < targetValid.length; i++) {
			targetValid[i] = validSet.get(i).getTarget();
		}

		// Initialization
		double[][] predTrain = new double[numClasses][targetTrain.length];
		double[][] probTrain = new double[numClasses][targetTrain.length];
		int[][] rTrain = new int[numClasses][targetTrain.length];
		for (int k = 0; k < numClasses; k++) {
			int[] rkTrain = rTrain[k];
			double[] probkTrain = probTrain[k];
			for (int i = 0; i < rkTrain.length; i++) {
				rkTrain[i] = MathUtils.indicator(targetTrain[i] == k);
				probkTrain[i] = 1.0 / numClasses;
			}
		}
		double[][] predValid = new double[numClasses][validSet.size()];

		for (int iter = 0; iter < maxNumIters; iter++) {
			// Prepare attributes
			if (alpha < 1) {
				int[] a = perm.getPermutation();
				for (int i = 0; i < indices.length; i++) {
					indices[i] = a[i];
				}
				Arrays.sort(indices);
				List<Attribute> attList = trainSet.getAttributes(indices);
				trainSet.setAttributes(attList);
			}

			for (int k = 0; k < numClasses; k++) {
				// Prepare training set
				int[] rkTrain = rTrain[k];
				double[] probkTrain = probTrain[k];
				for (int i = 0; i < targetTrain.length; i++) {
					Instance instance = trainSet.get(i);
					double pk = probkTrain[i];
					double t = rkTrain[i] - pk;
					double w = pk * (1 - pk);
					instance.setTarget(t * weightTrain[i]);
					instance.setWeight(w * weightTrain[i]);
				}

				RTree rt = (RTree) treeLearner.build(trainSet);
				rt.multiply(l);
				brt.trees[k].add(rt);

				double[] predkTrain = predTrain[k];
				for (int i = 0; i < predkTrain.length; i++) {
					double p = rt.regress(trainSet.get(i));
					predkTrain[i] += p;
				}

				double[] predkValid = predValid[k];
				for (int i = 0; i < predkValid.length; i++) {
					double p = rt.regress(validSet.get(i));
					predkValid[i] += p;
				}
			}

			if (alpha < 1) {
				// Restore attributes
				trainSet.setAttributes(attributes);
			}

			// Update probabilities
			computeProbabilities(predTrain, probTrain);

			if (verbose) {
				double error = 0;
				for (int i = 0; i < targetValid.length; i++) {
					double p = 0;
					double max = Double.NEGATIVE_INFINITY;
					for (int k = 0; k < numClasses; k++) {
						if (predValid[k][i] > max) {
							max = predValid[k][i];
							p = k;
						}
					}
					if (p != targetValid[i]) {
						error++;
					}
				}
				error /= targetValid.length;
				System.out.println("Iteration " + iter + ": " + error);
			}
		}

		// Restore targets and weights
		for (int i = 0; i < targetTrain.length; i++) {
			Instance instance = trainSet.get(i);
			instance.setTarget(targetTrain[i]);
			instance.setWeight(weightTrain[i]);
		}

		return brt;
	}

	/**
	 * Builds a classifier.
	 * 
	 * @param trainSet the training set.
	 * @param maxNumIters the maximum number of iterations.
	 * @return a classifier.
	 */
	public BRT buildClassifier(Instances trainSet, int maxNumIters) {
		Attribute classAttribute = trainSet.getTargetAttribute();
		if (classAttribute.getType() != Attribute.Type.NOMINAL) {
			throw new IllegalArgumentException("Class attribute must be nominal.");
		}
		NominalAttribute clazz = (NominalAttribute) classAttribute;
		final int numClasses = clazz.getStates().length;
		final int n = trainSet.size();
		final double l = learningRate * (numClasses - 1.0) / numClasses;

		BRT brt = new BRT(numClasses);

		List<Attribute> attributes = trainSet.getAttributes();
		int limit = (int) (attributes.size() * alpha);
		int[] indices = new int[limit];
		Permutation perm = new Permutation(attributes.size());
		if (alpha < 1) {
			perm.permute();
		}

		// Backup targets and weights
		double[] target = new double[n];
		double[] weight = new double[n];
		for (int i = 0; i < n; i++) {
			Instance instance = trainSet.get(i);
			target[i] = instance.getTarget();
			weight[i] = instance.getWeight();
		}

		// Initialization
		double[][] predTrain = new double[numClasses][n];
		double[][] probTrain = new double[numClasses][n];
		int[][] rTrain = new int[numClasses][n];
		for (int k = 0; k < numClasses; k++) {
			int[] rkTrain = rTrain[k];
			double[] probkTrain = probTrain[k];
			for (int i = 0; i < n; i++) {
				rkTrain[i] = MathUtils.indicator(target[i] == k);
				probkTrain[i] = 1.0 / numClasses;
			}
		}

		for (int iter = 0; iter < maxNumIters; iter++) {
			// Prepare attributes
			if (alpha < 1) {
				int[] a = perm.getPermutation();
				for (int i = 0; i < indices.length; i++) {
					indices[i] = a[i];
				}
				Arrays.sort(indices);
				List<Attribute> attList = trainSet.getAttributes(indices);
				trainSet.setAttributes(attList);
			}

			for (int k = 0; k < numClasses; k++) {
				// Prepare training set
				int[] rkTrain = rTrain[k];
				double[] probkTrain = probTrain[k];
				for (int i = 0; i < n; i++) {
					Instance instance = trainSet.get(i);
					double pk = probkTrain[i];
					double t = rkTrain[i] - pk;
					double w = pk * (1 - pk);
					instance.setTarget(t * weight[i]);
					instance.setWeight(w * weight[i]);
				}

				RTree rt = (RTree) treeLearner.build(trainSet);
				rt.multiply(l);
				brt.trees[k].add(rt);

				double[] predkTrain = predTrain[k];
				for (int i = 0; i < n; i++) {
					double p = rt.regress(trainSet.get(i));
					predkTrain[i] += p;
				}
			}

			if (alpha < 1) {
				// Restore attributes
				trainSet.setAttributes(attributes);
			}

			// Update probabilities
			computeProbabilities(predTrain, probTrain);

			if (verbose) {
				double error = 0;
				for (int i = 0; i < n; i++) {
					double p = 0;
					double maxProb = -1;
					for (int k = 0; k < numClasses; k++) {
						if (probTrain[k][i] > maxProb) {
							maxProb = probTrain[k][i];
							p = k;
						}
					}
					if (p != target[i]) {
						error++;
					}
				}
				error /= n;
				System.out.println("Iteration " + iter + ": " + error);
			}
		}

		// Restore targets and weights
		for (int i = 0; i < n; i++) {
			Instance instance = trainSet.get(i);
			instance.setTarget(target[i]);
			instance.setWeight(weight[i]);
		}

		return brt;
	}
	
	@Override
	public void setTreeLearner(TreeLearner treeLearner) {
		if (!(treeLearner instanceof RobustRegressionTreeLearner)) {
			throw new IllegalArgumentException("Only robust regression tree learners are accepted");
		}
		this.treeLearner = treeLearner;
	}

	protected void computeProbabilities(double[] pred, double[] prob) {
		for (int i = 0; i < pred.length; i++) {
			prob[i] = MathUtils.sigmoid(pred[i]);
		}
	}
	
	protected void computeProbabilities(double[][] pred, double[][] prob) {
		for (int i = 0; i < pred[0].length; i++) {
			double max = Double.NEGATIVE_INFINITY;
			for (int k = 0; k < pred.length; k++) {
				if (max < pred[k][i]) {
					max = pred[k][i];
				}
			}
			double sum = 0;
			for (int k = 0; k < pred.length; k++) {
				double p = Math.exp(pred[k][i] - max);
				prob[k][i] = p;
				sum += p;
			}
			for (int k = 0; k < pred.length; k++) {
				prob[k][i] /= sum;
			}
		}
	}

}
