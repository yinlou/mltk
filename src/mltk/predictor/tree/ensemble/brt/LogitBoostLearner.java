package mltk.predictor.tree.ensemble.brt;

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
import mltk.predictor.Learner;
import mltk.predictor.io.PredictorWriter;
import mltk.predictor.tree.RegressionTree;
import mltk.predictor.tree.RegressionTreeLearner.Mode;
import mltk.util.MathUtils;
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
public class LogitBoostLearner extends Learner {
	
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
	 * <p>
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
	 * </p>
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

		LogitBoostLearner logitBoostLearner = new LogitBoostLearner();
		logitBoostLearner.setLearningRate(opts.learningRate);
		logitBoostLearner.setMaxNumIters(opts.maxNumIters);
		logitBoostLearner.setMaxNumLeaves(opts.maxNumLeaves);
		logitBoostLearner.setVerbose(true);

		long start = System.currentTimeMillis();
		BRT brt = logitBoostLearner.build(trainSet);
		long end = System.currentTimeMillis();
		System.out.println("Time: " + (end - start) / 1000.0);

		if (opts.outputModelPath != null) {
			PredictorWriter.write(brt, opts.outputModelPath);
		}
	}

	private int maxNumIters;
	private int maxNumLeaves;
	private double learningRate;
	private double alpha;

	/**
	 * Constructor.
	 */
	public LogitBoostLearner() {
		verbose = false;
		maxNumIters = 3500;
		maxNumLeaves = 100;
		learningRate = 1;
		alpha = 1;
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
	 * Sets the maximum number of iterations.
	 * 
	 * @param maxNumIters the maximum number of iterations.
	 */
	public void setMaxNumIters(int maxNumIters) {
		this.maxNumIters = maxNumIters;
	}

	/**
	 * Returns the learning rate.
	 * 
	 * @return the learning rate.
	 */
	public double getLearningRate() {
		return learningRate;
	}

	/**
	 * Sets the learning rate.
	 * 
	 * @param learningRate the learning rate.
	 */
	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	/**
	 * Returns the maximum number of leaves.
	 * 
	 * @return the maximum number of leaves.
	 */
	public int getMaxNumLeaves() {
		return maxNumLeaves;
	}

	/**
	 * Sets the maximum number of leaves.
	 * 
	 * @param maxNumLeaves the maximum number of leaves.
	 */
	public void setMaxNumLeaves(int maxNumLeaves) {
		this.maxNumLeaves = maxNumLeaves;
	}

	/**
	 * Builds a classifier.
	 * 
	 * @param trainSet the training set.
	 * @param validSet the validation set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param maxNumLeaves the maximum number of leaves.
	 * @return a classifier.
	 */
	public BRT buildClassifier(Instances trainSet, Instances validSet, int maxNumIters, int maxNumLeaves) {
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

		RobustRegressionTreeLearner rtLearner = new RobustRegressionTreeLearner();
		rtLearner.setConstructionMode(Mode.NUM_LEAVES_LIMITED);
		rtLearner.setMaxNumLeaves(maxNumLeaves);

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

				RegressionTree rt = rtLearner.build(trainSet);
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
			predictProbabilities(predTrain, probTrain);

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
	 * @param maxNumLeaves the maximum number of leaves.
	 * @return a classifier.
	 */
	public BRT buildClassifier(Instances trainSet, int maxNumIters, int maxNumLeaves) {
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

		RobustRegressionTreeLearner rtLearner = new RobustRegressionTreeLearner();
		rtLearner.setConstructionMode(Mode.NUM_LEAVES_LIMITED);
		rtLearner.setMaxNumLeaves(maxNumLeaves);

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

				RegressionTree rt = rtLearner.build(trainSet);
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
			predictProbabilities(predTrain, probTrain);

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

	protected void predictProbabilities(double[][] pred, double[][] prob) {
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

	@Override
	public BRT build(Instances instances) {
		return buildClassifier(instances, maxNumIters, maxNumLeaves);
	}

}
