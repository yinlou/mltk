package mltk.predictor.tree.ensemble.brt;

import java.util.Arrays;
import java.util.List;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.cmdline.options.LearnerOptions;
import mltk.core.Attribute;
import mltk.core.Instances;
import mltk.core.io.InstancesReader;
import mltk.predictor.BaggedEnsemble;
import mltk.predictor.BaggedEnsembleLearner;
import mltk.predictor.Bagging;
import mltk.predictor.BoostedEnsemble;
import mltk.predictor.Learner;
import mltk.predictor.Predictor;
import mltk.predictor.io.PredictorWriter;
import mltk.predictor.tree.RegressionTree;
import mltk.predictor.tree.RegressionTreeLearner;
import mltk.predictor.tree.RegressionTreeLearner.Mode;
import mltk.util.Permutation;
import mltk.util.Random;
import mltk.util.StatUtils;

/**
 * Class for least-squares boost learner.
 *
 * @author Yin Lou
 *
 */
public class LSBoostLearner extends Learner {
	
	class Options extends LearnerOptions {

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
	 * Usage: mltk.predictor.tree.ensemble.brt.LSBoostLearner
	 * -t	train set path
	 * -m	maximum number of iterations
	 * [-v]	valid set path
	 * [-e]	evaluation metric (default: default metric of task)
	 * [-r]	attribute file path
	 * [-o]	output model path
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
		LSBoostLearner learner = new LSBoostLearner();
		Options opts = learner.new Options();
		CmdLineParser parser = new CmdLineParser(LSBoostLearner.class, opts);
		try {
			parser.parse(args);
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}

		Random.getInstance().setSeed(opts.seed);

		Instances trainSet = InstancesReader.read(opts.attPath, opts.trainPath);

		learner.setLearningRate(opts.learningRate);
		learner.setMaxNumIters(opts.maxNumIters);
		learner.setMaxNumLeaves(opts.maxNumLeaves);
		learner.setVerbose(opts.verbose);

		long start = System.currentTimeMillis();
		BRT brt = learner.build(trainSet);
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
	public LSBoostLearner() {
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
	 * Builds a regressor.
	 *
	 * @param trainSet the training set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param maxNumLeaves the maximum number of leaves.
	 * @return a regressor.
	 */
	public BRT buildRegressor(Instances trainSet, int maxNumIters, int maxNumLeaves) {
		BRT brt = new BRT(1);

		List<Attribute> attributes = trainSet.getAttributes();
		int limit = (int) (attributes.size() * alpha);
		int[] indices = new int[limit];
		Permutation perm = new Permutation(attributes.size());
		perm.permute();

		// Backup targets
		double[] target = new double[trainSet.size()];
		for (int i = 0; i < target.length; i++) {
			target[i] = trainSet.get(i).getTarget();
		}

		RegressionTreeLearner rtLearner = new RegressionTreeLearner();
		rtLearner.setConstructionMode(Mode.NUM_LEAVES_LIMITED);
		rtLearner.setMaxNumLeaves(maxNumLeaves);

		double[] rTrain = new double[trainSet.size()];
		for (int i = 0; i < rTrain.length; i++) {
			rTrain[i] = target[i];
		}

		// Gradient boosting
		for (int iter = 0; iter < maxNumIters; iter++) {
			// Prepare training set
			if (alpha < 1) {
				int[] a = perm.getPermutation();
				for (int i = 0; i < indices.length; i++) {
					indices[i] = a[i];
				}
				Arrays.sort(indices);
				List<Attribute> attList = trainSet.getAttributes(indices);
				trainSet.setAttributes(attList);
			}
			for (int i = 0; i < rTrain.length; i++) {
				trainSet.get(i).setTarget(rTrain[i]);
			}

			RegressionTree rt = rtLearner.build(trainSet);
			if (learningRate != 1) {
				rt.multiply(learningRate);
			}
			brt.trees[0].add(rt);

			if (alpha < 1) {
				// Restore attributes
				trainSet.setAttributes(attributes);
			}

			// Update residuals
			for (int i = 0; i < rTrain.length; i++) {
				double pred = rt.regress(trainSet.get(i));
				rTrain[i] -= pred;
			}

			if (verbose) {
				double rmse = StatUtils.rms(rTrain);
				System.out.println("Iteration " + iter + ": " + rmse);
			}
		}

		// Restore targets
		for (int i = 0; i < target.length; i++) {
			trainSet.get(i).setTarget(target[i]);
		}

		return brt;
	}

	/**
	 * Builds a regressor.
	 *
	 * @param trainSet the training set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param maxNumLeaves the maximum number of leaves.
	 * @param baggingIters the bagging iterations.
	 * @return a regressor.
	 */
	public BoostedEnsemble buildRegressor(Instances trainSet, int maxNumIters, int maxNumLeaves, int baggingIters) {
		BoostedEnsemble brt = new BoostedEnsemble();

		List<Attribute> attributes = trainSet.getAttributes();
		int limit = (int) (attributes.size() * alpha);
		int[] indices = new int[limit];
		Permutation perm = new Permutation(attributes.size());
		perm.permute();

		// Backup targets
		double[] target = new double[trainSet.size()];
		for (int i = 0; i < target.length; i++) {
			target[i] = trainSet.get(i).getTarget();
		}

		// Create bags
		Instances[] bags = Bagging.createBags(trainSet, baggingIters);

		RegressionTreeLearner rtLearner = new RegressionTreeLearner();
		rtLearner.setConstructionMode(Mode.NUM_LEAVES_LIMITED);
		rtLearner.setMaxNumLeaves(maxNumLeaves);
		BaggedEnsembleLearner btLearner = new BaggedEnsembleLearner(baggingIters, rtLearner);

		double[] residualTrain = new double[trainSet.size()];
		for (int i = 0; i < residualTrain.length; i++) {
			residualTrain[i] = target[i];
		}

		// Gradient boosting
		for (int iter = 0; iter < maxNumIters; iter++) {
			// Prepare training set
			int[] a = perm.getPermutation();
			for (int i = 0; i < indices.length; i++) {
				indices[i] = a[i];
			}
			Arrays.sort(indices);
			List<Attribute> attList = trainSet.getAttributes(indices);
			trainSet.setAttributes(attList);
			for (int i = 0; i < residualTrain.length; i++) {
				trainSet.get(i).setTarget(residualTrain[i]);
			}

			BaggedEnsemble bt = btLearner.build(bags);
			if (learningRate != 1) {
				for (Predictor predictor : bt.getPredictors()) {
					RegressionTree rt = (RegressionTree) predictor;
					rt.multiply(learningRate);
				}
			}
			brt.add(bt);

			// Restore attributes
			trainSet.setAttributes(attributes);

			// Update residuals
			for (int i = 0; i < residualTrain.length; i++) {
				double pred = bt.regress(trainSet.get(i));
				residualTrain[i] -= pred;
			}

			if (verbose) {
				double rmse = StatUtils.rms(residualTrain);
				System.out.println("Iteration " + iter + ": " + rmse);
			}
		}

		// Restore targets
		for (int i = 0; i < target.length; i++) {
			trainSet.get(i).setTarget(target[i]);
		}

		return brt;
	}

	@Override
	public BRT build(Instances instances) {
		return buildRegressor(instances, maxNumIters, maxNumLeaves);
	}

}
