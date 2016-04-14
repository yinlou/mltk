package mltk.predictor.tree.ensemble.brt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.cmdline.options.LearnerOptions;
import mltk.core.Attribute;
import mltk.core.Instances;
import mltk.core.io.InstancesReader;
import mltk.predictor.evaluation.Metric;
import mltk.predictor.io.PredictorWriter;
import mltk.predictor.tree.RTree;
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
public class LSBoostLearner extends BRTLearner {
	
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
	 * Trains a boosted tree ensemble using least square as the objective function.
	 *
	 * <pre>
	 * Usage: mltk.predictor.tree.ensemble.brt.LSBoostLearner
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
		CmdLineParser parser = new CmdLineParser(LSBoostLearner.class, opts);
		try {
			parser.parse(args);
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}

		Random.getInstance().setSeed(opts.seed);

		Instances trainSet = InstancesReader.read(opts.attPath, opts.trainPath);
		
		RegressionTreeLearner rtLearner = new RegressionTreeLearner();
		rtLearner.setConstructionMode(Mode.NUM_LEAVES_LIMITED);
		rtLearner.setMaxNumLeaves(opts.maxNumLeaves);

		LSBoostLearner learner = new LSBoostLearner();
		learner.setLearningRate(opts.learningRate);
		learner.setMaxNumIters(opts.maxNumIters);
		learner.setVerbose(opts.verbose);
		learner.setTreeLearner(rtLearner);

		long start = System.currentTimeMillis();
		BRT brt = learner.build(trainSet);
		long end = System.currentTimeMillis();
		System.out.println("Time: " + (end - start) / 1000.0);

		if (opts.outputModelPath != null) {
			PredictorWriter.write(brt, opts.outputModelPath);
		}
	}

	/**
	 * Constructor.
	 */
	public LSBoostLearner() {
		verbose = false;
		maxNumIters = 3500;
		learningRate = 1;
		alpha = 1;
		
		RegressionTreeLearner rtLearner = new RegressionTreeLearner();
		rtLearner.setConstructionMode(Mode.NUM_LEAVES_LIMITED);
		rtLearner.setMaxNumLeaves(100);
		
		treeLearner = rtLearner;
	}
	
	/**
	 * Builds a regressor.
	 *
	 * @param trainSet the training set.
	 * @param validSet the validation set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param metric the metric to optimize for on the validation set.
	 * @return a regressor.
	 */
	public BRT buildRegressor(Instances trainSet, Instances validSet, int maxNumIters, Metric metric) {
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

		double[] rTrain = new double[trainSet.size()];
		for (int i = 0; i < rTrain.length; i++) {
			rTrain[i] = target[i];
		}
		double[] pValid = new double[validSet.size()];

		List<Double> measureList = new ArrayList<>(maxNumIters);
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
			// Prepare training set
			for (int i = 0; i < rTrain.length; i++) {
				trainSet.get(i).setTarget(rTrain[i]);
			}

			RTree rt = (RTree) treeLearner.build(trainSet);
			if (learningRate != 1) {
				rt.multiply(learningRate);
			}
			brt.trees[0].add(rt);

			if (alpha < 1) {
				// Restore attributes
				trainSet.setAttributes(attributes);
			}

			// Update predictions and residuals
			for (int i = 0; i < rTrain.length; i++) {
				double pred = rt.regress(trainSet.get(i));
				rTrain[i] -= pred;
			}
			for (int i = 0; i < pValid.length; i++) {
				double pred = rt.regress(validSet.get(i));
				pValid[i] += pred;
			}

			double measure = metric.eval(pValid, validSet);
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
	 * @return a regressor.
	 */
	public BRT buildRegressor(Instances trainSet, int maxNumIters) {
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

		double[] rTrain = new double[trainSet.size()];
		for (int i = 0; i < rTrain.length; i++) {
			rTrain[i] = target[i];
		}

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

			RTree rt = (RTree) treeLearner.build(trainSet);
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

	@Override
	public BRT build(Instances instances) {
		return buildRegressor(instances, maxNumIters);
	}

}
