package mltk.predictor.tree.ensemble.brt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.cmdline.options.LearnerOptions;
import mltk.core.Attribute;
import mltk.core.Instances;
import mltk.core.io.InstancesReader;
import mltk.predictor.evaluation.Metric;
import mltk.predictor.io.PredictorWriter;
import mltk.predictor.tree.RegressionTree;
import mltk.predictor.tree.RegressionTreeLeaf;
import mltk.predictor.tree.RegressionTreeLearner;
import mltk.predictor.tree.RegressionTreeLearner.Mode;
import mltk.util.ArrayUtils;
import mltk.util.MathUtils;
import mltk.util.Permutation;
import mltk.util.Random;
import mltk.util.VectorUtils;

/**
 * Class for least-absolute-deviation boost learner.
 * 
 * @author Yin Lou
 * 
 */
public class LADBoostLearner extends BRTLearner {
	
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
	 * Trains a boosted tree ensemble using least-absolute-deviation as the objective function.
	 * 
	 * <pre>
	 * Usage: mltk.predictor.tree.ensemble.brt.LADBoostLearner
	 * -t	train set path
	 * -m	maximum number of iterations
	 * [-r]	attribute file path
	 * [-o]	output model path
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
		CmdLineParser parser = new CmdLineParser(LADBoostLearner.class, opts);
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

		LADBoostLearner learner = new LADBoostLearner();
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
	public LADBoostLearner() {
		verbose = false;
		maxNumIters = 3500;
		alpha = 1;
		learningRate = 1;
		
		RegressionTreeLearner rtLearner = new RegressionTreeLearner();
		rtLearner.setConstructionMode(Mode.NUM_LEAVES_LIMITED);
		rtLearner.setMaxNumLeaves(100);
		
		treeLearner = rtLearner;
	}

	@Override
	public BRT build(Instances instances) {
		return buildRegressor(instances, maxNumIters);
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
		double intercept = ArrayUtils.getMedian(target);
		RegressionTree initialTree = new RegressionTree(new RegressionTreeLeaf(intercept));
		brt.trees[0].add(initialTree);

		double[] rTrain = new double[trainSet.size()];
		for (int i = 0; i < rTrain.length; i++) {
			rTrain[i] = target[i] - intercept;
		}
		double[] pValid = new double[validSet.size()];
		Arrays.fill(pValid, intercept);

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
			for (int i = 0; i < rTrain.length; i++) {
				trainSet.get(i).setTarget(MathUtils.sign(rTrain[i]));
			}

			RegressionTree rt = (RegressionTree) treeLearner.build(trainSet);
			brt.trees[0].add(rt);

			if (alpha < 1) {
				// Restore attributes
				trainSet.setAttributes(attributes);
			}

			// Replace the leaf value by median
			Map<RegressionTreeLeaf, List<Integer>> map = new HashMap<>();
			for (int i = 0; i < rTrain.length; i++) {
				RegressionTreeLeaf leaf = rt.getLeafNode(trainSet.get(i));
				if (!map.containsKey(leaf)) {
					map.put(leaf, new ArrayList<Integer>());
				}
				map.get(leaf).add(i);
			}
			for (Map.Entry<RegressionTreeLeaf, List<Integer>> entry : map.entrySet()) {
				RegressionTreeLeaf leaf = entry.getKey();
				List<Integer> list = entry.getValue();
				double[] values = new double[list.size()];
				for (int i = 0; i < values.length; i++) {
					values[i] = rTrain[list.get(i)];
				}
				double pred = ArrayUtils.getMedian(values) * learningRate;
				leaf.setPrediction(pred);
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
		double intercept = ArrayUtils.getMedian(target);
		RegressionTree initialTree = new RegressionTree(new RegressionTreeLeaf(intercept));
		brt.trees[0].add(initialTree);

		double[] rTrain = new double[trainSet.size()];
		for (int i = 0; i < rTrain.length; i++) {
			rTrain[i] = target[i] - intercept;
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
			// Prepare training set
			for (int i = 0; i < rTrain.length; i++) {
				trainSet.get(i).setTarget(MathUtils.sign(rTrain[i]));
			}

			RegressionTree rt = (RegressionTree) treeLearner.build(trainSet);
			brt.trees[0].add(rt);

			if (alpha < 1) {
				// Restore attributes
				trainSet.setAttributes(attributes);
			}

			// Replace the leaf value by median
			Map<RegressionTreeLeaf, List<Integer>> map = new HashMap<>();
			for (int i = 0; i < rTrain.length; i++) {
				RegressionTreeLeaf leaf = rt.getLeafNode(trainSet.get(i));
				if (!map.containsKey(leaf)) {
					map.put(leaf, new ArrayList<Integer>());
				}
				map.get(leaf).add(i);
			}
			for (Map.Entry<RegressionTreeLeaf, List<Integer>> entry : map.entrySet()) {
				RegressionTreeLeaf leaf = entry.getKey();
				List<Integer> list = entry.getValue();
				double[] values = new double[list.size()];
				for (int i = 0; i < values.length; i++) {
					values[i] = rTrain[list.get(i)];
				}
				double pred = ArrayUtils.getMedian(values) * learningRate;
				leaf.setPrediction(pred);
			}

			// Update residuals
			for (int i = 0; i < rTrain.length; i++) {
				double pred = rt.regress(trainSet.get(i));
				rTrain[i] -= pred;
			}

			if (verbose) {
				double lad = VectorUtils.l1norm(rTrain) / rTrain.length;
				System.out.println("Iteration " + iter + ": " + lad);
			}
		}

		// Restore targets
		for (int i = 0; i < target.length; i++) {
			trainSet.get(i).setTarget(target[i]);
		}

		return brt;
	}

}
