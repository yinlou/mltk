package mltk.predictor.tree.ensemble.brt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.core.Attribute;
import mltk.core.Instances;
import mltk.core.io.InstancesReader;
import mltk.predictor.Learner;
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
public class LADBoostLearner extends Learner {
	
	static class Options {

		@Argument(name = "-r", description = "attribute file path")
		String attPath = null;

		@Argument(name = "-t", description = "train set path", required = true)
		String trainPath = null;

		@Argument(name = "-o", description = "output model path")
		String outputModelPath = null;

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
	 * </p>
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

		LADBoostLearner ladBoostLearner = new LADBoostLearner();
		ladBoostLearner.setLearningRate(opts.learningRate);
		ladBoostLearner.setMaxNumIters(opts.maxNumIters);
		ladBoostLearner.setMaxNumLeaves(opts.maxNumLeaves);
		ladBoostLearner.setVerbose(true);

		long start = System.currentTimeMillis();
		BRT brt = ladBoostLearner.build(trainSet);
		long end = System.currentTimeMillis();
		System.out.println("Time: " + (end - start) / 1000.0);

		if (opts.outputModelPath != null) {
			PredictorWriter.write(brt, opts.outputModelPath);
		}
	}

	private int maxNumIters;
	private int maxNumLeaves;
	private double alpha;
	private double learningRate;

	/**
	 * Constructor.
	 */
	public LADBoostLearner() {
		verbose = false;
		maxNumIters = 3500;
		maxNumLeaves = 100;
		alpha = 1;
		learningRate = 1;
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
	 * Builds a regressor.
	 * 
	 * @param trainSet the training set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param maxNumLeaves the maximum number of leaves.
	 * @return a regressor.
	 */
	public BRT build(Instances trainSet, int maxNumIters, int maxNumLeaves) {
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

		RegressionTreeLearner rtLearner = new RegressionTreeLearner();
		rtLearner.setConstructionMode(Mode.NUM_LEAVES_LIMITED);
		rtLearner.setMaxNumLeaves(maxNumLeaves);

		double[] residualTrain = new double[trainSet.size()];
		for (int i = 0; i < residualTrain.length; i++) {
			residualTrain[i] = target[i] - intercept;
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
				trainSet.get(i).setTarget(MathUtils.sign(residualTrain[i]));
			}

			RegressionTree rt = rtLearner.build(trainSet);
			brt.trees[0].add(rt);

			// Restore attributes
			trainSet.setAttributes(attributes);

			// Replace the leaf value by median
			Map<RegressionTreeLeaf, List<Integer>> map = new HashMap<>();
			for (int i = 0; i < residualTrain.length; i++) {
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
					values[i] = residualTrain[list.get(i)];
				}
				double pred = ArrayUtils.getMedian(values) * learningRate;
				leaf.setPrediction(pred);
			}

			// Update residuals
			for (int i = 0; i < residualTrain.length; i++) {
				double pred = rt.regress(trainSet.get(i));
				residualTrain[i] -= pred;
			}

			if (verbose) {
				double lad = VectorUtils.l1norm(residualTrain) / residualTrain.length;
				System.out.println("Iteration " + iter + ": " + lad);
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
		return build(instances, maxNumIters, maxNumLeaves);
	}

}
