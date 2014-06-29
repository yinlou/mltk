package mltk.predictor.tree;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.core.Attribute;
import mltk.core.BinnedAttribute;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.NominalAttribute;
import mltk.core.Attribute.Type;
import mltk.core.io.InstancesReader;
import mltk.predictor.BaggedEnsemble;
import mltk.predictor.BaggedEnsembleLearner;
import mltk.predictor.io.PredictorWriter;
import mltk.util.Permutation;
import mltk.util.Random;
import mltk.util.tuple.DoublePair;
import mltk.util.tuple.IntDoublePair;

/**
 * Class for learning random regression trees. With
 * {@link mltk.predictor.BaggedEnsembleLearner BaggedEnsembleLearner}, random
 * forests of regression trees can be built.
 * 
 * @author Yin Lou
 * 
 */
public class RandomRegressionTreeLearner extends RegressionTreeLearner {

	private int numFeatures;
	private Permutation perm;

	/**
	 * Constructor.
	 */
	public RandomRegressionTreeLearner() {
		numFeatures = -1;
		alpha = 0.01;
		mode = Mode.ALPHA_LIMITED;
	}

	/**
	 * Returns the maximum number of features to consider for each node.
	 * 
	 * @return the maximum number of features to consider for each node.
	 */
	public int getNumFeatures() {
		return numFeatures;
	}

	/**
	 * Sets the maximum number of features to consider for each node.
	 * 
	 * @param numFeatures
	 *            the new maximum number of features.
	 */
	public void setNumFeatures(int numFeatures) {
		this.numFeatures = numFeatures;
	}

	@Override
	public RegressionTree build(Instances instances) {
		if (numFeatures <= 0) {
			numFeatures = instances.getAttributes().size() / 3;
		}
		if (perm == null || perm.size() != instances.getAttributes().size()) {
			perm = new Permutation(instances.getAttributes().size());
		}
		RegressionTree rt = null;
		switch (mode) {
		case ALPHA_LIMITED:
			rt = buildAlphaLimitedTree(instances, alpha);
			break;
		case NUM_LEAVES_LIMITED:
			rt = buildNumLeafLimitedTree(instances, maxNumLeaves);
			break;
		case DEPTH_LIMITED:
			rt = buildDepthLimitedTree(instances, maxDepth);
			break;
		default:
			break;
		}
		return rt;
	}

	protected RegressionTreeNode createNode(Dataset dataset, int limit,
			double[] stats) {
		boolean stdIs0 = getStats(dataset.instances, stats);
		final double totalWeights = stats[0];
		final double weightedMean = stats[1];
		final double sum = totalWeights * weightedMean;

		// 1. Check basic leaf conditions
		if (stats[0] < limit || stdIs0) {
			RegressionTreeNode node = new RegressionTreeLeaf(weightedMean);
			return node;
		}

		// 2. Find best split
		double bestEval = Double.POSITIVE_INFINITY;
		List<IntDoublePair> splits = new ArrayList<>();
		List<Attribute> attributes = dataset.instances.getAttributes();
		int[] a = perm.permute().getPermutation();
		Set<Integer> selected = new HashSet<>(numFeatures);
		for (int i = 0; i < numFeatures; i++) {
			selected.add(a[i]);
		}
		for (int i = 0; i < attributes.size(); i++) {
			Attribute attribute = attributes.get(i);
			int attIndex = attribute.getIndex();
			if (!selected.contains(i)) {
				continue;
			}
			List<Double> uniqueValues = null;
			List<DoublePair> histogram = null;
			if (attribute.getType() == Type.NOMINAL) {
				NominalAttribute attr = (NominalAttribute) attribute;
				DoublePair[] hist = new DoublePair[attr.getCardinality()];
				for (int j = 0; j < hist.length; j++) {
					hist[j] = new DoublePair(0, 0);
				}
				for (Instance instance : dataset.instances) {
					int idx = (int) instance.getValue(attIndex);
					hist[idx].v2 += instance.getTarget() * instance.getWeight();
					hist[idx].v1 += instance.getWeight();
				}

				uniqueValues = new ArrayList<>(hist.length);
				histogram = new ArrayList<>(hist.length);
				for (int j = 0; j < hist.length; j++) {
					if (hist[j].v1 != 0) {
						histogram.add(hist[j]);
						uniqueValues.add((double) j);
					}
				}
			} else if (attribute.getType() == Type.BINNED) {
				BinnedAttribute attr = (BinnedAttribute) attribute;
				DoublePair[] hist = new DoublePair[attr.getNumBins()];
				for (int j = 0; j < hist.length; j++) {
					hist[j] = new DoublePair(0, 0);
				}
				for (Instance instance : dataset.instances) {
					int idx = (int) instance.getValue(attIndex);
					hist[idx].v2 += instance.getTarget() * instance.getWeight();
					hist[idx].v1 += instance.getWeight();
				}

				uniqueValues = new ArrayList<>(hist.length);
				histogram = new ArrayList<>(hist.length);
				for (int j = 0; j < hist.length; j++) {
					if (hist[j].v1 != 0) {
						histogram.add(hist[j]);
						uniqueValues.add((double) j);
					}
				}
			} else {
				List<IntDoublePair> sortedList = dataset.sortedLists.get(i);
				int capacity = dataset.instances.size();
				uniqueValues = new ArrayList<>(capacity);
				histogram = new ArrayList<>(capacity);
				getHistogram(dataset.instances, sortedList, uniqueValues,
						histogram);
			}

			if (uniqueValues.size() > 1) {
				DoublePair split = split(uniqueValues, histogram, totalWeights,
						sum);
				if (split.v2 <= bestEval) {
					IntDoublePair splitPoint = new IntDoublePair(attIndex,
							split.v1);
					if (split.v2 < bestEval) {
						splits.clear();
						bestEval = split.v2;
					}
					splits.add(splitPoint);
				}
			}
		}
		if (bestEval < Double.POSITIVE_INFINITY) {
			Random rand = Random.getInstance();
			IntDoublePair splitPoint = splits.get(rand.nextInt(splits.size()));
			int attIndex = splitPoint.v1;
			RegressionTreeNode node = new RegressionTreeInteriorNode(attIndex,
					splitPoint.v2);
			if (stats.length > 2) {
				stats[2] = bestEval + totalWeights * weightedMean
						* weightedMean;
			}
			return node;
		} else {
			RegressionTreeNode node = new RegressionTreeLeaf(weightedMean);
			return node;
		}
	}

	static class Options {

		@Argument(name = "-r", description = "attribute file path", required = true)
		String attPath = null;

		@Argument(name = "-t", description = "train set path", required = true)
		String trainPath = null;

		@Argument(name = "-o", description = "output model path")
		String outputModelPath = null;

		@Argument(name = "-m", description = "construction mode:parameter. Construction mode can be alpha limited (a), depth limited (d), and number of leaves limited (l) (default: a:0.001)")
		String mode = "a:0.001";

		@Argument(name = "-f", description = "number of features to consider")
		int numFeatures = -1;

		@Argument(name = "-b", description = "bagging iterations (default: 100)")
		int baggingIters = 100;

	}

	/**
	 * Trains a random forest of regression trees.
	 * 
	 * When bagging is turned off (b = 0), this procedure generates a single
	 * random regression tree. When the number of features to consider is the
	 * number of total features, this procedure builds bagged tree.
	 * 
	 * <p>
	 * 
	 * <pre>
	 * Usage: RandomRegressionTreeLearner
	 * -r	attribute file path
	 * -t	train set path
	 * [-o]	output model path
	 * [-m]	construction mode:parameter. Construction mode can be alpha limited (a), depth limited (d), and number of leaves limited (l) (default: a:0.001)
	 * [-f]	number of features to consider
	 * [-b]	bagging iterations (default: 100)
	 * </pre>
	 * 
	 * </p>
	 * 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(
				RandomRegressionTreeLearner.class, opts);
		RandomRegressionTreeLearner rtLearner = new RandomRegressionTreeLearner();
		try {
			parser.parse(args);
			String[] data = opts.mode.split(":");
			if (data.length != 2) {
				throw new IllegalArgumentException();
			}
			switch (data[0]) {
			case "a":
				rtLearner.setConstructionMode(Mode.ALPHA_LIMITED);
				rtLearner.setAlpha(Double.parseDouble(data[1]));
				break;
			case "d":
				rtLearner.setConstructionMode(Mode.DEPTH_LIMITED);
				rtLearner.setMaxDepth(Integer.parseInt(data[1]));
				break;
			case "l":
				rtLearner.setConstructionMode(Mode.NUM_LEAVES_LIMITED);
				rtLearner.setMaxNumLeaves(Integer.parseInt(data[1]));
				break;
			default:
				throw new IllegalArgumentException();
			}
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}

		Instances trainSet = InstancesReader.read(opts.attPath, opts.trainPath);

		rtLearner.setNumFeatures(opts.numFeatures);
		BaggedEnsembleLearner rfLearner = new BaggedEnsembleLearner(
				opts.baggingIters, rtLearner);
		long start = System.currentTimeMillis();
		BaggedEnsemble rf = rfLearner.build(trainSet);
		long end = System.currentTimeMillis();
		System.out.println("Time: " + (end - start) / 1000.0 + " (s).");

		if (opts.outputModelPath != null) {
			PredictorWriter.write(rf, opts.outputModelPath);
		}
	}

}
