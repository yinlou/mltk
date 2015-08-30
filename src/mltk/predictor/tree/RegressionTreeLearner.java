package mltk.predictor.tree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.cmdline.options.LearnerOptions;
import mltk.core.Attribute;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.SparseVector;
import mltk.core.io.InstancesReader;
import mltk.predictor.Bagging;
import mltk.predictor.Learner;
import mltk.predictor.evaluation.Evaluator;
import mltk.predictor.io.PredictorWriter;
import mltk.util.Random;
import mltk.util.Stack;
import mltk.util.Element;
import mltk.util.tuple.DoublePair;
import mltk.util.tuple.IntDoublePair;
import mltk.util.tuple.IntDoublePairComparator;

/**
 * Class for learning regression trees.
 *
 * @author Yin Lou
 *
 */
public class RegressionTreeLearner extends Learner {
	
	static class Options extends LearnerOptions {

		@Argument(name = "-m", description = "construction mode:parameter. Construction mode can be alpha limited (a), depth limited (d), number of leaves limited (l) and minimum leaf size limited (s) (default: a:0.001)")
		String mode = "a:0.001";

		@Argument(name = "-s", description = "seed of the random number generator (default: 0)")
		long seed = 0L;

	}
	
	/**
	 * Trains a regression tree.
	 *
	 * <p>
	 *
	 * <pre>
	 * Usage: mltk.predictor.tree.RegressionTreeLearner
	 * -t	train set path
	 * [-r]	attribute file path
	 * [-o]	output model path
	 * [-V]	verbose (default: true)
	 * [-m]	construction mode:parameter. Construction mode can be alpha limited (a), depth limited (d), number of leaves limited (l) and minimum leaf size limited (s) (default: a:0.001)
	 * [-s]	seed of the random number generator (default: 0)
	 * </pre>
	 *
	 * </p>
	 *
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(RegressionTreeLearner.class, opts);
		RegressionTreeLearner learner = new RegressionTreeLearner();
		try {
			parser.parse(args);
			String[] data = opts.mode.split(":");
			if (data.length != 2) {
				throw new IllegalArgumentException();
			}
			switch (data[0]) {
				case "a":
					learner.setConstructionMode(Mode.ALPHA_LIMITED);
					learner.setAlpha(Double.parseDouble(data[1]));
					break;
				case "d":
					learner.setConstructionMode(Mode.DEPTH_LIMITED);
					learner.setMaxDepth(Integer.parseInt(data[1]));
					break;
				case "l":
					learner.setConstructionMode(Mode.NUM_LEAVES_LIMITED);
					learner.setMaxNumLeaves(Integer.parseInt(data[1]));
					break;
				case "s":
					learner.setConstructionMode(Mode.MIN_LEAF_SIZE_LIMITED);
					learner.setMinLeafSize(Integer.parseInt(data[1]));
				default:
					throw new IllegalArgumentException();
			}
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}

		Random.getInstance().setSeed(opts.seed);

		Instances trainSet = InstancesReader.read(opts.attPath, opts.trainPath);
		Instances bag = Bagging.createBootstrapSample(trainSet);
		long start = System.currentTimeMillis();
		RegressionTree rt = learner.build(bag);
		long end = System.currentTimeMillis();
		System.out.println("Time: " + (end - start) / 1000.0 + " (s).");
		System.out.println(Evaluator.evalRMSE(rt, trainSet));

		if (opts.outputModelPath != null) {
			PredictorWriter.write(rt, opts.outputModelPath);
		}
	}

	/**
	 * Enumeration of construction mode.
	 *
	 * @author Yin Lou
	 *
	 */
	public enum Mode {

		DEPTH_LIMITED, NUM_LEAVES_LIMITED, ALPHA_LIMITED, MIN_LEAF_SIZE_LIMITED;
	}

	protected static class Dataset {

		static Dataset create(Instances instances) {
			Dataset dataset = new Dataset(instances);
			List<Attribute> attributes = instances.getAttributes();
			// Feature selection may be applied
			Map<Integer, Integer> attMap = new HashMap<>();
			for (int j = 0; j < attributes.size(); j++) {
				Attribute attribute = attributes.get(j);
				attMap.put(attribute.getIndex(), j);
			}
			for (int j = 0; j < instances.dimension(); j++) {
				dataset.sortedLists.add(new ArrayList<IntDoublePair>());
			}
			for (int i = 0; i < instances.size(); i++) {
				Instance instance = instances.get(i);
				dataset.instances.add(instance.clone());
				if (instance.isSparse()) {
					SparseVector sv = (SparseVector) instance.getVector();
					int[] indices = sv.getIndices();
					double[] values = sv.getValues();
					for (int k = 0; k < indices.length; k++) {
						if (attMap.containsKey(indices[k])) {
							int idx = attMap.get(indices[k]);
							dataset.sortedLists.get(idx).add(new IntDoublePair(i, values[k]));
						}
					}
				} else {
					double[] values = instance.getValues();
					for (int j = 0; j < values.length; j++) {
						if (attMap.containsKey(j) && values[j] != 0.0) {
							int idx = attMap.get(j);
							dataset.sortedLists.get(idx).add(new IntDoublePair(i, values[j]));
						}
					}
				}
			}
			IntDoublePairComparator comp = new IntDoublePairComparator(false);
			for (List<IntDoublePair> sortedList : dataset.sortedLists) {
				Collections.sort(sortedList, comp);
			}
			return dataset;
		}

		public Instances instances;
		public List<List<IntDoublePair>> sortedLists;

		Dataset(Instances instances) {
			this.instances = new Instances(instances.getAttributes(), instances.getTargetAttribute());
			sortedLists = new ArrayList<>(instances.dimension());
		}

		void split(RegressionTreeInteriorNode node, Dataset left, Dataset right) {
			int[] leftHash = new int[instances.size()];
			int[] rightHash = new int[instances.size()];
			Arrays.fill(leftHash, -1);
			Arrays.fill(rightHash, -1);
			for (int i = 0; i < instances.size(); i++) {
				Instance instance = instances.get(i);
				if (node.goLeft(instance)) {
					left.instances.add(instance);
					leftHash[i] = left.instances.size() - 1;
				} else {
					right.instances.add(instance);
					rightHash[i] = right.instances.size() - 1;
				}
			}

			for (int i = 0; i < sortedLists.size(); i++) {
				left.sortedLists.add(new ArrayList<IntDoublePair>(left.instances.size()));
				right.sortedLists.add(new ArrayList<IntDoublePair>(right.instances.size()));

				List<IntDoublePair> sortedList = sortedLists.get(i);
				for (IntDoublePair pair : sortedList) {
					int leftIdx = leftHash[pair.v1];
					int rightIdx = rightHash[pair.v1];
					if (leftIdx != -1) {
						left.sortedLists.get(i).add(new IntDoublePair(leftIdx, pair.v2));
					}
					if (rightIdx != -1) {
						right.sortedLists.get(i).add(new IntDoublePair(rightIdx, pair.v2));
					}
				}
			}
		}

	}
	
	protected int maxDepth;
	protected int maxNumLeaves;
	protected int minLeafSize;
	protected double alpha;
	protected Mode mode;
	protected static final Double ZERO = new Double(0.0);

	/**
	 * Constructor.
	 */
	public RegressionTreeLearner() {
		alpha = 0.01;
		mode = Mode.ALPHA_LIMITED;
	}

	@Override
	public RegressionTree build(Instances instances) {
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
			case MIN_LEAF_SIZE_LIMITED:
				rt = buildMinLeafSizeLimitedTree(instances, minLeafSize);
			default:
				break;
		}
		return rt;
	}

	/**
	 * Returns the alpha.
	 *
	 * @return the alpha.
	 */
	public double getAlpha() {
		return alpha;
	}

	/**
	 * Returns the construction mode.
	 *
	 * @return the construction mode.
	 */
	public Mode getConstructionMode() {
		return mode;
	}

	/**
	 * Returns the maximum depth.
	 *
	 * @return the maximum depth.
	 */
	public int getMaxDepth() {
		return maxDepth;
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
	 * Returns the minimum leaf size.
	 *
	 * @return the minimum leaf size.
	 */
	public int getMinLeafSize() {
		return minLeafSize;
	}

	/**
	 * Sets the alpha. Alpha is the maximum proportion of the training set in the leaf node.
	 *
	 * @param alpha the alpha.
	 */
	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}

	/**
	 * Sets the construction mode.
	 *
	 * @param mode the construction mode.
	 */
	public void setConstructionMode(Mode mode) {
		this.mode = mode;
	}

	/**
	 * Sets the maximum depth.
	 *
	 * @param maxDepth the maximum depth.
	 */
	public void setMaxDepth(int maxDepth) {
		this.maxDepth = maxDepth;
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
	 * Sets the minimum leaf size.
	 *
	 * @param minLeafSize
	 */
	public void setMinLeafSize(int minLeafSize) {
		this.minLeafSize = minLeafSize;
	}

	protected RegressionTree buildAlphaLimitedTree(Instances instances, double alpha) {
		final int limit = (int) (alpha * instances.size());
		return buildMinLeafSizeLimitedTree(instances, limit);
	}

	protected RegressionTree buildDepthLimitedTree(Instances instances, int maxDepth) {
		RegressionTree tree = new RegressionTree();
		final int limit = 5;
		// stats[0]: totalWeights
		// stats[1]: weightedMean
		// stats[2]: splitEval
		double[] stats = new double[3];
		if (maxDepth == 1) {
			getStats(instances, stats);
			tree.root = new RegressionTreeLeaf(stats[1]);
			return tree;
		}
		Map<RegressionTreeNode, Dataset> datasets = new HashMap<>();
		Map<RegressionTreeNode, Integer> depths = new HashMap<>();
		Dataset dataset = Dataset.create(instances);
		tree.root = createNode(dataset, limit, stats);
		PriorityQueue<Element<RegressionTreeNode>> q = new PriorityQueue<>();
		q.add(new Element<RegressionTreeNode>(tree.root, stats[2]));
		datasets.put(tree.root, dataset);
		depths.put(tree.root, 1);

		while (!q.isEmpty()) {
			Element<RegressionTreeNode> elemt = q.remove();
			RegressionTreeNode node = elemt.element;
			Dataset data = datasets.get(node);
			int depth = depths.get(node);
			if (!node.isLeaf()) {
				RegressionTreeInteriorNode interiorNode = (RegressionTreeInteriorNode) node;
				Dataset left = new Dataset(data.instances);
				Dataset right = new Dataset(data.instances);
				data.split(interiorNode, left, right);

				if (depth + 1 == maxDepth) {
					getStats(left.instances, stats);
					interiorNode.left = new RegressionTreeLeaf(stats[1]);
					getStats(right.instances, stats);
					interiorNode.right = new RegressionTreeLeaf(stats[1]);
				} else {
					interiorNode.left = createNode(left, limit, stats);
					if (!interiorNode.left.isLeaf()) {
						q.add(new Element<RegressionTreeNode>(interiorNode.left, stats[2]));
						datasets.put(interiorNode.left, left);
						depths.put(interiorNode.left, depth + 1);
					}
					interiorNode.right = createNode(right, limit, stats);
					if (!interiorNode.right.isLeaf()) {
						q.add(new Element<RegressionTreeNode>(interiorNode.right, stats[2]));
						datasets.put(interiorNode.right, right);
						depths.put(interiorNode.right, depth + 1);
					}
				}
			}
		}

		return tree;
	}

	protected RegressionTree buildMinLeafSizeLimitedTree(Instances instances, int limit) {
		RegressionTree tree = new RegressionTree();
		double[] stats = new double[3];
		Dataset dataset = Dataset.create(instances);
		Stack<RegressionTreeNode> nodes = new Stack<>();
		Stack<Dataset> datasets = new Stack<>();
		tree.root = createNode(dataset, limit, stats);
		nodes.push(tree.root);
		datasets.push(dataset);
		while (!nodes.isEmpty()) {
			RegressionTreeNode node = nodes.pop();
			Dataset data = datasets.pop();
			if (!node.isLeaf()) {
				RegressionTreeInteriorNode interiorNode = (RegressionTreeInteriorNode) node;
				Dataset left = new Dataset(data.instances);
				Dataset right = new Dataset(data.instances);
				data.split(interiorNode, left, right);
				interiorNode.left = createNode(left, limit, stats);
				interiorNode.right = createNode(right, limit, stats);
				nodes.push(interiorNode.left);
				datasets.push(left);
				nodes.push(interiorNode.right);
				datasets.push(right);
			}
		}
		return tree;
	}

	protected RegressionTree buildNumLeafLimitedTree(Instances instances, int maxNumLeaves) {
		RegressionTree tree = new RegressionTree();
		final int limit = 5;
		// stats[0]: totalWeights
		// stats[1]: weightedMean
		// stats[2]: splitEval
		double[] stats = new double[3];
		Map<RegressionTreeNode, Double> nodePred = new HashMap<>();
		Map<RegressionTreeNode, Dataset> datasets = new HashMap<>();
		Dataset dataset = Dataset.create(instances);
		PriorityQueue<Element<RegressionTreeNode>> q = new PriorityQueue<>();
		tree.root = createNode(dataset, limit, stats);
		q.add(new Element<RegressionTreeNode>(tree.root, stats[2]));
		datasets.put(tree.root, dataset);
		nodePred.put(tree.root, stats[1]);

		int numLeaves = 0;
		while (!q.isEmpty()) {
			Element<RegressionTreeNode> elemt = q.remove();
			RegressionTreeNode node = elemt.element;
			Dataset data = datasets.get(node);
			if (!node.isLeaf()) {
				RegressionTreeInteriorNode interiorNode = (RegressionTreeInteriorNode) node;
				Dataset left = new Dataset(data.instances);
				Dataset right = new Dataset(data.instances);
				data.split(interiorNode, left, right);

				interiorNode.left = createNode(left, limit, stats);
				if (!interiorNode.left.isLeaf()) {
					nodePred.put(interiorNode.left, stats[1]);
					q.add(new Element<RegressionTreeNode>(interiorNode.left, stats[2]));
					datasets.put(interiorNode.left, left);
				} else {
					numLeaves++;
				}
				interiorNode.right = createNode(right, limit, stats);
				if (!interiorNode.right.isLeaf()) {
					nodePred.put(interiorNode.right, stats[1]);
					q.add(new Element<RegressionTreeNode>(interiorNode.right, stats[2]));
					datasets.put(interiorNode.right, right);
				} else {
					numLeaves++;
				}

				if (numLeaves + q.size() >= maxNumLeaves) {
					break;
				}
			}
		}

		// Convert interior nodes to leaves
		Map<RegressionTreeNode, RegressionTreeNode> parent = new HashMap<>();
		traverse(tree.root, parent);
		while (!q.isEmpty()) {
			Element<RegressionTreeNode> elemt = q.remove();
			RegressionTreeNode node = elemt.element;

			double prediction = nodePred.get(node);
			RegressionTreeInteriorNode interiorNode = (RegressionTreeInteriorNode) parent.get(node);
			if (interiorNode.left == node) {
				interiorNode.left = new RegressionTreeLeaf(prediction);
			} else {
				interiorNode.right = new RegressionTreeLeaf(prediction);
			}
		}

		return tree;
	}

	protected RegressionTreeNode createNode(Dataset dataset, int limit, double[] stats) {
		boolean stdIs0 = getStats(dataset.instances, stats);
		final double totalWeights = stats[0];
		final double weightedMean = stats[1];
		final double sum = totalWeights * weightedMean;

		// 1. Check basic leaf conditions
		if (dataset.instances.size() < limit || stdIs0) {
			RegressionTreeNode node = new RegressionTreeLeaf(weightedMean);
			return node;
		}

		// 2. Find best split
		double bestEval = Double.POSITIVE_INFINITY;
		List<IntDoublePair> splits = new ArrayList<>();
		List<Attribute> attributes = dataset.instances.getAttributes();
		for (int j = 0; j < attributes.size(); j++) {
			int attIndex = attributes.get(j).getIndex();
			List<IntDoublePair> sortedList = dataset.sortedLists.get(j);
			List<Double> uniqueValues = new ArrayList<>(sortedList.size());
			List<DoublePair> histogram = new ArrayList<>(sortedList.size());
			getHistogram(dataset.instances, sortedList, uniqueValues, totalWeights, sum, histogram);

			if (uniqueValues.size() > 1) {
				DoublePair split = split(uniqueValues, histogram, totalWeights, sum);
				if (split.v2 <= bestEval) {
					IntDoublePair splitPoint = new IntDoublePair(attIndex, split.v1);
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
			RegressionTreeNode node = new RegressionTreeInteriorNode(attIndex, splitPoint.v2);
			if (stats.length > 2) {
				stats[2] = bestEval + totalWeights * weightedMean * weightedMean;
			}
			return node;
		} else {
			RegressionTreeNode node = new RegressionTreeLeaf(weightedMean);
			return node;
		}
	}

	protected void getHistogram(Instances instances, List<IntDoublePair> pairs, List<Double> uniqueValues, double w,
			double s, List<DoublePair> histogram) {
		if (pairs.size() == 0) {
			return;
		}
		double lastValue = pairs.get(0).v2;
		double totalWeight = instances.get(pairs.get(0).v1).getWeight();
		double sum = instances.get(pairs.get(0).v1).getTarget() * totalWeight;

		for (int i = 1; i < pairs.size(); i++) {
			IntDoublePair pair = pairs.get(i);
			double value = pair.v2;
			double weight = instances.get(pairs.get(i).v1).getWeight();
			double resp = instances.get(pairs.get(i).v1).getTarget();
			if (value != lastValue) {
				uniqueValues.add(lastValue);
				histogram.add(new DoublePair(totalWeight, sum));
				lastValue = value;
				totalWeight = weight;
				sum = resp * weight;
			} else {
				totalWeight += weight;
				sum += resp * weight;
			}
		}
		uniqueValues.add(lastValue);
		histogram.add(new DoublePair(totalWeight, sum));

		if (pairs.size() != instances.size()) {
			// Zero entries are present
			double sumWeight = 0;
			double sumTarget = 0;
			for (DoublePair pair : histogram) {
				sumWeight += pair.v1;
				sumTarget += pair.v2;
			}

			double weightOnZero = w - sumWeight;
			double sumOnZero = s - sumTarget;
			int idx = Collections.binarySearch(uniqueValues, ZERO);
			if (idx < 0) {
				// This should always happen
				uniqueValues.add(-idx - 1, ZERO);
				histogram.add(-idx - 1, new DoublePair(weightOnZero, sumOnZero));
			}
		}
	}

	protected boolean getStats(Instances instances, double[] stats) {
		stats[0] = stats[1] = 0;
		if (instances.size() == 0) {
			return true;
		}
		double firstTarget = instances.get(0).getTarget();
		boolean stdIs0 = true;
		for (Instance instance : instances) {
			double weight = instance.getWeight();
			double target = instance.getTarget();
			stats[0] += weight;
			stats[1] += weight * target;
			if (stdIs0 && target != firstTarget) {
				stdIs0 = false;
			}
		}
		stats[1] /= stats[0];
		return stdIs0;
	}

	protected DoublePair split(List<Double> uniqueValues, List<DoublePair> hist, double totalWeights, double sum) {
		double weight1 = hist.get(0).v1;
		double weight2 = totalWeights - weight1;
		double sum1 = hist.get(0).v2;
		double sum2 = sum - sum1;

		double bestEval = -sum1 * sum1 / weight1 - sum2 * sum2 / weight2;
		List<Double> splits = new ArrayList<>();
		splits.add((uniqueValues.get(0) + uniqueValues.get(0 + 1)) / 2);
		for (int i = 1; i < uniqueValues.size() - 1; i++) {
			final double w = hist.get(i).v1;
			final double s = hist.get(i).v2;
			weight1 += w;
			weight2 -= w;
			sum1 += s;
			sum2 -= s;
			double eval = -sum1 * sum1 / weight1 - sum2 * sum2 / weight2;
			if (eval <= bestEval) {
				double split = (uniqueValues.get(i) + uniqueValues.get(i + 1)) / 2;
				if (eval < bestEval) {
					bestEval = eval;
					splits.clear();
				}
				splits.add(split);
			}
		}
		Random rand = Random.getInstance();
		double split = splits.get(rand.nextInt(splits.size()));
		return new DoublePair(split, bestEval);
	}

	protected void traverse(RegressionTreeNode node, Map<RegressionTreeNode, RegressionTreeNode> parent) {
		if (!node.isLeaf()) {
			RegressionTreeInteriorNode interiorNode = (RegressionTreeInteriorNode) node;
			if (interiorNode.left != null) {
				parent.put(interiorNode.left, node);
				traverse(interiorNode.left, parent);
			}
			if (interiorNode.right != null) {
				parent.put(interiorNode.right, node);
				traverse(interiorNode.right, parent);
			}
		}
	}

}
