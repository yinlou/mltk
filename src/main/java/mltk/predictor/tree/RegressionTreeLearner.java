package mltk.predictor.tree;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.cmdline.options.LearnerOptions;
import mltk.core.Attribute;
import mltk.core.Instances;
import mltk.core.io.InstancesReader;
import mltk.predictor.evaluation.Evaluator;
import mltk.predictor.io.PredictorWriter;
import mltk.util.Random;
import mltk.util.Stack;
import mltk.util.Element;
import mltk.util.tuple.DoublePair;
import mltk.util.tuple.IntDoublePair;

/**
 * Class for learning regression trees.
 *
 * @author Yin Lou
 *
 */
public class RegressionTreeLearner extends RTreeLearner {
	
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
		long start = System.currentTimeMillis();
		RegressionTree rt = learner.build(trainSet);
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
	
	protected int maxDepth;
	protected int maxNumLeaves;
	protected int minLeafSize;
	protected double alpha;
	protected Mode mode;

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
		// stats[1]: sum
		// stats[2]: weightedMean
		// stats[3]: splitEval
		double[] stats = new double[4];
		if (maxDepth <= 0) {
			getStats(instances, stats);
			tree.root = new RegressionTreeLeaf(stats[1]);
			return tree;
		}
		Map<TreeNode, Dataset> datasets = new HashMap<>();
		Map<TreeNode, Integer> depths = new HashMap<>();
		Dataset dataset = Dataset.create(instances);
		tree.root = createNode(dataset, limit, stats);
		PriorityQueue<Element<TreeNode>> q = new PriorityQueue<>();
		q.add(new Element<TreeNode>(tree.root, stats[2]));
		datasets.put(tree.root, dataset);
		depths.put(tree.root, 0);

		while (!q.isEmpty()) {
			Element<TreeNode> elemt = q.remove();
			TreeNode node = elemt.element;
			Dataset data = datasets.get(node);
			int depth = depths.get(node);
			if (!node.isLeaf()) {
				TreeInteriorNode interiorNode = (TreeInteriorNode) node;
				Dataset left = new Dataset(data.instances);
				Dataset right = new Dataset(data.instances);
				split(data, interiorNode, left, right);

				if (depth >= maxDepth) {
					getStats(left.instances, stats);
					interiorNode.left = new RegressionTreeLeaf(stats[2]);
					getStats(right.instances, stats);
					interiorNode.right = new RegressionTreeLeaf(stats[2]);
				} else {
					interiorNode.left = createNode(left, limit, stats);
					if (!interiorNode.left.isLeaf()) {
						q.add(new Element<TreeNode>(interiorNode.left, stats[3]));
						datasets.put(interiorNode.left, left);
						depths.put(interiorNode.left, depth + 1);
					}
					interiorNode.right = createNode(right, limit, stats);
					if (!interiorNode.right.isLeaf()) {
						q.add(new Element<TreeNode>(interiorNode.right, stats[3]));
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
		// stats[0]: totalWeights
		// stats[1]: sum
		// stats[2]: weightedMean
		// stats[3]: splitEval
		double[] stats = new double[4];
		Dataset dataset = Dataset.create(instances);
		Stack<TreeNode> nodes = new Stack<>();
		Stack<Dataset> datasets = new Stack<>();
		tree.root = createNode(dataset, limit, stats);
		nodes.push(tree.root);
		datasets.push(dataset);
		while (!nodes.isEmpty()) {
			TreeNode node = nodes.pop();
			Dataset data = datasets.pop();
			if (!node.isLeaf()) {
				TreeInteriorNode interiorNode = (TreeInteriorNode) node;
				Dataset left = new Dataset(data.instances);
				Dataset right = new Dataset(data.instances);
				split(data, interiorNode, left, right);
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
		// stats[1]: sum
		// stats[2]: weightedMean
		// stats[3]: splitEval
		double[] stats = new double[4];
		Map<TreeNode, Double> nodePred = new HashMap<>();
		Map<TreeNode, Dataset> datasets = new HashMap<>();
		Dataset dataset = Dataset.create(instances);
		PriorityQueue<Element<TreeNode>> q = new PriorityQueue<>();
		tree.root = createNode(dataset, limit, stats);
		q.add(new Element<TreeNode>(tree.root, stats[2]));
		datasets.put(tree.root, dataset);
		nodePred.put(tree.root, stats[1]);

		int numLeaves = 0;
		while (!q.isEmpty()) {
			Element<TreeNode> elemt = q.remove();
			TreeNode node = elemt.element;
			Dataset data = datasets.get(node);
			if (!node.isLeaf()) {
				TreeInteriorNode interiorNode = (TreeInteriorNode) node;
				Dataset left = new Dataset(data.instances);
				Dataset right = new Dataset(data.instances);
				split(data, interiorNode, left, right);

				interiorNode.left = createNode(left, limit, stats);
				if (!interiorNode.left.isLeaf()) {
					nodePred.put(interiorNode.left, stats[2]);
					q.add(new Element<TreeNode>(interiorNode.left, stats[3]));
					datasets.put(interiorNode.left, left);
				} else {
					numLeaves++;
				}
				interiorNode.right = createNode(right, limit, stats);
				if (!interiorNode.right.isLeaf()) {
					nodePred.put(interiorNode.right, stats[2]);
					q.add(new Element<TreeNode>(interiorNode.right, stats[3]));
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
		Map<TreeNode, TreeNode> parent = new HashMap<>();
		traverse(tree.root, parent);
		while (!q.isEmpty()) {
			Element<TreeNode> elemt = q.remove();
			TreeNode node = elemt.element;

			double prediction = nodePred.get(node);
			TreeInteriorNode interiorNode = (TreeInteriorNode) parent.get(node);
			if (interiorNode.left == node) {
				interiorNode.left = new RegressionTreeLeaf(prediction);
			} else {
				interiorNode.right = new RegressionTreeLeaf(prediction);
			}
		}

		return tree;
	}

	protected TreeNode createNode(Dataset dataset, int limit, double[] stats) {
		boolean stdIs0 = getStats(dataset.instances, stats);
		final double totalWeights = stats[0];
		final double sum = stats[1];
		final double weightedMean = stats[2];

		// 1. Check basic leaf conditions
		if (dataset.instances.size() < limit || stdIs0) {
			TreeNode node = new RegressionTreeLeaf(weightedMean);
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
			TreeNode node = new TreeInteriorNode(attIndex, splitPoint.v2);
			stats[3] = bestEval + totalWeights * weightedMean * weightedMean;
			return node;
		} else {
			TreeNode node = new RegressionTreeLeaf(weightedMean);
			return node;
		}
	}

	protected void split(Dataset data, TreeInteriorNode node, Dataset left, Dataset right) {
		data.split(node.getSplitAttributeIndex(), node.getSplitPoint(), left, right);
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

	protected void traverse(TreeNode node, Map<TreeNode, TreeNode> parent) {
		if (!node.isLeaf()) {
			TreeInteriorNode interiorNode = (TreeInteriorNode) node;
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
