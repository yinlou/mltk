package mltk.predictor.function;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.PriorityQueue;

import mltk.core.Attribute;
import mltk.core.BinnedAttribute;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.NominalAttribute;
import mltk.predictor.Learner;
import mltk.util.Random;
import mltk.util.Element;
import mltk.util.tuple.DoublePair;

/**
 * Class for cutting lines.
 * 
 * @author Yin Lou
 * 
 */
public class LineCutter extends Learner {

	static class Interval implements Comparable<Interval> {

		boolean finalized;
		int start;
		int end;
		// INF: Can split but split point not computed
		// NaN: Declared as leaf
		// Other: Split point
		double split;
		double weight;
		double sum;
		double value; // mean * sum, or sum * sum /weight; negative gain
		double gain;
		Interval left;
		Interval right;

		Interval() {
			split = Double.POSITIVE_INFINITY;
		}

		Interval(int start, int end, double weight, double sum) {
			this.start = start;
			this.end = end;
			this.split = Double.POSITIVE_INFINITY;
			this.weight = weight;
			this.sum = sum;
		}

		@Override
		public int compareTo(Interval o) {
			if (this.value < o.value) {
				return -1;
			} else if (this.value > o.value) {
				return 1;
			} else {
				return 0;
			}
		}

		double getPrediction() {
			return sum / weight;
		}

		boolean isFinalized() {
			return finalized;
		}

		boolean isInteriorNode() {
			return split < Double.POSITIVE_INFINITY;
		}

		boolean isLeaf() {
			return Double.isNaN(split);
		}

	}

	protected static void build(Function1D func, List<Double> uniqueValues, List<DoublePair> stats, double limit) {
		// 1. Check basic leaf conditions
		if (uniqueValues.size() == 1) {
			func.splits = new double[] { Double.POSITIVE_INFINITY };
			DoublePair stat = stats.get(0);
			func.predictions = new double[] { stat.v2 / stat.v1 };
			return;
		}

		// 2. Cut the line
		// 2.1 First cut
		DoublePair pair = sumUp(stats, 0, stats.size());
		Interval root = new Interval(0, stats.size(), pair.v1, pair.v2);
		split(uniqueValues, stats, root, limit);

		PriorityQueue<Interval> q = new PriorityQueue<>();
		if (!root.isLeaf()) {
			q.add(root);
		}

		int numSplits = 0;
		while (!q.isEmpty()) {
			Interval parent = q.remove();
			parent.finalized = true;
			split(uniqueValues, stats, parent.left, limit);
			split(uniqueValues, stats, parent.right, limit);

			if (!parent.left.isLeaf()) {
				q.add(parent.left);
			}
			if (!parent.right.isLeaf()) {
				q.add(parent.right);
			}
			numSplits++;
		}

		List<Double> splits = new ArrayList<>(numSplits);
		List<Double> predictions = new ArrayList<>(numSplits + 1);
		inorder(root, splits, predictions);
		func.splits = new double[predictions.size()];
		func.predictions = new double[predictions.size()];
		for (int i = 0; i < func.predictions.length; i++) {
			func.predictions[i] = predictions.get(i);
		}
		for (int i = 0; i < func.splits.length - 1; i++) {
			func.splits[i] = splits.get(i);
		}
		func.splits[func.splits.length - 1] = Double.POSITIVE_INFINITY;
	}

	protected static void build(Function1D func, List<Double> uniqueValues, List<DoublePair> stats, int numIntervals) {
		// 1. Check basic leaf conditions
		if (uniqueValues.size() == 1) {
			func.splits = new double[] { Double.POSITIVE_INFINITY };
			DoublePair stat = stats.get(0);
			func.predictions = new double[] { stat.v2 / stat.v1 };
			return;
		}

		// 2. Cut the line
		// 2.1 First cut
		DoublePair pair = sumUp(stats, 0, stats.size());
		Interval root = new Interval(0, stats.size(), pair.v1, pair.v2);
		split(uniqueValues, stats, root);

		if (numIntervals == 2) {
			func.splits = new double[] { root.split, Double.POSITIVE_INFINITY };
			func.predictions = new double[] { root.left.getPrediction(), root.right.getPrediction() };
		} else if (numIntervals > 2) {
			PriorityQueue<Interval> q = new PriorityQueue<>();
			if (!root.isLeaf()) {
				q.add(root);
			}

			int numSplits = 0;
			while (!q.isEmpty()) {
				Interval parent = q.remove();
				parent.finalized = true;
				split(uniqueValues, stats, parent.left);
				split(uniqueValues, stats, parent.right);

				if (!parent.left.isLeaf()) {
					q.add(parent.left);
				}
				if (!parent.right.isLeaf()) {
					q.add(parent.right);
				}

				numSplits++;
				if (numSplits >= numIntervals - 1) {
					break;
				}
			}

			List<Double> splits = new ArrayList<>(numIntervals - 1);
			List<Double> predictions = new ArrayList<>(numIntervals);
			inorder(root, splits, predictions);
			func.splits = new double[predictions.size()];
			func.predictions = new double[predictions.size()];
			for (int i = 0; i < func.predictions.length; i++) {
				func.predictions[i] = predictions.get(i);
			}
			for (int i = 0; i < func.splits.length - 1; i++) {
				func.splits[i] = splits.get(i);
			}
			func.splits[func.splits.length - 1] = Double.POSITIVE_INFINITY;
		}
	}

	protected static void getStats(List<Element<DoublePair>> pairs, List<Double> uniqueValues, List<DoublePair> stats) {
		if (pairs.size() == 0) {
			return;
		}
		double lastValue = pairs.get(0).weight;
		double totalWeight = pairs.get(0).element.v2;
		double sum = pairs.get(0).element.v1 * totalWeight;
		double lastResp = pairs.get(0).element.v1;
		boolean isStd0 = true;
		for (int i = 1; i < pairs.size(); i++) {
			Element<DoublePair> element = pairs.get(i);
			double value = element.weight;
			double weight = element.element.v2;
			double resp = element.element.v1;
			if (value != lastValue) {
				uniqueValues.add(lastValue);
				stats.add(new DoublePair(totalWeight, sum));
				lastValue = value;
				totalWeight = weight;
				sum = resp * weight;
				lastResp = resp;
				isStd0 = true;
			} else {
				totalWeight += weight;
				sum += weight * resp;
				isStd0 = isStd0 && (lastResp == resp);
			}
		}
		uniqueValues.add(lastValue);
		stats.add(new DoublePair(totalWeight, sum));
	}

	protected static void inorder(Interval parent, List<Double> splits, List<Double> predictions) {
		if (parent.isFinalized()) {
			inorder(parent.left, splits, predictions);
			splits.add(parent.split);
			inorder(parent.right, splits, predictions);
		} else {
			predictions.add(parent.getPrediction());
		}
	}

	/**
	 * Performs a line search for a function for classification.
	 * 
	 * @param instances the training set.
	 * @param func the function.
	 */
	public static void lineSearch(Instances instances, Function1D func) {
		double[] predictions = func.getPredictions();
		double[] numerator = new double[predictions.length];
		double[] denominator = new double[numerator.length];
		for (Instance instance : instances) {
			int idx = func.getSegmentIndex(instance);
			numerator[idx] += instance.getTarget() * instance.getWeight();
			double t = Math.abs(instance.getTarget());
			denominator[idx] += t * (1 - t) * instance.getWeight();
		}
		for (int i = 0; i < predictions.length; i++) {
			predictions[i] = denominator[i] == 0 ? 0 : numerator[i] / denominator[i];
		}
	}

	protected static void split(List<Double> uniqueValues, List<DoublePair> stats, Interval parent) {
		split(uniqueValues, stats, parent, 5);
	}

	protected static void split(List<Double> uniqueValues, List<DoublePair> stats, Interval parent, double limit) {
		// Test if we need to split
		if (parent.weight <= limit || parent.end - parent.start <= 1) {
			parent.split = Double.NaN; // Declared as leaf
		} else {
			parent.left = new Interval();
			parent.right = new Interval();
			int start = parent.left.start = parent.start;
			int end = parent.right.end = parent.end;
			final double totalWeight = parent.weight;
			final double sum = parent.sum;

			double weight1 = stats.get(start).v1;
			double weight2 = totalWeight - weight1;
			double sum1 = stats.get(start).v2;
			double sum2 = sum - sum1;

			double bestEval = -sum1 * sum1 / weight1 - sum2 * sum2 / weight2;
			List<double[]> splits = new ArrayList<>();
			splits.add(new double[] { (uniqueValues.get(start) + uniqueValues.get(start + 1)) / 2, start, weight1,
					sum1, weight2, sum2 });
			for (int i = start + 1; i < end - 1; i++) {
				final double w = stats.get(i).v1;
				final double s = stats.get(i).v2;
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
					splits.add(new double[] { split, i, weight1, sum1, weight2, sum2 });
				}
			}
			Random rand = Random.getInstance();
			double[] split = splits.get(rand.nextInt(splits.size()));
			parent.split = split[0];
			parent.left.end = (int) split[1] + 1;
			parent.right.start = (int) split[1] + 1;
			parent.left.weight = split[2];
			parent.left.sum = split[3];
			parent.right.weight = split[4];
			parent.right.sum = split[5];
			parent.gain = (split[3] / split[2]) * split[3] + (split[5] / split[4]) * split[5];
			parent.value = -parent.gain + (sum / totalWeight * sum);
		}
	}

	protected static DoublePair sumUp(List<DoublePair> stats, int start, int end) {
		double weight = 0;
		double sum = 0;
		for (int i = start; i < end; i++) {
			DoublePair stat = stats.get(i);
			weight += stat.v1;
			sum += stat.v2;
		}
		return new DoublePair(weight, sum);
	}

	private int attIndex;

	private int numIntervals;

	private boolean lineSearch;

	private boolean leafLimited;

	private double alpha;

	/**
	 * Constructor.
	 */
	public LineCutter() {
		this(false);
	}

	/**
	 * Constructor.
	 * 
	 * @param lineSearch <code>true</code> if line search is performed in the end.
	 */
	public LineCutter(boolean lineSearch) {
		attIndex = -1;
		this.lineSearch = lineSearch;
		leafLimited = true;
	}

	@Override
	public Function1D build(Instances instances) {
		Function1D func = leafLimited ? build(instances, attIndex, numIntervals) : build(instances, attIndex, alpha);
		return func;
	}

	/**
	 * Builds a 1D function.
	 * 
	 * @param instances the training set.
	 * @param attribute the attribute.
	 * @param alpha the alpha.
	 * @return a 1D function.
	 */
	public Function1D build(Instances instances, Attribute attribute, double alpha) {
		Function1D func = new Function1D();
		func.attIndex = attribute.getIndex();
		// TODO potential bugs
		double limit = alpha * instances.size();

		if (attribute.getType() == Attribute.Type.NUMERIC) {
			// weight: attribute value
			// DoublePair.v1: target value
			// DoublePair.v2: instance weight
			List<Element<DoublePair>> pairs = new ArrayList<>(instances.size());
			for (Instance instance : instances) {
				double weight = instance.getWeight();
				double value = instance.getValue(func.attIndex);
				double target = instance.getTarget();
				pairs.add(new Element<DoublePair>(new DoublePair(target, weight), value));
			}
			Collections.sort(pairs);

			List<Double> uniqueValues = new ArrayList<>();
			List<DoublePair> stats = new ArrayList<>();
			getStats(pairs, uniqueValues, stats);

			build(func, uniqueValues, stats, limit);
		} else if (attribute.getType() == Attribute.Type.BINNED) {
			// Building histograms
			BinnedAttribute attr = (BinnedAttribute) attribute;
			DoublePair[] histogram = new DoublePair[attr.getNumBins()];
			for (int i = 0; i < histogram.length; i++) {
				histogram[i] = new DoublePair(0, 0);
			}
			for (Instance instance : instances) {
				int idx = (int) instance.getValue(func.attIndex);
				histogram[idx].v2 += instance.getTarget() * instance.getWeight();
				histogram[idx].v1 += instance.getWeight();
			}

			List<Double> uniqueValues = new ArrayList<>(histogram.length);
			List<DoublePair> stats = new ArrayList<>(histogram.length);
			for (int i = 0; i < histogram.length; i++) {
				if (histogram[i].v1 != 0) {
					stats.add(histogram[i]);
					uniqueValues.add((double) i);
				}
			}

			build(func, uniqueValues, stats, limit);
		} else {
			// Nominal attributes
			// Building histograms
			NominalAttribute attr = (NominalAttribute) attribute;
			DoublePair[] histogram = new DoublePair[attr.getStates().length];
			for (int i = 0; i < histogram.length; i++) {
				histogram[i] = new DoublePair(0, 0);
			}
			for (Instance instance : instances) {
				int idx = (int) instance.getValue(func.attIndex);
				histogram[idx].v2 += instance.getTarget() * instance.getWeight();
				histogram[idx].v1 += instance.getWeight();
			}

			List<Double> uniqueValues = new ArrayList<>(histogram.length);
			List<DoublePair> stats = new ArrayList<>(histogram.length);
			for (int i = 0; i < histogram.length; i++) {
				if (histogram[i].v1 != 0) {
					stats.add(histogram[i]);
					uniqueValues.add((double) i);
				}
			}

			build(func, uniqueValues, stats, limit);
		}

		if (lineSearch) {
			lineSearch(instances, func);
		}

		return func;
	}

	/**
	 * Builds a 1D function.
	 * 
	 * @param instances the training set.
	 * @param attribute the attribute.
	 * @param numIntervals the number of intervals.
	 * @return a 1D function.
	 */
	public Function1D build(Instances instances, Attribute attribute, int numIntervals) {
		Function1D func = new Function1D();
		func.attIndex = attribute.getIndex();

		if (attribute.getType() == Attribute.Type.NUMERIC) {
			// weight: attribute value
			// DoublePair.v1: target value
			// DoublePair.v2: instance weight
			List<Element<DoublePair>> pairs = new ArrayList<>(instances.size());
			for (Instance instance : instances) {
				double weight = instance.getWeight();
				double value = instance.getValue(func.attIndex);
				double target = instance.getTarget();
				pairs.add(new Element<DoublePair>(new DoublePair(target, weight), value));
			}
			Collections.sort(pairs);

			List<Double> uniqueValues = new ArrayList<>();
			List<DoublePair> stats = new ArrayList<>();
			getStats(pairs, uniqueValues, stats);

			build(func, uniqueValues, stats, numIntervals);
		} else if (attribute.getType() == Attribute.Type.BINNED) {
			// Building histograms
			BinnedAttribute attr = (BinnedAttribute) attribute;
			DoublePair[] histogram = new DoublePair[attr.getNumBins()];
			for (int i = 0; i < histogram.length; i++) {
				histogram[i] = new DoublePair(0, 0);
			}
			for (Instance instance : instances) {
				int idx = (int) instance.getValue(func.attIndex);
				histogram[idx].v2 += instance.getTarget() * instance.getWeight();
				histogram[idx].v1 += instance.getWeight();
			}

			List<Double> uniqueValues = new ArrayList<>(histogram.length);
			List<DoublePair> stats = new ArrayList<>(histogram.length);
			for (int i = 0; i < histogram.length; i++) {
				if (histogram[i].v1 != 0) {
					stats.add(histogram[i]);
					uniqueValues.add((double) i);
				}
			}

			build(func, uniqueValues, stats, numIntervals);
		} else {
			// Nominal attributes
			// Building histograms
			NominalAttribute attr = (NominalAttribute) attribute;
			DoublePair[] histogram = new DoublePair[attr.getStates().length];
			for (int i = 0; i < histogram.length; i++) {
				histogram[i] = new DoublePair(0, 0);
			}
			for (Instance instance : instances) {
				int idx = (int) instance.getValue(func.attIndex);
				histogram[idx].v2 += instance.getTarget() * instance.getWeight();
				histogram[idx].v1 += instance.getWeight();
			}

			List<Double> uniqueValues = new ArrayList<>(histogram.length);
			List<DoublePair> stats = new ArrayList<>(histogram.length);
			for (int i = 0; i < histogram.length; i++) {
				if (histogram[i].v1 != 0) {
					stats.add(histogram[i]);
					uniqueValues.add((double) i);
				}
			}

			build(func, uniqueValues, stats, numIntervals);
		}

		if (lineSearch) {
			lineSearch(instances, func);
		}

		return func;
	}

	/**
	 * Builds a 1D function.
	 * 
	 * @param instances the training set.
	 * @param attIndex the index in the attribute list of training set.
	 * @param alpha the alpha.
	 * @return a 1D function.
	 */
	public Function1D build(Instances instances, int attIndex, double alpha) {
		Attribute attribute = instances.getAttributes().get(attIndex);
		return build(instances, attribute, alpha);
	}

	/**
	 * Builds a 1D function.
	 * 
	 * @param instances the training set.
	 * @param attIndex the index in the attribute list of the training set.
	 * @param numIntervals the number of intervals.
	 * @return a 1D function.
	 */
	public Function1D build(Instances instances, int attIndex, int numIntervals) {
		Attribute attribute = instances.getAttributes().get(attIndex);
		return build(instances, attribute, numIntervals);
	}

	/**
	 * Returns <code>true</code> if line search is performed in the end.
	 * 
	 * @return <code>true</code> if line search is performed in the end.
	 */
	public boolean doLineSearch() {
		return lineSearch;
	}

	/**
	 * Returns the index in the attribute list of the training set.
	 * 
	 * @return the index in the attribute list of the training set.
	 */
	public int getAttributeIndex() {
		return attIndex;
	}

	/**
	 * Returns the number of intervals.
	 * 
	 * @return the number of intervals.
	 */
	public int getNumIntervals() {
		return numIntervals;
	}

	/**
	 * Sets the alpha.
	 * 
	 * @param alpha the minimum percentage of points in each interval.
	 */
	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}

	/**
	 * Sets the index in the attribute list of the training set.
	 * 
	 * @param attIndex the attribute index.
	 */
	public void setAttributeIndex(int attIndex) {
		this.attIndex = attIndex;
	}

	/**
	 * Sets whether we cut lines according to number of leaves or alpha.
	 * 
	 * @param leafLimited whether we cut lines according to number of leaves or alpha.
	 */
	public void setLeafLimited(boolean leafLimited) {
		this.leafLimited = leafLimited;
	}

	/**
	 * Sets <code>true</code> if line search is performed in the end.
	 * 
	 * @param lineSearch <code>true</code> if line search is performed in the end.
	 */
	public void setLineSearch(boolean lineSearch) {
		this.lineSearch = lineSearch;
	}

	/**
	 * Sets the number of intervals.
	 * 
	 * @param numIntervals the number of intervals.
	 */
	public void setNumIntervals(int numIntervals) {
		this.numIntervals = numIntervals;
	}

}
