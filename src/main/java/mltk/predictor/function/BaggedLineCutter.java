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
import mltk.core.Sampling;
import mltk.predictor.BaggedEnsemble;
import mltk.predictor.Learner;
import mltk.util.Random;
import mltk.util.tuple.IntPair;
import mltk.util.Element;
import mltk.util.MathUtils;
import mltk.util.OptimUtils;

public class BaggedLineCutter extends Learner {
	
	class Interval implements Comparable<Interval> {

		boolean finalized;
		int start;
		int end;
		// INF: Can split but split point not computed
		// NaN: Declared as leaf
		// Other: Split point
		double split;
		double sum;
		double weight;
		double value; // mean * sum, or sum * sum /weight; negative gain
		double gain;
		Interval left;
		Interval right;

		Interval() {
			split = Double.POSITIVE_INFINITY;
		}

		Interval(int start, int end, double sum, double weight) {
			this.start = start;
			this.end = end;
			this.split = Double.POSITIVE_INFINITY;
			this.sum = sum;
			this.weight = weight;
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
			return MathUtils.divide(sum, weight, 0.0);
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

	private int attIndex;

	private int numIntervals;
	
	private List<IntPair[]> samples;
	
	private boolean isClassification;
	

	/**
	 * Constructor.
	 */
	public BaggedLineCutter() {
		this(false);
	}
	
	/**
	 * Constructor.
	 * 
	 * @param isClassification {@code true} if it is a classification problem.
	 */
	public BaggedLineCutter(boolean isClassification) {
		this.isClassification = isClassification;
	}
	
	/**
	 * Initializes internal bootstrap samples.
	 * 
	 * @param n the size of the dataset to sample.
	 * @param baggingIters the number of bagging iterations.
	 */
	public void createBootstrapSamples(int n, int baggingIters) {
		samples = new ArrayList<>(baggingIters);
		if (baggingIters <= 0) {
			IntPair[] indices = new IntPair[n];
			for (int i = 0; i < n; i++) {
				indices[i] = new IntPair(i, 1);
			}
			samples.add(indices);
		} else {
			for (int b = 0; b < baggingIters; b++) {
				samples.add(Sampling.createBootstrapSampleIndices(n));
			}
		}
	}

	@Override
	public BaggedEnsemble build(Instances instances) {
		return build(instances, attIndex, numIntervals);
	}

	/**
	 * Builds a 1D function.
	 * 
	 * @param instances the training set.
	 * @param attribute the attribute.
	 * @param numIntervals the number of intervals.
	 * @return a 1D function.
	 */
	public BaggedEnsemble build(Instances instances, Attribute attribute, int numIntervals) {
		int attIndex = attribute.getIndex();
		
		BaggedEnsemble ensemble = new BaggedEnsemble(samples.size());
		
		double[] targets = new double[instances.size()];
		double[] fvalues = new double[instances.size()];
		double[] weights = new double[instances.size()];
		for (int i = 0; i < instances.size(); i++) {
			Instance instance = instances.get(i);
			targets[i] = instance.getTarget();
			fvalues[i] = instance.getValue(attribute);
			weights[i] = instance.getWeight();
		}
		
		if (attribute.getType() == Attribute.Type.NUMERIC) {
			for (IntPair[] indices : samples) {
				double sumRespOnMV = 0.0;
				double sumWeightOnMV = 0.0;
				
				List<Element<double[]>> pairs = new ArrayList<>(instances.size());
				for (IntPair entry : indices) {
					int index = entry.v1;
					int w = entry.v2;
					double weight = weights[index];
					double value = fvalues[index];
					double target = targets[index];
					if (!Double.isNaN(value)) {
						if (isClassification) {
							pairs.add(new Element<double[]>(new double[] { target * w, weight * w }, value));
						} else {
							pairs.add(new Element<double[]>(new double[] { target * weight * w, weight * w }, value));
						}
					} else {
						if (isClassification) {
							sumRespOnMV += target * w;
						} else {
							sumRespOnMV += target * weight * w;
						}
						sumWeightOnMV += weight * w;
					}
				}
				
				Collections.sort(pairs);
				List<double[]> histograms = new ArrayList<>();
				getHistograms(pairs, histograms);
				histograms.add(new double[] { Double.NaN, sumRespOnMV, sumWeightOnMV });
				
				Function1D func = build(attIndex, histograms, numIntervals);
				ensemble.add(func);
			}
		} else {
			int size = 0;
			if (attribute.getType() == Attribute.Type.BINNED) {
				size = ((BinnedAttribute) attribute).getNumBins();
			} else {
				size = ((NominalAttribute) attribute).getCardinality();
			}
			for (IntPair[] indices : samples) {
				List<double[]> histograms;
				double sumRespOnMV = 0.0;
				double sumWeightOnMV = 0.0;
				
				double[][] histogram = new double[size][3];
				for (IntPair entry : indices) {
					int index = entry.v1;
					int w = entry.v2;
					double weight = weights[index];
					double value = fvalues[index];
					double target = targets[index];
					
					if (!Double.isNaN(value)) {
						int idx = (int) value;
						if (isClassification) {
							histogram[idx][0] += target * w;
						} else {
							histogram[idx][0] += target * weight * w;
						}
						histogram[idx][1] += weight * w;
					} else {
						if (isClassification) {
							sumRespOnMV += target * w;
						} else {
							sumRespOnMV += target * weight * w;
						}
						sumWeightOnMV += weight * w;
					}
				}

				histograms = new ArrayList<>(histogram.length + 1);
				for (int i = 0; i < histogram.length; i++) {
					if (!MathUtils.isZero(histogram[i][1])) {
						double[] hist = histogram[i];
						histograms.add(new double[] {i, hist[0], hist[1]});
					}
				}
				histograms.add(new double[] { Double.NaN, sumRespOnMV, sumWeightOnMV });
				
				Function1D func = build(attIndex, histograms, numIntervals);
				ensemble.add(func);
			}
		}
		
		return ensemble;
	}

	public BaggedEnsemble build(Instances instances, int attIndex, int numIntervals) {
		Attribute attribute = instances.getAttributes().get(attIndex);
		return build(instances, attribute, numIntervals);
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
	 * Sets the index in the attribute list of the training set.
	 * 
	 * @param attIndex the attribute index.
	 */
	public void setAttributeIndex(int attIndex) {
		this.attIndex = attIndex;
	}

	/**
	 * Sets the number of intervals.
	 * 
	 * @param numIntervals the number of intervals.
	 */
	public void setNumIntervals(int numIntervals) {
		this.numIntervals = numIntervals;
	}
	
	protected void getHistograms(List<Element<double[]>> pairs, List<double[]> histograms) {
		if (pairs.size() == 0) {
			return;
		}
		// Element(new double[] {sum, weight}, value)
		double[] hist = pairs.get(0).element;
		double lastValue = pairs.get(0).weight;
		double sum = hist[0];
		double weight = hist[1];
		for (int i = 1; i < pairs.size(); i++) {
			Element<double[]> element = pairs.get(i);
			hist = element.element;
			double value = element.weight;
			double s = hist[0];
			double w = hist[1];
			if (value != lastValue) {
				histograms.add(new double[] { lastValue, sum, weight });
				lastValue = value;
				sum = s;
				weight = w;
			} else {
				sum += s;
				weight += w;
			}
		}
		histograms.add(new double[] { lastValue, sum, weight });
	}
	
	protected Function1D build(int attIndex, List<double[]> histograms, int numIntervals) {
		Function1D func = new Function1D();
		func.attIndex = attIndex;
		// [feature value, sum, weight]
		double[] histOnMV = histograms.get(histograms.size() - 1);
		func.predictionOnMV = MathUtils.divide(histOnMV[1], histOnMV[2], 0.0);
		
		// 1. Check basic leaf conditions
		if (histograms.size() <= 2) {
			func.splits = new double[] { Double.POSITIVE_INFINITY };
			double prediction = 0.0;
			if (histograms.size() == 2) {
				double[] hist = histograms.get(0);
				prediction = MathUtils.divide(hist[1], hist[2], 0.0);
			}
			func.predictions = new double[] { prediction };
			return func;
		}

		// 2. Cut the line
		// 2.1 First cut
		double[] stats = sumUp(histograms, 0, histograms.size() - 1);
		Interval root = new Interval(0, histograms.size() - 1, stats[0], stats[1]);
		split(histograms, root);

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
				split(histograms, parent.left);
				split(histograms, parent.right);

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
		return func;
	}
	
	protected double[] sumUp(List<double[]> histograms, int start, int end) {
		double sum = 0;
		double weight = 0;
		for (int i = start; i < end; i++) {
			double[] hist = histograms.get(i);
			sum += hist[1];
			weight += hist[2];
		}
		return new double[] { sum, weight };
	}

	protected void split(List<double[]> histograms, Interval parent) {
		split(histograms, parent, 5);
	}

	protected void split(List<double[]> histograms, Interval parent, double limit) {
		// Test if we need to split
		if (parent.weight <= limit || parent.end - parent.start <= 1) {
			parent.split = Double.NaN; // Declared as leaf
		} else {
			parent.left = new Interval();
			parent.right = new Interval();
			int start = parent.left.start = parent.start;
			int end = parent.right.end = parent.end;
			final double sum = parent.sum;
			final double totalWeight = parent.weight;

			double sum1 = histograms.get(start)[1];
			double sum2 = sum - sum1;
			double weight1 = histograms.get(start)[2];
			double weight2 = totalWeight - weight1;

			double bestEval = -(OptimUtils.getGain(sum1, weight1) + OptimUtils.getGain(sum2, weight2));
			List<double[]> splits = new ArrayList<>();
			splits.add(new double[] { (histograms.get(start)[0] + histograms.get(start + 1)[0]) / 2, start, sum1,
					weight1, sum2, weight2 });
			for (int i = start + 1; i < end - 1; i++) {
				double[] hist = histograms.get(i);
				final double s = hist[1];
				final double w = hist[2];
				sum1 += s;
				sum2 -= s;
				weight1 += w;
				weight2 -= w;
				double eval1 = OptimUtils.getGain(sum1, weight1);
				double eval2 = OptimUtils.getGain(sum2, weight2);
				double eval = -(eval1 + eval2);
				if (eval <= bestEval) {
					double split = (histograms.get(i)[0] + histograms.get(i + 1)[0]) / 2;
					if (eval < bestEval) {
						bestEval = eval;
						splits.clear();
					}
					splits.add(new double[] { split, i, sum1, weight1, sum2, weight2 });
				}
			}
			Random rand = Random.getInstance();
			double[] split = splits.get(rand.nextInt(splits.size()));
			parent.split = split[0];
			parent.left.end = (int) split[1] + 1;
			parent.right.start = (int) split[1] + 1;
			parent.left.sum = split[2];
			parent.left.weight = split[3];
			parent.right.sum = split[4];
			parent.right.weight = split[5];
			parent.gain = OptimUtils.getGain(parent.left.sum, parent.left.weight)
					+ OptimUtils.getGain(parent.right.sum, parent.right.weight);
			parent.value = -parent.gain + (sum / totalWeight * sum);
		}
	}
	
	protected void inorder(Interval parent, List<Double> splits, List<Double> predictions) {
		if (parent.isFinalized()) {
			inorder(parent.left, splits, predictions);
			splits.add(parent.split);
			inorder(parent.right, splits, predictions);
		} else {
			predictions.add(parent.getPrediction());
		}
	}

}
