package mltk.predictor.tree;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import mltk.core.Attribute;
import mltk.core.Attribute.Type;
import mltk.core.BinnedAttribute;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.NominalAttribute;
import mltk.util.ArrayUtils;
import mltk.util.OptimUtils;
import mltk.util.Random;
import mltk.util.StatUtils;
import mltk.util.tuple.DoublePair;
import mltk.util.tuple.IntDoublePair;
import mltk.util.tuple.LongDoublePair;
import mltk.util.tuple.LongDoublePairComparator;

/**
 * Class for learning decision tables. 
 * 
 * <p>
 * Reference:<br>
 * Y. Lou and M. Obukhov. BDT: Boosting Decision Tables for High Accuracy and Scoring Efficiency. In <i>Proceedings of the
 * 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)</i>, Halifax, Nova Scotia, Canada, 2017.
 * </p>
 * 
 * This class has a different implementation to better fit the design of this package.
 * 
 * @author Yin Lou
 *
 */
public class DecisionTableLearner extends RTreeLearner {
	
	/**
	 * Enumeration of construction mode.
	 *
	 * @author Yin Lou
	 *
	 */
	public enum Mode {

		ONE_PASS_GREEDY, MULTI_PASS_CYCLIC, MULTI_PASS_RANDOM;
	}
	
	protected Mode mode;
	protected int maxDepth;
	protected int numPasses;
	
	/**
	 * Constructor.
	 */
	public DecisionTableLearner() {
		mode = Mode.ONE_PASS_GREEDY;
		maxDepth = 6;
		numPasses = 2;
	}
	
	@Override
	public void setParameters(String mode) {
		String[] data = mode.split(":");
		if (data.length != 2) {
			throw new IllegalArgumentException();
		}
		this.setMaxDepth(Integer.parseInt(data[1]));
		switch (data[0]) {
			case "g":
				this.setConstructionMode(Mode.ONE_PASS_GREEDY);
				break;
			case "c":
				this.setConstructionMode(Mode.MULTI_PASS_CYCLIC);
				this.setNumPasses(2);
				break;
			case "r":
				this.setConstructionMode(Mode.MULTI_PASS_RANDOM);
				this.setNumPasses(2);
				break;
			default:
				throw new IllegalArgumentException();
		}
	}
	
	@Override
	public boolean isRobust() {
		return false;
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
	 * Sets the construction mode.
	 *
	 * @param mode the construction mode.
	 */
	public void setConstructionMode(Mode mode) {
		this.mode = mode;
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
	 * Sets the maximum depth.
	 *
	 * @param maxDepth the maximum depth.
	 */
	public void setMaxDepth(int maxDepth) {
		this.maxDepth = maxDepth;
	}
	
	/**
	 * Returns the number of passes. This parameter is used in multi-pass cyclic mode.
	 * 
	 * @return the number of passes.
	 */
	public int getNumPasses() {
		return numPasses;
	}
	
	/**
	 * Sets the number of passes.
	 * 
	 * @param numPasses the number of passes.
	 */
	public void setNumPasses(int numPasses) {
		this.numPasses = numPasses;
	}

	@Override
	public DecisionTable build(Instances instances) {
		DecisionTable ot = null;
		switch (mode) {
			case ONE_PASS_GREEDY:
				ot = buildOnePassGreedy(instances, maxDepth);
				break;
			case MULTI_PASS_CYCLIC:
				ot = buildMultiPassCyclic(instances, maxDepth, numPasses);
				break;
			case MULTI_PASS_RANDOM:
				ot = buildMultiPassRandom(instances, maxDepth, numPasses);
			default:
				break;
		}
		return ot;
	}
	
	/**
	 * Builds a standard oblivious regression tree using greedy tree induction.
	 * 
	 * @param instances the training set.
	 * @param maxDepth the maximum depth.
	 * @return an oblivious regression tree.
	 */
	public DecisionTable buildOnePassGreedy(Instances instances, int maxDepth) {
		// stats[0]: totalWeights
		// stats[1]: sum
		// stats[2]: weightedMean
		double[] stats = new double[3];
		Map<Long, Dataset> map = new HashMap<>(instances.size());
		List<Integer> attList = new ArrayList<>(maxDepth);
		List<Double> splitList = new ArrayList<>(maxDepth);
		Dataset dataset = null;
		if (this.cache != null) {
			dataset = Dataset.create(this.cache, instances);
		} else {
			dataset = Dataset.create(instances);
		}
		map.put(Long.valueOf(0L), dataset);
		
		if (maxDepth <= 0) {
			getStats(dataset.instances, stats);
			final double weightedMean = stats[2];
			return new DecisionTable(
					new int[] {},
					new double[] {}, 
					new long[] { 0L },
					new double[] { weightedMean });
		}
		
		List<Attribute> attributes = instances.getAttributes();
		List<List<Double>> featureValues = new ArrayList<>(attributes.size());
		for (int j = 0; j < attributes.size(); j++) {
			Attribute attribute = attributes.get(j);
			List<Double> values = new ArrayList<>();
			
			if (attribute.getType() == Type.BINNED) {
				int numBins = ((BinnedAttribute) attribute).getNumBins();
				for (int i = 0; i < numBins; i++) {
					values.add((double) i);
				}
			} else if (attribute.getType() == Type.NOMINAL) {
				int cardinality = ((NominalAttribute) attribute).getCardinality();
				for (int i = 0; i < cardinality; i++) {
					values.add((double) i);
				}
			} else {
				Set<Double> set = new HashSet<>();
				for (Instance instance : instances) {
					set.add(instance.getValue(attribute));
				}
				values.addAll(set);
				Collections.sort(values);
			}
			featureValues.add(values);
		}
		
		for (int d = 0; d < maxDepth; d++) {
			double bestGain = Double.NEGATIVE_INFINITY;
			List<IntDoublePair> splitCandidates = new ArrayList<>();
			
			for (int j = 0; j < attributes.size(); j++) {
				List<Double> values = featureValues.get(j);
				if (values.size() <= 1) {
					continue;
				}
				
				Attribute attribute = attributes.get(j);
				int attIndex = attribute.getIndex();
				String attName = attribute.getName();
				
				double[] gains = new double[values.size() - 1];
				for (Dataset data : map.values()) {
					getStats(data.instances, stats);
					final double totalWeights = stats[0];
					final double sum = stats[1];
					
					List<IntDoublePair> sortedList = data.sortedLists.get(attName);
					List<Double> uniqueValues = new ArrayList<>(sortedList.size());
					List<DoublePair> histogram = new ArrayList<>(sortedList.size());
					getHistogram(data.instances, sortedList, uniqueValues, totalWeights, sum, histogram);
					double[] localGains = evalSplits(uniqueValues, histogram, totalWeights, sum);
					processGains(uniqueValues, localGains, values, gains);
				}
				
				int idx = StatUtils.indexOfMax(gains);
				if (bestGain <= gains[idx]) {
					double split = (values.get(idx) + values.get(idx + 1)) / 2;
					if (bestGain < gains[idx]) {
						bestGain = gains[idx];
						splitCandidates.clear();
					}
					splitCandidates.add(new IntDoublePair(attIndex, split));
				}
			}
			
			if (splitCandidates.size() == 0) {
				break;
			}
			
			Random rand = Random.getInstance();
			IntDoublePair split = splitCandidates.get(rand.nextInt(splitCandidates.size()));
			attList.add(split.v1);
			splitList.add(split.v2);
			
			Map<Long, Dataset> mapNew = new HashMap<>(map.size() * 2);
			for (Map.Entry<Long, Dataset> entry : map.entrySet()) {
				Long key = entry.getKey();
				Dataset data = entry.getValue();
				Dataset left = new Dataset(data.instances);
				Dataset right = new Dataset(data.instances);
				data.split(split.v1, split.v2, left, right);
				if (left.instances.size() > 0) {
					Long leftKey = (key << 1) | 1L;
					mapNew.put(leftKey, left);
				}
				if (right.instances.size() > 0) {
					Long rightKey = key << 1;
					mapNew.put(rightKey, right);
				}
			}
			map = mapNew;
		}
		
		int[] attIndices = ArrayUtils.toIntArray(attList);
		double[] splits = ArrayUtils.toDoubleArray(splitList);
		
		List<LongDoublePair> list = new ArrayList<>(splits.length);
		for (Map.Entry<Long, Dataset> entry : map.entrySet()) {
			Long key = entry.getKey();
			Dataset data = entry.getValue();
			getStats(data.instances, stats);
			list.add(new LongDoublePair(key, stats[2]));
		}
		Collections.sort(list, new LongDoublePairComparator());
		long[] predIndices = new long[list.size()];
		double[] predValues = new double[list.size()];
		for (int i = 0; i < predIndices.length; i++) {
			LongDoublePair pair = list.get(i);
			predIndices[i] = pair.v1;
			predValues[i] = pair.v2;
		}
		
		return new DecisionTable(attIndices, splits, predIndices, predValues);
	}
	
	/**
	 * Builds an oblivious regression tree using multi-pass cyclic backfitting.
	 * 
	 * @param instances the training set.
	 * @param maxDepth the maximum depth.
	 * @param numPasses the number of passes.
	 * @return an oblivious regression tree.
	 */
	public DecisionTable buildMultiPassCyclic(Instances instances, int maxDepth, int numPasses) {
		// stats[0]: totalWeights
		// stats[1]: sum
		// stats[2]: weightedMean
		double[] stats = new double[3];
		Map<Long, Dataset> map = new HashMap<>(instances.size());
		int[] attIndices = new int[maxDepth];
		double[] splits = new double[maxDepth];
		Dataset dataset = null;
		if (this.cache != null) {
			dataset = Dataset.create(this.cache, instances);
		} else {
			dataset = Dataset.create(instances);
		}
		map.put(Long.valueOf(0L), dataset);
		
		if (maxDepth <= 0) {
			getStats(dataset.instances, stats);
			final double weightedMean = stats[2];
			return new DecisionTable(
					new int[] {},
					new double[] {}, 
					new long[] { 0L },
					new double[] { weightedMean });
		}
		
		List<Attribute> attributes = instances.getAttributes();
		List<List<Double>> featureValues = new ArrayList<>(attributes.size());
		for (int j = 0; j < attributes.size(); j++) {
			Attribute attribute = attributes.get(j);
			List<Double> values = new ArrayList<>();
			
			if (attribute.getType() == Type.BINNED) {
				int numBins = ((BinnedAttribute) attribute).getNumBins();
				for (int i = 0; i < numBins; i++) {
					values.add((double) i);
				}
			} else if (attribute.getType() == Type.NOMINAL) {
				int cardinality = ((NominalAttribute) attribute).getCardinality();
				for (int i = 0; i < cardinality; i++) {
					values.add((double) i);
				}
			} else {
				Set<Double> set = new HashSet<>();
				for (Instance instance : instances) {
					set.add(instance.getValue(attribute));
				}
				values.addAll(set);
				Collections.sort(values);
			}
			featureValues.add(values);
		}
		
		for (int pass = 0; pass < numPasses; pass++) {
			for (int d = 0; d < maxDepth; d++) {
				double bestGain = Double.NEGATIVE_INFINITY;
				List<IntDoublePair> splitCandidates = new ArrayList<>();
				
				// Remove depth d
				Set<Long> processedKeys = new HashSet<>();
				Map<Long, Dataset> mapNew = new HashMap<>(map.size());
				for (Map.Entry<Long, Dataset> entry : map.entrySet()) {
					Long key = entry.getKey();
					if (processedKeys.contains(key)) {
						continue;
					}
					Dataset data = entry.getValue();
					int s = maxDepth - d - 1;
					Long otherKey = key ^ (1L << s);
					if (map.containsKey(otherKey)) {
						long check = (key >> s) & 1;
						Dataset left = null;
						Dataset right = null;
						if (check > 0) {
							left = data;
							right = map.get(otherKey);
						} else {
							left = map.get(otherKey);
							right = data;
						}
						// This key will be updated anyway
						mapNew.put(key, Dataset.merge(left, right));
						processedKeys.add(key);
						processedKeys.add(otherKey);
					} else {
						mapNew.put(key, data);
					}
				}
				map = mapNew;
				
				for (int j = 0; j < attributes.size(); j++) {
					Attribute attribute = attributes.get(j);
					int attIndex = attribute.getIndex();
					String attName = attribute.getName();
					
					List<Double> values = featureValues.get(j);
					if (values.size() <= 1) {
						continue;
					}
					
					double[] gains = new double[values.size() - 1];
					for (Dataset data : map.values()) {
						getStats(data.instances, stats);
						final double totalWeights = stats[0];
						final double sum = stats[1];
						
						List<IntDoublePair> sortedList = data.sortedLists.get(attName);
						List<Double> uniqueValues = new ArrayList<>(sortedList.size());
						List<DoublePair> histogram = new ArrayList<>(sortedList.size());
						getHistogram(data.instances, sortedList, uniqueValues, totalWeights, sum, histogram);
						double[] localGains = evalSplits(uniqueValues, histogram, totalWeights, sum);
						processGains(uniqueValues, localGains, values, gains);
					}
					
					int idx = StatUtils.indexOfMax(gains);
					if (bestGain <= gains[idx]) {
						double split = (values.get(idx) + values.get(idx + 1)) / 2;
						if (bestGain < gains[idx]) {
							bestGain = gains[idx];
							splitCandidates.clear();
						}
						splitCandidates.add(new IntDoublePair(attIndex, split));
					}
				}
				
				if (splitCandidates.size() == 0) {
					break;
				}
				
				Random rand = Random.getInstance();
				IntDoublePair split = splitCandidates.get(rand.nextInt(splitCandidates.size()));
				attIndices[d] = split.v1;
				splits[d] = split.v2;
				
				mapNew = new HashMap<>(map.size() * 2);
				for (Map.Entry<Long, Dataset> entry : map.entrySet()) {
					Long key = entry.getKey();
					Dataset data = entry.getValue();
					Dataset left = new Dataset(data.instances);
					Dataset right = new Dataset(data.instances);
					data.split(split.v1, split.v2, left, right);
					int s = maxDepth - d - 1;
					if (left.instances.size() > 0) {
						Long leftKey = key | (1L << s);
						mapNew.put(leftKey, left);
					}
					if (right.instances.size() > 0) {
						Long rightKey = key & ~(1L << s);
						mapNew.put(rightKey, right);
					}
				}
				map = mapNew;
			}
		}
		
		List<LongDoublePair> list = new ArrayList<>(splits.length);
		for (Map.Entry<Long, Dataset> entry : map.entrySet()) {
			Long key = entry.getKey();
			Dataset data = entry.getValue();
			getStats(data.instances, stats);
			list.add(new LongDoublePair(key, stats[2]));
		}
		Collections.sort(list, new LongDoublePairComparator());
		long[] predIndices = new long[list.size()];
		double[] predValues = new double[list.size()];
		for (int i = 0; i < predIndices.length; i++) {
			LongDoublePair pair = list.get(i);
			predIndices[i] = pair.v1;
			predValues[i] = pair.v2;
		}
		
		return new DecisionTable(attIndices, splits, predIndices, predValues);
	}
	
	/**
	 * Builds an oblivious regression tree using multi-pass random backfitting.
	 * 
	 * @param instances the training set.
	 * @param maxDepth the maximum depth.
	 * @param numPasses the number of passes.
	 * @return an oblivious regression tree.
	 */
	public DecisionTable buildMultiPassRandom(Instances instances, int maxDepth, int numPasses) {
		// stats[0]: totalWeights
		// stats[1]: sum
		// stats[2]: weightedMean
		double[] stats = new double[3];
		Map<Long, Dataset> map = new HashMap<>(instances.size());
		int[] attIndices = new int[maxDepth];
		double[] splits = new double[maxDepth];
		Dataset dataset = null;
		if (this.cache != null) {
			dataset = Dataset.create(this.cache, instances);
		} else {
			dataset = Dataset.create(instances);
		}
		map.put(Long.valueOf(0L), dataset);
		
		if (maxDepth <= 0) {
			getStats(dataset.instances, stats);
			final double weightedMean = stats[2];
			return new DecisionTable(
					new int[] {},
					new double[] {}, 
					new long[] { 0L },
					new double[] { weightedMean });
		}
		
		List<Attribute> attributes = instances.getAttributes();
		List<List<Double>> featureValues = new ArrayList<>(attributes.size());
		for (int j = 0; j < attributes.size(); j++) {
			Attribute attribute = attributes.get(j);
			List<Double> values = new ArrayList<>();
			
			if (attribute.getType() == Type.BINNED) {
				int numBins = ((BinnedAttribute) attribute).getNumBins();
				for (int i = 0; i < numBins; i++) {
					values.add((double) i);
				}
			} else if (attribute.getType() == Type.NOMINAL) {
				int cardinality = ((NominalAttribute) attribute).getCardinality();
				for (int i = 0; i < cardinality; i++) {
					values.add((double) i);
				}
			} else {
				Set<Double> set = new HashSet<>();
				for (Instance instance : instances) {
					set.add(instance.getValue(attribute));
				}
				values.addAll(set);
				Collections.sort(values);
			}
			featureValues.add(values);
		}
		
		for (int iter = 0; iter < numPasses; iter++) {
			for (int k = 0; k < maxDepth; k++) {
				double bestGain = Double.NEGATIVE_INFINITY;
				List<IntDoublePair> splitCandidates = new ArrayList<>();
				
				int d = k;
				if (iter > 0) {
					d = Random.getInstance().nextInt(maxDepth);
				}
				
				// Remove depth d
				Set<Long> processedKeys = new HashSet<>();
				Map<Long, Dataset> mapNew = new HashMap<>(map.size());
				for (Map.Entry<Long, Dataset> entry : map.entrySet()) {
					Long key = entry.getKey();
					if (processedKeys.contains(key)) {
						continue;
					}
					Dataset data = entry.getValue();
					int s = maxDepth - d - 1;
					Long otherKey = key ^ (1L << s);
					if (map.containsKey(otherKey)) {
						long check = (key >> s) & 1;
						Dataset left = null;
						Dataset right = null;
						if (check > 0) {
							left = data;
							right = map.get(otherKey);
						} else {
							left = map.get(otherKey);
							right = data;
						}
						// This key will be updated anyway
						mapNew.put(key, Dataset.merge(left, right));
						processedKeys.add(key);
						processedKeys.add(otherKey);
					} else {
						mapNew.put(key, data);
					}
				}
				map = mapNew;
				
				for (int j = 0; j < attributes.size(); j++) {
					Attribute attribute = attributes.get(j);
					int attIndex = attribute.getIndex();
					String attName = attribute.getName();
					
					List<Double> values = featureValues.get(j);
					if (values.size() <= 1) {
						continue;
					}
					
					double[] gains = new double[values.size() - 1];
					for (Dataset data : map.values()) {
						getStats(data.instances, stats);
						final double totalWeights = stats[0];
						final double sum = stats[1];
						
						List<IntDoublePair> sortedList = data.sortedLists.get(attName);
						List<Double> uniqueValues = new ArrayList<>(sortedList.size());
						List<DoublePair> histogram = new ArrayList<>(sortedList.size());
						getHistogram(data.instances, sortedList, uniqueValues, totalWeights, sum, histogram);
						double[] localGains = evalSplits(uniqueValues, histogram, totalWeights, sum);
						processGains(uniqueValues, localGains, values, gains);
					}
					
					int idx = StatUtils.indexOfMax(gains);
					if (bestGain <= gains[idx]) {
						double split = (values.get(idx) + values.get(idx + 1)) / 2;
						if (bestGain < gains[idx]) {
							bestGain = gains[idx];
							splitCandidates.clear();
						}
						splitCandidates.add(new IntDoublePair(attIndex, split));
					}
				}
				
				if (splitCandidates.size() == 0) {
					break;
				}
				
				Random rand = Random.getInstance();
				IntDoublePair split = splitCandidates.get(rand.nextInt(splitCandidates.size()));
				attIndices[d] = split.v1;
				splits[d] = split.v2;
				
				mapNew = new HashMap<>(map.size() * 2);
				for (Map.Entry<Long, Dataset> entry : map.entrySet()) {
					Long key = entry.getKey();
					Dataset data = entry.getValue();
					Dataset left = new Dataset(data.instances);
					Dataset right = new Dataset(data.instances);
					data.split(split.v1, split.v2, left, right);
					int s = maxDepth - d - 1;
					if (left.instances.size() > 0) {
						Long leftKey = key | (1L << s);
						mapNew.put(leftKey, left);
					}
					if (right.instances.size() > 0) {
						Long rightKey = key & ~(1L << s);
						mapNew.put(rightKey, right);
					}
				}
				map = mapNew;
			}
		}
		
		List<LongDoublePair> list = new ArrayList<>(splits.length);
		for (Map.Entry<Long, Dataset> entry : map.entrySet()) {
			Long key = entry.getKey();
			Dataset data = entry.getValue();
			getStats(data.instances, stats);
			list.add(new LongDoublePair(key, stats[2]));
		}
		Collections.sort(list, new LongDoublePairComparator());
		long[] predIndices = new long[list.size()];
		double[] predValues = new double[list.size()];
		for (int i = 0; i < predIndices.length; i++) {
			LongDoublePair pair = list.get(i);
			predIndices[i] = pair.v1;
			predValues[i] = pair.v2;
		}
		
		return new DecisionTable(attIndices, splits, predIndices, predValues);
	}
	
	protected void processGains(List<Double> uniqueValues, double[] localGains, List<Double> values, double[] gains) {
		int i = 0;
		int j = 0;
		double noSplitGain = localGains[localGains.length - 1];
		double minV = uniqueValues.get(0);
		while (j < gains.length) {
			double v2 = values.get(j);
			if (v2 < minV) {
				gains[j] += noSplitGain;
				j++;
			} else {
				break;
			}
		}
		double prevGain = localGains[i];
		while (i < localGains.length && j < gains.length) {
			double v1 = uniqueValues.get(i);
			double v2 = values.get(j);
			if (v1 == v2) {
				gains[j] += localGains[i];
				prevGain = localGains[i];
				i++;
				j++;
			}
			while (v1 > v2) {
				gains[j] += prevGain;
				j++;
				v2 = values.get(j);
			}
		}
		while (j < gains.length) {
			gains[j] += noSplitGain;
			j++;
		}
	}
	
	protected double[] evalSplits(List<Double> uniqueValues, List<DoublePair> hist, double totalWeights, double sum) {
		double weight1 = hist.get(0).v1;
		double weight2 = totalWeights - weight1;
		double sum1 = hist.get(0).v2;
		double sum2 = sum - sum1;

		double[] gains = new double[uniqueValues.size()];
		
		gains[0] = OptimUtils.getGain(sum1, weight1) + OptimUtils.getGain(sum2, weight2);
		for (int i = 1; i < uniqueValues.size() - 1; i++) {
			final double w = hist.get(i).v1;
			final double s = hist.get(i).v2;
			weight1 += w;
			weight2 -= w;
			sum1 += s;
			sum2 -= s;
			gains[i] = OptimUtils.getGain(sum1, weight1) + OptimUtils.getGain(sum2, weight2);
		}
		// gain for no split
		gains[uniqueValues.size() - 1] = OptimUtils.getGain(sum, totalWeights);
		
		return gains;
	}

}
