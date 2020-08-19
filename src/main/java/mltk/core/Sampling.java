package mltk.core;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import mltk.util.Permutation;
import mltk.util.Random;
import mltk.util.tuple.IntPair;

/**
 * Class for creating samples.
 * 
 * @author Yin Lou
 * 
 */
public class Sampling {

	/**
	 * Returns a bootstrap sample.
	 * 
	 * @param instances the data set.
	 * @return a bootstrap sample.
	 */
	public static Instances createBootstrapSample(Instances instances) {
		Random rand = Random.getInstance();
		Map<Integer, Integer> map = new HashMap<>();
		for (int i = 0; i < instances.size(); i++) {
			int idx = rand.nextInt(instances.size());
			map.put(idx, map.getOrDefault(idx, 0) + 1);
		}
		Instances bag = new Instances(instances.getAttributes(), instances.getTargetAttribute(), map.size());
		for (Integer idx : map.keySet()) {
			int weight = map.get(idx);
			Instance instance = instances.get(idx).clone();
			instance.setWeight(weight);
			bag.add(instance);
		}
		return bag;
	}

	/**
	 * Returns a bootstrap sample with out-of-bag samples.
	 * 
	 * @param instances the data set.
	 * @param bagIndices the index of sampled instances with weights.
	 * @param oobIndices the out-of-bag indexes.
	 */
	public static void createBootstrapSample(Instances instances, Map<Integer, Integer> bagIndices,
			List<Integer> oobIndices) {
		Random rand = Random.getInstance();
		for (;;) {
			bagIndices.clear();
			oobIndices.clear();
			for (int i = 0; i < instances.size(); i++) {
				int idx = rand.nextInt(instances.size());
				bagIndices.put(idx, bagIndices.getOrDefault(idx, 0) + 1);
			}
			for (int i = 0; i < instances.size(); i++) {
				if (!bagIndices.containsKey(i)) {
					oobIndices.add(i);
				}
			}
			if (oobIndices.size() > 0) {
				break;
			}
		}
	}
	
	/**
	 * Returns a bootstrap sample of indices and weights.
	 * 
	 * @param n the size of the dataset to sample.
	 * @return a bootstrap sample of indices and weights.
	 */
	public static IntPair[] createBootstrapSampleIndices(int n) {
		Random rand = Random.getInstance();
		Map<Integer, Integer> map = new HashMap<>();
		for (int i = 0; i < n; i++) {
			int idx = rand.nextInt(n);
			map.put(idx, map.getOrDefault(idx, 0) + 1);
		}
		IntPair[] indices = new IntPair[map.size()];
		int k = 0;
		for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
			indices[k++] = new IntPair(entry.getKey(), entry.getValue());
		}
		return indices;
	}

	/**
	 * Returns a subsample.
	 * 
	 * @param instances the dataset.
	 * @param n the sample size.
	 * @return a subsample.
	 */
	public static Instances createSubsample(Instances instances, int n) {
		Permutation perm = new Permutation(instances.size());
		perm.permute();
		int[] a = perm.getPermutation();
		Instances sample = new Instances(instances.getAttributes(), instances.getTargetAttribute(), n);
		for (int i = 0; i < n; i++) {
			sample.add(instances.get(a[i]));
		}
		return sample;
	}

	/**
	 * Returns a set of bags.
	 * 
	 * @param instances the dataset.
	 * @param b the number of bagging iterations.
	 * @return a set of bags.
	 */
	public static Instances[] createBags(Instances instances, int b) {
		Instances[] bags = null;
		if (b <= 0) {
			// No bagging
			bags = new Instances[] { instances };
		} else {
			bags = new Instances[b];
			for (int i = 0; i < b; i++) {
				bags[i] = Sampling.createBootstrapSample(instances);
			}
		}
		return bags;
	}

}
