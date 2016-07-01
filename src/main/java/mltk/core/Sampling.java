package mltk.core;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import mltk.util.Random;

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
	 * Returns a set of bags.
	 * 
	 * @param instances the dataset.
	 * @param baggingIter the number of bagging iterations.
	 * @return a set of bags.
	 */
	public static Instances[] createBags(Instances instances, int baggingIter) {
		Instances[] bags = null;
		if (baggingIter <= 0) {
			// No bagging
			bags = new Instances[] { instances };
		} else {
			bags = new Instances[baggingIter];
			for (int i = 0; i < baggingIter; i++) {
				bags[i] = Sampling.createBootstrapSample(instances);
			}
		}
		return bags;
	}

}
