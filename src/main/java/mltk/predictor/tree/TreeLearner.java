package mltk.predictor.tree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import mltk.core.Attribute;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.SparseVector;
import mltk.predictor.Learner;
import mltk.util.tuple.IntDoublePair;
import mltk.util.tuple.IntDoublePairComparator;

/**
 * Abstract class for learning trees.
 * 
 * @author Yin Lou
 *
 */
public abstract class TreeLearner extends Learner {
	
	protected static final Double ZERO = new Double(0.0);
	protected static final IntDoublePairComparator COMP = new IntDoublePairComparator(false);
	
	protected Dataset cache;
	
	/**
	 * Caches the auxiliary data structures. This method is used in ensemble method
	 * so that same data structures can be shared across iterations.
	 * 
	 * @param instances the instances.
	 */
	public void cache(Instances instances) {
		cache = Dataset.create(instances);
	}
	
	/**
	 * Evicts the cached data structures.
	 */
	public void evictCache() {
		cache = null;
	}
	
	/**
	 * Sets the parameters for this tree learner.
	 * 
	 * @param mode the parameters.
	 */
	public abstract void setParameters(String mode);

	protected static class Dataset {
		
		static Dataset create(Instances instances) {
			Dataset dataset = new Dataset(instances);
			List<Attribute> attributes = instances.getAttributes();
			// From attIndex to attName
			Map<Integer, String> attMap = new HashMap<>();
			for (int j = 0; j < attributes.size(); j++) {
				Attribute attribute = attributes.get(j);
				attMap.put(attribute.getIndex(), attribute.getName());
			}
			for (Attribute attribute : attributes) {
				dataset.sortedLists.put(attribute.getName(), new ArrayList<IntDoublePair>());
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
							String attName = attMap.get(indices[k]);
							dataset.sortedLists.get(attName).add(new IntDoublePair(i, values[k]));
						}
					}
				} else {
					double[] values = instance.getValues();
					for (int j = 0; j < values.length; j++) {
						if (attMap.containsKey(j) && values[j] != 0.0) {
							String attName = attMap.get(j);
							dataset.sortedLists.get(attName).add(new IntDoublePair(i, values[j]));
						}
					}
				}
			}
			for (List<IntDoublePair> sortedList : dataset.sortedLists.values()) {
				Collections.sort(sortedList, COMP);
			}
			return dataset;
		}
		
		static Dataset create(Dataset dataset, Instances instances) {
			Dataset copy = new Dataset();
			copy.instances = instances;
			copy.sortedLists = new HashMap<>(instances.dimension());
			List<Attribute> attributes = instances.getAttributes();
			for (Attribute attribute : attributes) {
				String attName = attribute.getName();
				List<IntDoublePair> sortedList = dataset.sortedLists.get(attName);
				if (sortedList == null) {
					// This should not happen very often
					sortedList = new ArrayList<>();
					for (int i = 0; i < instances.size(); i++) {
						Instance instance = instances.get(i);
						double v = instance.getValue(attribute);
						if (v != 0.0) {
							sortedList.add(new IntDoublePair(i, v));
						}
					}
					Collections.sort(sortedList, COMP);
					dataset.sortedLists.put(attName, sortedList);
				}
				List<IntDoublePair> copySortedList = new ArrayList<>(sortedList.size());
				for (IntDoublePair pair : sortedList) {
					copySortedList.add(new IntDoublePair(pair.v1, pair.v2));
				}
				copy.sortedLists.put(attName, copySortedList);
			}
			return copy;
		}

		public Instances instances;
		public Map<String, List<IntDoublePair>> sortedLists;
		
		Dataset() {
			
		}

		Dataset(Instances instances) {
			this.instances = new Instances(instances.getAttributes(), instances.getTargetAttribute());
			sortedLists = new HashMap<>(instances.dimension());
		}
		
		static Dataset merge(Dataset left, Dataset right) {
			Dataset data = new Dataset(left.instances);
			int lSize = left.instances.size();
			for (Instance instance : left.instances) {
				data.instances.add(instance);
			}
			for (Instance instance : right.instances) {
				data.instances.add(instance);
			}
			for (String attName : left.sortedLists.keySet()) {
				List<IntDoublePair> lSortedList = left.sortedLists.get(attName);
				List<IntDoublePair> rSortedList = right.sortedLists.get(attName);
				List<IntDoublePair> sortedList = new ArrayList<>(data.instances.size());
				int i = 0;
				int j = 0;
				while (i < lSortedList.size() && j < rSortedList.size()) {
					IntDoublePair l = lSortedList.get(i);
					IntDoublePair r = rSortedList.get(j);
					if (l.v2 < r.v2) {
						sortedList.add(l);
						i++;
					} else if (l.v2 > r.v2) {
						r.v1 += + lSize;
						sortedList.add(r);
						j++;
					} else {
						sortedList.add(l);
						r.v1 += lSize;
						sortedList.add(r);
						i++;
						j++;
					}
				}
				while (i < lSortedList.size()) {
					IntDoublePair l = lSortedList.get(i);
					sortedList.add(l);
					i++;
				}
				while (j < rSortedList.size()) {
					IntDoublePair r = rSortedList.get(j);
					r.v1 += lSize;
					sortedList.add(r);
					j++;
				}
				data.sortedLists.put(attName, sortedList);
			}
			
			return data;
		}
		
		void split(int attIndex, double split, Dataset left, Dataset right) {
			int[] leftHash = new int[instances.size()];
			int[] rightHash = new int[instances.size()];
			Arrays.fill(leftHash, -1);
			Arrays.fill(rightHash, -1);
			for (int i = 0; i < instances.size(); i++) {
				Instance instance = instances.get(i);
				if (instance.getValue(attIndex) <= split) {
					left.instances.add(instance);
					leftHash[i] = left.instances.size() - 1;
				} else {
					right.instances.add(instance);
					rightHash[i] = right.instances.size() - 1;
				}
			}

			for (String attName : sortedLists.keySet()) {
				left.sortedLists.put(attName, new ArrayList<IntDoublePair>(left.instances.size()));
				right.sortedLists.put(attName, new ArrayList<IntDoublePair>(right.instances.size()));

				List<IntDoublePair> sortedList = sortedLists.get(attName);
				for (IntDoublePair pair : sortedList) {
					int leftIdx = leftHash[pair.v1];
					int rightIdx = rightHash[pair.v1];
					if (leftIdx != -1) {
						pair.v1 = leftIdx;
						left.sortedLists.get(attName).add(pair);
					}
					if (rightIdx != -1) {
						pair.v1 = rightIdx;
						right.sortedLists.get(attName).add(pair);
					}
				}
			}
		}

	}
	
}
