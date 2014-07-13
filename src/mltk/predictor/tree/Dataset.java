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
import mltk.util.tuple.IntDoublePair;
import mltk.util.tuple.IntDoublePairComparator;

class Dataset {

	Instances instances;

	List<List<IntDoublePair>> sortedLists;

	Dataset(Instances instances) {
		this.instances = new Instances(instances.getAttributes(), instances.getTargetAttribute());
		sortedLists = new ArrayList<>(instances.dimension());
	}

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
