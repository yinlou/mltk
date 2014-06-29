package mltk.predictor.tree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import mltk.core.Attribute;
import mltk.core.Attribute.Type;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.util.tuple.IntDoublePair;
import mltk.util.tuple.IntDoublePairComparator;

class Dataset {

	Instances instances;

	List<List<IntDoublePair>> sortedLists;

	Dataset(Instances instances) {
		this.instances = new Instances(instances.getAttributes(),
				instances.getTargetAttribute());
		sortedLists = new ArrayList<>(instances.dimension());
	}

	static Dataset create(Instances instances) {
		Dataset dataset = new Dataset(instances);
		List<Attribute> attributes = instances.getAttributes();
		for (int i = 0; i < attributes.size(); i++) {
			Attribute attribute = attributes.get(i);
			int capacity = attribute.getType() == Type.NUMERIC ? instances
					.size() : 0;
			dataset.sortedLists.add(new ArrayList<IntDoublePair>(capacity));
		}
		for (int i = 0; i < instances.size(); i++) {
			Instance instance = instances.get(i);
			dataset.instances.add(instance.clone());
			for (int j = 0; j < attributes.size(); j++) {
				Attribute attribute = attributes.get(j);
				if (attribute.getType() == Type.NUMERIC) {
					dataset.sortedLists.get(j).add(
							new IntDoublePair(i, instance.getValue(attribute)));
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

		List<Attribute> attributes = instances.getAttributes();
		for (int i = 0; i < sortedLists.size(); i++) {
			Attribute attribute = attributes.get(i);
			if (attribute.getType() == Type.NUMERIC) {
				left.sortedLists.add(new ArrayList<IntDoublePair>(
						left.instances.size()));
				right.sortedLists.add(new ArrayList<IntDoublePair>(
						right.instances.size()));
			} else {
				left.sortedLists.add(new ArrayList<IntDoublePair>(0));
				right.sortedLists.add(new ArrayList<IntDoublePair>(0));
			}

			List<IntDoublePair> sortedList = sortedLists.get(i);
			for (IntDoublePair pair : sortedList) {
				int leftIdx = leftHash[pair.v1];
				int rightIdx = rightHash[pair.v1];
				if (leftIdx != -1) {
					left.sortedLists.get(i).add(
							new IntDoublePair(leftIdx, pair.v2));
				}
				if (rightIdx != -1) {
					right.sortedLists.get(i).add(
							new IntDoublePair(rightIdx, pair.v2));
				}
			}
		}
	}

}
