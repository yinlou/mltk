package mltk.predictor.tree.ensemble;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import mltk.core.Copyable;
import mltk.core.Instance;
import mltk.core.SparseVector;
import mltk.predictor.tree.DecisionTable;
import mltk.predictor.tree.RTree;

/**
 * Class for boosted decision tables.
 * 
 * @author Yin Lou
 *
 */
public class BoostedDTables implements Copyable<BoostedDTables> {
	
	static class IndexElement implements Comparable<IndexElement> {
		
		double cut;
		int tid;
		int pos;
		
		public IndexElement(double cut, int tid, int pos) {
			this.cut = cut;
			this.tid = tid;
			this.pos = pos;
		}

		@Override
		public int compareTo(IndexElement o) {
			return Double.compare(o.cut, this.cut);
		}
		
		public IndexElement copy() {
			return new IndexElement(cut, tid, pos);
		}
		
	}
	
	static class Index {
		
		IndexElement[] elements;
		
		public Index(IndexElement[] elements) {
			this.elements = elements;
		}
		
		void setPredIdx(long[] predIndices, double v) {
			for (IndexElement element : elements) {
				if (v <= element.cut) {
					long t = 1 << element.pos;
					int tid = element.tid;
					predIndices[tid] |= t;
				} else {
					break;
				}
			}
		}
		
		public Index copy() {
			IndexElement[] elementsCopy = new IndexElement[elements.length];
			for (int i = 0; i < elementsCopy.length; i++) {
				elementsCopy[i] = elements[i].copy();
			}
			return new Index(elementsCopy);
		}
		
	}
	
	protected static final IndexElement[] EMPTY_INDEX = new IndexElement[0];

	protected List<DecisionTable> dtList;
	protected Index[] indexes;
	
	/**
	 * Constructor.
	 */
	public BoostedDTables() {
		dtList = new ArrayList<>();
	}
	
	/**
	 * Constructor.
	 * 
	 * @param trees
	 */
	public BoostedDTables(BoostedRTrees trees) {
		dtList = new ArrayList<>(trees.size());
		for (RTree tree : trees) {
			dtList.add((DecisionTable) tree);
		}
		buildIndex();
	}
	
	/**
	 * Builds the index for fast scoring.
	 */
	public void buildIndex() {
		Map<Integer, List<IndexElement>> map = new HashMap<>();
		int p = -1;
		for (int i = 0; i < dtList.size(); i++) {
			DecisionTable dt = dtList.get(i);
			int[] attIndices = dt.getAttributeIndices();
			double[] cuts = dt.getSplits();
			for (int k = 0; k < attIndices.length; k++) {
				IndexElement element = new IndexElement(cuts[k], i, attIndices.length - k - 1);
				if (attIndices[k] > p) {
					p = attIndices[k];
				}
				int attIdx = attIndices[k];
				if (!map.containsKey(attIdx)) {
					map.put(attIdx, new ArrayList<IndexElement>());
				}
				map.get(attIdx).add(element);
			}
		}
		p++;
		indexes = new Index[p];
		for (int j = 0; j < p; j++) {
			List<IndexElement> elements = map.get(j);
			if (elements != null) {
				Collections.sort(elements);
				indexes[j] = new Index(elements.toArray(new IndexElement[elements.size()]));
			} else {
				indexes[j] = new Index(EMPTY_INDEX);
			}
		}
	}
	
	/**
	 * Adds a decision table to the list.
	 * 
	 * @param dt the decision table to add.
	 */
	public void add(DecisionTable dt) {
		dtList.add(dt);
	}
	
	/**
	 * Returns the table at the specified position in this list.
	 * 
	 * @param index the index of the element to return.
	 * @return the table at the specified position in this list.
	 */
	public DecisionTable get(int index) {
		return dtList.get(index);
	}
	
	/**
	 * Removes the last tree.
	 */
	public void removeLast() {
		if (dtList.size() > 0) {
			dtList.remove(dtList.size() - 1);
		}
	}
	
	/**
	 * Returns the size of this boosted decision table list.
	 * 
	 * @return the size of this boosted decision table list.
	 */
	public int size() {
		return dtList.size();
	}
	
	/**
	 * Replaces the table at the specified position in this list with the new table.
	 * 
	 * @param index the index of the element to replace.
	 * @param dt the decision table to be stored at the specified position.
	 */
	public void set(int index, DecisionTable dt) {
		dtList.set(index, dt);
	}
	
	/**
	 * Regresses an instance.
	 * 
	 * @param instance the instance to regress.
	 * @return a regressed value.
	 */
	public double regress(Instance instance) {
		double[] values = instance.getValues();
		long[] predIndices = new long[dtList.size()];
		if (instance.isSparse()) {
			int[] indices = ((SparseVector) instance.getVector()).getIndices();
			for (int j = 0; j < Math.min(values.length, indexes.length); j++) {
				double v = values[j];
				int idx = indices[j];
				indexes[idx].setPredIdx(predIndices, v);
			}
		} else {
			for (int j = 0; j < Math.min(values.length, indexes.length); j++) {
				double v = values[j];
				indexes[j].setPredIdx(predIndices, v);
			}
		}
		
		double pred = 0;
		for (int i = 0 ; i < predIndices.length; i++) {
			DecisionTable dt = dtList.get(i);
			pred += dt.regress(predIndices[i]);
		}
		return pred;
	}
	
	@Override
	public BoostedDTables copy() {
		return copy(true);
	}
	
	/**
	 * Copies this object.
	 * 
	 * @param copyIndexes <code>true</code> if the indexes are also copied;
	 * @return this object.
	 */
	public BoostedDTables copy(boolean copyIndexes) {
		BoostedDTables copy = new BoostedDTables();
		copy.dtList = new ArrayList<>();
		for (DecisionTable dt : dtList) {
			copy.dtList.add(dt.copy());
		}
		if (copyIndexes) {
			copy.indexes = new Index[this.indexes.length];
			for (int i = 0; i < copy.indexes.length; i++) {
				copy.indexes[i] = indexes[i].copy();
			}
		}
		return copy;
	}

	/**
	 * Reads in this boosted decision tables.
	 * 
	 * @param in the reader.
	 * @throws Exception
	 */
	public void read(BufferedReader in) throws Exception {
		int n = Integer.parseInt(in.readLine().split(": ")[1]);
		for (int j = 0; j < n; j++) {
			String line = in.readLine();
			String predictorName = line.substring(1, line.length() - 1).split(": ")[1];
			Class<?> clazz = Class.forName(predictorName);
			DecisionTable dt = (DecisionTable) clazz.newInstance();
			dt.read(in);
			this.dtList.add(dt);

			in.readLine();
		}
		buildIndex();
	}

	/**
	 * Writes this boosted decision tables.
	 * 
	 * @param out the writer.
	 * @throws Exception
	 */
	public void write(PrintWriter out) throws Exception {
		out.printf("[Predictor: %s]\n", this.getClass().getCanonicalName());
		out.println("Length: " + dtList.size());
		for (DecisionTable dt : dtList) {
			dt.write(out);
			out.println();
		}
	}
	
}
