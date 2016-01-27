package mltk.predictor.tree.ensemble;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import mltk.core.Copyable;
import mltk.predictor.tree.RTree;
import mltk.predictor.tree.RegressionTree;

/**
 * Class for regression tree list.
 * 
 * @author Yin Lou
 * 
 */
public class RTreeList implements Iterable<RTree>, Copyable<RTreeList> {

	protected List<RTree> trees;

	/**
	 * Constructor.
	 */
	public RTreeList() {
		trees = new ArrayList<>();
	}
	
	/**
	 * Constructor.
	 * 
	 * @param capacity the capacity of this tree list.
	 */
	public RTreeList(int capacity) {
		trees = new ArrayList<>(capacity);
	}

	/**
	 * Adds a regression tree to the list.
	 * 
	 * @param tree the regression tree to add.
	 */
	public void add(RTree tree) {
		trees.add(tree);
	}

	@Override
	public RTreeList copy() {
		RTreeList copy = new RTreeList();
		for (RTree rt : trees) {
			copy.trees.add(rt.copy());
		}
		return copy;
	}

	/**
	 * Returns the tree at the specified position in this list.
	 * 
	 * @param index the index of the element to return.
	 * @return the tree at the specified position in this list.
	 */
	public RTree get(int index) {
		return trees.get(index);
	}

	@Override
	public Iterator<RTree> iterator() {
		return trees.iterator();
	}

	/**
	 * Removes the last tree.
	 */
	public void removeLast() {
		if (trees.size() > 0) {
			trees.remove(trees.size() - 1);
		}
	}

	/**
	 * Replaces the tree at the specified position in this list with the new tree.
	 * 
	 * @param index the index of the element to replace.
	 * @param rt the regression tree to be stored at the specified position.
	 */
	public void set(int index, RegressionTree rt) {
		trees.set(index, rt);
	}

	/**
	 * Returns the size of this list.
	 * 
	 * @return the size of this list.
	 */
	public int size() {
		return trees.size();
	}

}
