package mltk.predictor.tree;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import mltk.core.Copyable;

/**
 * Class for regression tree list.
 * 
 * @author Yin Lou
 * 
 */
public class RegressionTreeList implements Iterable<RegressionTree>, Copyable<RegressionTreeList> {

	protected List<RegressionTree> trees;

	/**
	 * Constructor.
	 */
	public RegressionTreeList() {
		trees = new ArrayList<>();
	}
	
	/**
	 * Constructor.
	 * 
	 * @param capacity the capacity of this tree list.
	 */
	public RegressionTreeList(int capacity) {
		trees = new ArrayList<>(capacity);
	}

	/**
	 * Adds a regression tree to the list.
	 * 
	 * @param tree the regression tree to add.
	 */
	public void add(RegressionTree tree) {
		trees.add(tree);
	}

	@Override
	public RegressionTreeList copy() {
		RegressionTreeList copy = new RegressionTreeList();
		for (RegressionTree rt : trees) {
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
	public RegressionTree get(int index) {
		return trees.get(index);
	}

	@Override
	public Iterator<RegressionTree> iterator() {
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
