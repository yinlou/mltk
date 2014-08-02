package mltk.predictor.tree.ensemble.brt;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import mltk.core.Instance;
import mltk.predictor.tree.RegressionTree;

/**
 * Class for boosted regression trees. This is a base class for BRT.
 * 
 * @author Yin Lou
 * 
 */
class BoostedRegressionTrees implements Iterable<RegressionTree> {

	protected List<RegressionTree> trees;

	/**
	 * Constructor.
	 */
	public BoostedRegressionTrees() {
		trees = new ArrayList<>();
	}

	/**
	 * Constructs a regression tree list of length n. By default each tree is null.
	 * 
	 * @param n the length.
	 */
	public BoostedRegressionTrees(int n) {
		trees = new ArrayList<>(n);
		for (int i = 0; i < n; i++) {
			trees.add(null);
		}
	}

	/**
	 * Adds a regression tree to the list.
	 * 
	 * @param tree the regression tree to add.
	 */
	public void add(RegressionTree tree) {
		trees.add(tree);
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
	 * Regresses an instance.
	 * 
	 * @param instance the instance.
	 * @return the regressed
	 */
	public double regress(Instance instance) {
		double pred = 0;
		for (RegressionTree rt : trees) {
			pred += rt.regress(instance);
		}
		return pred;
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
