package mltk.predictor.tree.ensemble;

import java.util.ArrayList;

import mltk.core.Instance;
import mltk.predictor.tree.RTree;

/**
 * Class for bagged regression trees.
 * 
 * @author Yin Lou
 * 
 */
public class BaggedRTrees extends RTreeList {

	/**
	 * Constructor.
	 */
	public BaggedRTrees() {
		super();
	}

	/**
	 * Constructs a regression tree list of length n. By default each tree is null.
	 * 
	 * @param n the length.
	 */
	public BaggedRTrees(int n) {
		trees = new ArrayList<>(n);
		for (int i = 0; i < n; i++) {
			trees.add(null);
		}
	}

	/**
	 * Regresses an instance.
	 * 
	 * @param instance the instance.
	 * @return the regressed
	 */
	public double regress(Instance instance) {
		double pred = 0;
		for (RTree rt : trees) {
			pred += rt.regress(instance);
		}
		return pred / trees.size();
	}

}
