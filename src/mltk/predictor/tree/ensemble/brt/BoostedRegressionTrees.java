package mltk.predictor.tree.ensemble.brt;

import java.util.ArrayList;

import mltk.core.Instance;
import mltk.predictor.tree.RegressionTree;
import mltk.predictor.tree.RegressionTreeList;

/**
 * Class for boosted regression trees. This is a base class for BRT.
 * 
 * @author Yin Lou
 * 
 */
public class BoostedRegressionTrees extends RegressionTreeList {

	/**
	 * Constructor.
	 */
	public BoostedRegressionTrees() {
		super();
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

}
