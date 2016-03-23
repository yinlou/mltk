package mltk.predictor.tree;

import mltk.predictor.Regressor;

/**
 * Interface for regression trees.
 * 
 * @author Yin Lou
 *
 */
public interface RTree extends Regressor {

	/**
	 * Multiplies this tree with a constant.
	 * 
	 * @param c the constant.
	 */
	public void multiply(double c);
	
	/**
	 * Returns a deep copy of this tree.
	 */
	public RTree copy();
	
}
