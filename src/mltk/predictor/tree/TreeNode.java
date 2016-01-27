package mltk.predictor.tree;

import mltk.core.Copyable;
import mltk.core.Writable;

/**
 * Abstract class for tree nodes.
 * 
 * @author Yin Lou
 * 
 */
public abstract class TreeNode implements Writable, Copyable<TreeNode> {

	/**
	 * Returns <code>true</code> if the node is a leaf.
	 * 
	 * @return <code>true</code> if the node is a leaf.
	 */
	public abstract boolean isLeaf();

}
