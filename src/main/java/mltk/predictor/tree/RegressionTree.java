package mltk.predictor.tree;

import java.io.BufferedReader;
import java.io.PrintWriter;

import mltk.core.Instance;

/**
 * Class for regression trees.
 * 
 * @author Yin Lou
 * 
 */
public class RegressionTree implements RTree {

	/**
	 * The root of a tree.
	 */
	protected TreeNode root;

	/**
	 * Constructs an empty tree.
	 */
	public RegressionTree() {
		root = null;
	}

	/**
	 * Constructs a regression tree with specified root.
	 * 
	 * @param root the root.
	 */
	public RegressionTree(TreeNode root) {
		this.root = root;
	}

	/**
	 * Returns the root of this regression tree.
	 * 
	 * @return the root of this regression tree.
	 */
	public TreeNode getRoot() {
		return root;
	}

	/**
	 * Sets the root for this regression tree.
	 * 
	 * @param root the new root.
	 */
	public void setRoot(TreeNode root) {
		this.root = root;
	}

	/**
	 * Returns the leaf node.
	 * 
	 * @param instance the data point.
	 * @return the leaf node.
	 */
	public RegressionTreeLeaf getLeafNode(Instance instance) {
		TreeNode node = root;
		while (!node.isLeaf()) {
			TreeInteriorNode interiorNode = (TreeInteriorNode) node;
			if (interiorNode.goLeft(instance)) {
				node = interiorNode.getLeftChild();
			} else {
				node = interiorNode.getRightChild();
			}
		}
		return (RegressionTreeLeaf) node;
	}

	/**
	 * Multiplies this regression tree with a constant.
	 * 
	 * @param c the constant.
	 */
	public void multiply(double c) {
		multiply(root, c);
	}

	/**
	 * Multiplies this subtree with a constant.
	 * 
	 * @param node the root of the subtree.
	 * @param c the constant.
	 */
	protected void multiply(TreeNode node, double c) {
		if (node.isLeaf()) {
			RegressionTreeLeaf leaf = (RegressionTreeLeaf) node;
			leaf.prediction *= c;
		} else {
			TreeInteriorNode interiorNode = (TreeInteriorNode) node;
			multiply(interiorNode.left, c);
			multiply(interiorNode.right, c);
		}
	}

	@Override
	public double regress(Instance instance) {
		return getLeafNode(instance).getPrediction();
	}

	@Override
	public void read(BufferedReader in) throws Exception {
		in.readLine();
		Class<?> clazz = Class.forName(in.readLine());
		root = (TreeNode) clazz.newInstance();
		root.read(in);
	}

	@Override
	public void write(PrintWriter out) throws Exception {
		out.printf("[Predictor: %s]\n", this.getClass().getCanonicalName());
		out.println();
		root.write(out);
	}

	@Override
	public RegressionTree copy() {
		return new RegressionTree(root.copy());
	}

}
