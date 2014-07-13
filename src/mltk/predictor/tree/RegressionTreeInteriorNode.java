package mltk.predictor.tree;

import java.io.BufferedReader;
import java.io.PrintWriter;

import mltk.core.Instance;

/**
 * Class for regression tree interior nodes.
 * 
 * @author Yin Lou
 * 
 */
public class RegressionTreeInteriorNode extends RegressionTreeNode {

	protected RegressionTreeNode left;
	protected RegressionTreeNode right;
	protected int attIndex;
	protected double splitPoint;

	/**
	 * Constructor.
	 */
	public RegressionTreeInteriorNode() {

	}

	/**
	 * Constructs an interior node with attribute index and split point.
	 * 
	 * @param attIndex the attribute index.
	 * @param splitPoint the split point.
	 */
	public RegressionTreeInteriorNode(int attIndex, double splitPoint) {
		this.attIndex = attIndex;
		this.splitPoint = splitPoint;
	}

	/**
	 * Returns the left child.
	 * 
	 * @return the left child.
	 */
	public RegressionTreeNode getLeftChild() {
		return left;
	}

	/**
	 * Returns the right child.
	 * 
	 * @return the right child.
	 */
	public RegressionTreeNode getRightChild() {
		return right;
	}

	/**
	 * Returns the split attribute index.
	 * 
	 * @return the split attribute index.
	 */
	public int getSplitAttributeIndex() {
		return attIndex;
	}

	/**
	 * Returns the split point.
	 * 
	 * @return the split point.
	 */
	public double getSplitPoint() {
		return splitPoint;
	}

	@Override
	public boolean isLeaf() {
		return false;
	}

	/**
	 * Returns <code>true</code> if going to left child.
	 * 
	 * @param instance the instance.
	 * @return <code>true</code> if going to left child.
	 */
	public boolean goLeft(Instance instance) {
		double value = instance.getValue(attIndex);
		return value <= splitPoint;
	}

	@Override
	public void read(BufferedReader in) throws Exception {
		attIndex = Integer.parseInt(in.readLine().split(": ")[1]);
		splitPoint = Double.parseDouble(in.readLine().split(": ")[1]);
		in.readLine();

		Class<?> clazzLeft = Class.forName(in.readLine());
		left = (RegressionTreeNode) clazzLeft.newInstance();
		left.read(in);

		in.readLine();

		Class<?> clazzRight = Class.forName(in.readLine());
		right = (RegressionTreeNode) clazzRight.newInstance();
		right.read(in);
	}

	@Override
	public void write(PrintWriter out) throws Exception {
		out.println(this.getClass().getCanonicalName());
		out.println("AttIndex: " + attIndex);
		out.println("SplintPoint: " + splitPoint);
		out.println();
		left.write(out);
		out.println();
		right.write(out);
	}

	@Override
	public RegressionTreeNode copy() {
		RegressionTreeInteriorNode copy = new RegressionTreeInteriorNode(attIndex, splitPoint);
		copy.left = this.left.copy();
		copy.right = this.right.copy();
		return copy;
	}

}
