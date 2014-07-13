package mltk.predictor.tree;

import java.io.BufferedReader;
import java.io.PrintWriter;

/**
 * Class for regression tree leaves.
 * 
 * @author Yin Lou
 * 
 */
public class RegressionTreeLeaf extends RegressionTreeNode {

	protected double prediction;

	/**
	 * Constructor.
	 */
	public RegressionTreeLeaf() {

	}

	/**
	 * Constructs a leaf node with a constant prediction.
	 * 
	 * @param prediction the prediction for this leaf node.
	 */
	public RegressionTreeLeaf(double prediction) {
		this.prediction = prediction;
	}

	@Override
	public boolean isLeaf() {
		return true;
	}

	/**
	 * Sets the prediction for this leaf node.
	 * 
	 * @param prediction the prediction for this leaf node.
	 */
	public void setPrediction(double prediction) {
		this.prediction = prediction;
	}

	/**
	 * Returns the prediction for this leaf node.
	 * 
	 * @return the prediction for this leaf node.
	 */
	public double getPrediction() {
		return prediction;
	}

	@Override
	public void read(BufferedReader in) throws Exception {
		prediction = Double.parseDouble(in.readLine().split(": ")[1]);
	}

	@Override
	public void write(PrintWriter out) throws Exception {
		out.println(this.getClass().getCanonicalName());
		out.println("Prediction: " + prediction);
	}

	@Override
	public RegressionTreeLeaf copy() {
		return new RegressionTreeLeaf(prediction);
	}

}
