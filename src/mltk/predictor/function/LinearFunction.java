package mltk.predictor.function;

import java.io.BufferedReader;
import java.io.PrintWriter;

import mltk.core.Instance;
import mltk.predictor.Regressor;

/**
 * Class for linear functions.
 * 
 * @author Yin Lou
 *
 */
public class LinearFunction implements Regressor, UnivariateFunction {

	/**
	 * The attribute index.
	 */
	protected int attIndex;

	/**
	 * The slope.
	 */
	protected double beta;

	/**
	 * Constructor.
	 */
	public LinearFunction() {

	}

	/**
	 * Constructs a linear function with a provided slope value.
	 * 
	 * @param beta the slope.
	 */
	public LinearFunction(double beta) {
		this(-1, beta);
	}

	/**
	 * Constructs a linear function with a provided slope value and attribute index.
	 * 
	 * @param attIndex the attribute index.
	 * @param beta the slope.
	 */
	public LinearFunction(int attIndex, double beta) {
		this.attIndex = attIndex;
		this.beta = beta;
	}

	@Override
	public void read(BufferedReader in) throws Exception {
		attIndex = Integer.parseInt(in.readLine().split(": ")[1]);
		beta = Double.parseDouble(in.readLine().split(": ")[1]);
	}

	@Override
	public void write(PrintWriter out) throws Exception {
		out.printf("[Predictor: %s]\n", this.getClass().getCanonicalName());
		out.println("AttIndex: " + attIndex);
		out.println("Beta: " + beta);
	}

	@Override
	public double evaluate(double x) {
		return beta * x;
	}

	@Override
	public double regress(Instance instance) {
		return evaluate(instance.getValue(attIndex));
	}

	public double getSlope() {
		return beta;
	}

	public void setSlope(double beta) {
		this.beta = beta;
	}

	public int getAttributeIndex() {
		return attIndex;
	}

	@Override
	public LinearFunction copy() {
		return new LinearFunction(attIndex, beta);
	}

}
