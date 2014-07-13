package mltk.predictor.function;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.Arrays;

import mltk.core.Instance;
import mltk.predictor.Regressor;
import mltk.util.ArrayUtils;

/**
 * Class for cubic splines. Given knots k, the cubic spline uses the following basis: 1, x, x^2, x^3, (x - k[i])_+^3.
 * 
 * @author Yin Lou
 * 
 */
public class CubicSpline implements Regressor, UnivariateFunction {

	protected int attIndex;
	protected double intercept;
	protected double[] knots;
	protected double[] w;

	/**
	 * Constructor.
	 * 
	 * @param attIndex the attribute index.
	 * @param intercept the intercept.
	 * @param knots the knots.
	 * @param w the coefficient vector.
	 */
	public CubicSpline(int attIndex, double intercept, double[] knots, double[] w) {
		this.attIndex = attIndex;
		this.intercept = intercept;
		this.knots = knots;
		this.w = w;
	}

	/**
	 * Constructor.
	 * 
	 * @param intercept the intercept.
	 * @param knots the knots.
	 * @param w the coefficient vector.
	 */
	public CubicSpline(double intercept, double[] knots, double[] w) {
		this(-1, intercept, knots, w);
	}

	/**
	 * Constructor.
	 * 
	 * @param knots the knots.
	 * @param w the coefficient vector.
	 */
	public CubicSpline(double[] knots, double[] w) {
		this(-1, 0, knots, w);
	}

	/**
	 * Construct a cubic spline with specified knots and all coefficients set to zero.
	 * 
	 * @param knots the knots.
	 */
	public CubicSpline(double[] knots) {
		this(-1, 0, knots, new double[knots.length + 3]);
	}

	/**
	 * Constructor.
	 */
	public CubicSpline() {

	}

	@Override
	public void read(BufferedReader in) throws Exception {
		attIndex = Integer.parseInt(in.readLine().split(": ")[1]);
		intercept = Double.parseDouble(in.readLine().split(": ")[1]);
		in.readLine();
		knots = ArrayUtils.parseDoubleArray(in.readLine());
		in.readLine();
		w = ArrayUtils.parseDoubleArray(in.readLine());
	}

	@Override
	public void write(PrintWriter out) throws Exception {
		out.printf("[Predictor: %s]\n", this.getClass().getCanonicalName());
		out.println("AttIndex: " + attIndex);
		out.println("Intercept: " + intercept);
		out.println("Knots: " + knots.length);
		out.println(Arrays.toString(knots));
		out.println("Coefficients: " + w.length);
		out.println(Arrays.toString(w));
	}

	@Override
	public double evaluate(double x) {
		double pred = intercept + w[0] * x + w[1] * x * x + w[2] * x * x * x;
		for (int i = 0; i < knots.length; i++) {
			pred += h(x, knots[i]) * w[i + 3];
		}
		return pred;
	}

	/**
	 * Calculate the basis.
	 * 
	 * @param x a real.
	 * @param k a knot.
	 * @return h(x, z), a basis in cubic spline.
	 */
	public static double h(double x, double k) {
		double t = x - k;
		if (t < 0) {
			return 0;
		}
		return t * t * t;
	}

	@Override
	public double regress(Instance instance) {
		return evaluate(instance.getValue(attIndex));
	}

	/**
	 * Returns the attribute index.
	 * 
	 * @return the attribute index.
	 */
	public int getAttributeIndex() {
		return attIndex;
	}

	/**
	 * Sets the attribute index.
	 * 
	 * @param attIndex the new attribute index.
	 */
	public void setAttributeIndex(int attIndex) {
		this.attIndex = attIndex;
	}

	/**
	 * Returns the coefficient vector.
	 * 
	 * @return the coefficient vector.
	 */
	public double[] getCoefficients() {
		return w;
	}

	/**
	 * Returns the knot vector.
	 * 
	 * @return the knot vector.
	 */
	public double[] getKnots() {
		return knots;
	}

	/**
	 * Returns the intercept.
	 * 
	 * @return the intercept.
	 */
	public double getIntercept() {
		return intercept;
	}

	@Override
	public CubicSpline copy() {
		double[] newW = w.clone();
		double[] newKnots = knots.clone();
		return new CubicSpline(attIndex, intercept, newW, newKnots);
	}

}
