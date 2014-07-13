package mltk.predictor.glm;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.Arrays;

import mltk.core.Instance;
import mltk.core.SparseVector;
import mltk.predictor.ProbabilisticClassifier;
import mltk.predictor.Regressor;
import mltk.util.ArrayUtils;
import mltk.util.StatUtils;

/**
 * Class for generalized linear models (GLMs).
 * 
 * @author Yin Lou
 * 
 */
public class GLM implements ProbabilisticClassifier, Regressor {

	/**
	 * The coefficient vectors.
	 */
	protected double[][] w;

	/**
	 * The intercept vector.
	 */
	protected double[] intercept;

	/**
	 * Constructor.
	 */
	public GLM() {

	}

	/**
	 * Constructs a GLM with the specified dimension.
	 * 
	 * @param dimension the dimension.
	 */
	public GLM(int dimension) {
		this(1, dimension);
	}

	/**
	 * Constructs a GLM with the specified dimension.
	 * 
	 * @param numClasses the number of classes.
	 * @param dimension the dimension.
	 */
	public GLM(int numClasses, int dimension) {
		w = new double[numClasses][dimension];
		intercept = new double[numClasses];
	}

	/**
	 * Constructs a GLM with the intercept vector and the coefficient vectors.
	 * 
	 * @param intercept the intercept vector.
	 * @param w the coefficient vectors.
	 */
	public GLM(double[] intercept, double[][] w) {
		if (intercept.length != w.length) {
			throw new IllegalArgumentException("Dimensions of intercept and w must match.");
		}
		this.intercept = intercept;
		this.w = w;
	}

	/**
	 * Returns the coefficient vectors.
	 * 
	 * @return the coefficient vectors.
	 */
	public double[][] coefficients() {
		return w;
	}

	/**
	 * Returns the coefficient vectors for class k.
	 * 
	 * @param k the index of the class.
	 * @return the coefficient vectors for class k.
	 */
	public double[] coefficients(int k) {
		return w[k];
	}

	/**
	 * Returns the intercept vector.
	 * 
	 * @return the intercept vector.
	 */
	public double[] intercept() {
		return intercept;
	}

	/**
	 * Returns the intercept for class k.
	 * 
	 * @param k the index of the class.
	 * @return the intercept for class k.
	 */
	public double intercept(int k) {
		return intercept[k];
	}

	@Override
	public void read(BufferedReader in) throws Exception {
		in.readLine();
		intercept = ArrayUtils.parseDoubleArray(in.readLine());
		int p = Integer.parseInt(in.readLine().split(": ")[1]);
		w = new double[intercept.length][p];
		for (int j = 0; j < p; j++) {
			String[] data = in.readLine().split("\\s+");
			for (int i = 0; i < w.length; i++) {
				w[i][j] = Double.parseDouble(data[i]);
			}
		}
	}

	@Override
	public void write(PrintWriter out) throws Exception {
		out.printf("[Predictor: %s]\n", this.getClass().getCanonicalName());
		out.println("Intercept: " + intercept.length);
		out.println(Arrays.toString(intercept));
		final int p = w[0].length;
		out.println("Coefficients: " + p);
		for (int j = 0; j < p; j++) {
			out.print(w[0][j]);
			for (int i = 1; i < w.length; i++) {
				out.print(" " + w[i][j]);
			}
			out.println();
		}
	}

	@Override
	public double regress(Instance instance) {
		return regress(intercept[0], w[0], instance);
	}

	@Override
	public int classify(Instance instance) {
		double[] prob = predictProbabilities(instance);
		return StatUtils.indexOfMax(prob);
	}

	@Override
	public double[] predictProbabilities(Instance instance) {
		if (w.length == 1) {
			double[] prob = new double[2];
			double pred = regress(intercept[0], w[0], instance);
			prob[0] = 1 / (1 + Math.exp(-pred));
			prob[1] = 1 - prob[0];
			return prob;
		} else {
			double[] prob = new double[w.length];
			double[] pred = new double[w.length];
			double sum = 0;
			for (int i = 0; i < w.length; i++) {
				pred[i] = regress(intercept[i], w[i], instance);
				prob[i] = 1 / (1 + Math.exp(-pred[i]));
				sum += prob[i];
			}
			for (int i = 0; i < prob.length; i++) {
				prob[i] /= sum;
			}
			return prob;
		}
	}

	@Override
	public GLM copy() {
		double[][] copyW = new double[w.length][];
		for (int i = 0; i < copyW.length; i++) {
			copyW[i] = Arrays.copyOf(w[i], w[i].length);
		}
		return new GLM(intercept, copyW);
	}

	protected double regress(double intercept, double[] coef, Instance instance) {
		if (!instance.isSparse()) {
			double pred = intercept;
			for (int i = 0; i < coef.length; i++) {
				pred += coef[i] * instance.getValue(i);
			}
			return pred;
		} else {
			double pred = intercept;
			SparseVector vector = (SparseVector) instance.getVector();
			int[] indices = vector.getIndices();
			double[] values = vector.getValues();
			for (int i = 0; i < indices.length; i++) {
				int index = indices[i];
				if (index < coef.length) {
					pred += coef[index] * values[i];
				}
			}
			return pred;
		}
	}

}
