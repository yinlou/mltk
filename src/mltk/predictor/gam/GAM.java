package mltk.predictor.gam;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import mltk.core.Instance;
import mltk.predictor.ProbabilisticClassifier;
import mltk.predictor.Regressor;
import mltk.util.ArrayUtils;

/**
 * Class for generalized additive models (GAMs).
 * 
 * @author Yin Lou
 * 
 */
public class GAM implements ProbabilisticClassifier, Regressor {

	class RegressorList implements Iterable<Regressor> {

		List<Regressor> regressors;

		RegressorList() {
			regressors = new ArrayList<>();
		}

		@Override
		public Iterator<Regressor> iterator() {
			return regressors.iterator();
		}
	}

	class TermList implements Iterable<int[]> {

		List<int[]> terms;

		TermList() {
			terms = new ArrayList<>();
		}

		@Override
		public Iterator<int[]> iterator() {
			return terms.iterator();
		}
	}

	protected double intercept;
	protected List<Regressor> regressors;
	protected List<int[]> terms;

	/**
	 * Constructor.
	 */
	public GAM() {
		regressors = new ArrayList<>();
		terms = new ArrayList<>();
		intercept = 0;
	}

	/**
	 * Returns the intercept.
	 * 
	 * @return the intercept.
	 */
	public double getIntercept() {
		return intercept;
	}

	/**
	 * Sets the intercept.
	 * 
	 * @param intercept the new intercept.
	 */
	public void setIntercept(double intercept) {
		this.intercept = intercept;
	}

	@Override
	public void read(BufferedReader in) throws Exception {
		intercept = Double.parseDouble(in.readLine().split(": ")[1]);
		int size = Integer.parseInt(in.readLine().split(": ")[1]);
		regressors = new ArrayList<>(size);
		terms = new ArrayList<>(size);
		in.readLine();
		for (int i = 0; i < size; i++) {
			int[] term = ArrayUtils.parseIntArray(in.readLine().split(": ")[1]);
			terms.add(term);
			in.readLine();

			String line = in.readLine();
			String regressorName = line.substring(1, line.length() - 1).split(": ")[1];
			Class<?> clazz = Class.forName(regressorName);
			Regressor regressor = (Regressor) clazz.newInstance();
			regressor.read(in);
			regressors.add(regressor);
			in.readLine();
		}
	}

	@Override
	public void write(PrintWriter out) throws Exception {
		out.printf("[Predictor: %s]\n", this.getClass().getCanonicalName());
		out.println("Intercept: " + intercept);
		out.println("Components: " + regressors.size());
		out.println();
		for (int i = 0; i < regressors.size(); i++) {
			out.println("Component: " + Arrays.toString(terms.get(i)));
			out.println();
			regressors.get(i).write(out);
			out.println();
		}
	}

	/**
	 * Adds a new term into this GAM. The term is an array of attribute indices that are used in the regressor.
	 * 
	 * @param term the new term to add.
	 * @param regressor the new regressor to add.
	 */
	public void add(int[] term, Regressor regressor) {
		terms.add(term);
		regressors.add(regressor);
	}

	@Override
	public double regress(Instance instance) {
		double pred = intercept;
		for (Regressor regressor : regressors) {
			pred += regressor.regress(instance);
		}
		return pred;
	}

	@Override
	public int classify(Instance instance) {
		double pred = regress(instance);
		return pred >= 0 ? 1 : 0;
	}

	@Override
	public double[] predictProbabilities(Instance instance) {
		double pred = regress(instance);
		double prob = 1 / (1 + Math.exp(-pred));
		return new double[] { 1 - prob, prob };
	}

	/**
	 * Returns the term list.
	 * 
	 * @return the term list.
	 */
	public List<int[]> getTerms() {
		return terms;
	}

	/**
	 * Returns the regressor list.
	 * 
	 * @return the regressor list.
	 */
	public List<Regressor> getRegressors() {
		return regressors;
	}

	@Override
	public GAM copy() {
		GAM copy = new GAM();
		copy.intercept = intercept;
		for (Regressor regressor : regressors) {
			copy.regressors.add((Regressor) regressor.copy());
		}
		for (int[] term : terms) {
			copy.terms.add(term.clone());
		}

		return copy;
	}

}
