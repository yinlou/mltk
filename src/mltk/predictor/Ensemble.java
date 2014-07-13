package mltk.predictor;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Abstract class for ensembles.
 * 
 * @author Yin Lou
 * 
 */
public abstract class Ensemble implements Classifier, Regressor {

	protected List<Predictor> predictors;

	/**
	 * Constructor.
	 */
	public Ensemble() {
		predictors = new ArrayList<>();
	}

	/**
	 * Constructor.
	 * 
	 * @param capacity the capacity of this ensemble.
	 */
	public Ensemble(int capacity) {
		predictors = new ArrayList<>(capacity);
	}

	/**
	 * Returns a particular predictor.
	 * 
	 * @param index the index of predictor.
	 * @return a particular predictor.
	 */
	public Predictor get(int index) {
		return predictors.get(index);
	}

	/**
	 * Returns the internal predictors.
	 * 
	 * @return the internal predictors.
	 */
	public List<Predictor> getPredictors() {
		return predictors;
	}

	/**
	 * Adds a new predictor to the ensemble.
	 * 
	 * @param predictor the new predictor.
	 */
	public void add(Predictor predictor) {
		predictors.add(predictor);
	}

	/**
	 * Returns the size of this ensemble.
	 * 
	 * @return the size of this ensemble.
	 */
	public int size() {
		return predictors.size();
	}

	/**
	 * Clears this ensemble.
	 */
	public void clear() {
		predictors.clear();
	}

	@Override
	public void read(BufferedReader in) throws Exception {
		int capacity = Integer.parseInt(in.readLine().split(": ")[1]);
		predictors = new ArrayList<>(capacity);
		in.readLine();
		for (int i = 0; i < capacity; i++) {
			String line = in.readLine();
			String predictorName = line.substring(1, line.length() - 1).split(": ")[1];
			Class<?> clazz = Class.forName(predictorName);
			Predictor predictor = (Predictor) clazz.newInstance();
			predictor.read(in);
			predictors.add(predictor);
			in.readLine();
		}
	}

	@Override
	public void write(PrintWriter out) throws Exception {
		out.printf("[Predictor: %s]\n", this.getClass().getCanonicalName());
		out.println("Ensemble: " + predictors.size());
		out.println();
		for (Predictor predictor : predictors) {
			predictor.write(out);
			out.println();
		}
	}

}
