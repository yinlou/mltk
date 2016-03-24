package mltk.predictor;

import mltk.core.Instance;

/**
 * Class for boosted ensembles.
 * 
 * @author Yin Lou
 * 
 */
public class BoostedEnsemble extends Ensemble {

	/**
	 * Constructor.
	 */
	public BoostedEnsemble() {
		super();
	}

	/**
	 * Constructor.
	 * 
	 * @param capacity the capacity of the boosted ensemble.
	 */
	public BoostedEnsemble(int capacity) {
		super(capacity);
	}

	@Override
	public double regress(Instance instance) {
		double prediction = 0.0;
		for (Predictor predictor : predictors) {
			Regressor regressor = (Regressor) predictor;
			prediction += regressor.regress(instance);
		}
		return prediction;
	}

	@Override
	public int classify(Instance instance) {
		double pred = regress(instance);
		if (pred >= 0) {
			return 1;
		} else {
			return -1;
		}
	}

	/**
	 * Removes a particular predictor.
	 * 
	 * @param index the index of the predictor to remove.
	 */
	public void remove(int index) {
		predictors.remove(index);
	}

	/**
	 * Removes the last predictor.
	 */
	public void removeLast() {
		if (predictors.size() > 0) {
			predictors.remove(predictors.size() - 1);
		}
	}

	@Override
	public BoostedEnsemble copy() {
		BoostedEnsemble copy = new BoostedEnsemble(predictors.size());
		for (Predictor predictor : predictors) {
			copy.add(predictor.copy());
		}
		return copy;
	}

}
