package mltk.predictor;

import java.util.HashMap;
import java.util.Map;

import mltk.core.Instance;

/**
 * Class for bagged ensembles.
 * 
 * @author Yin Lou
 * 
 */
public class BaggedEnsemble extends Ensemble {

	/**
	 * Constructor.
	 */
	public BaggedEnsemble() {
		super();
	}

	/**
	 * Constructor.
	 * 
	 * @param capacity the capacity of this bagged ensemble.
	 */
	public BaggedEnsemble(int capacity) {
		super(capacity);
	}

	@Override
	public double regress(Instance instance) {
		if (predictors.size() == 0) {
			return 0.0;
		} else {
			double prediction = 0.0;
			for (Predictor predictor : predictors) {
				Regressor regressor = (Regressor) predictor;
				prediction += regressor.regress(instance);
			}
			return prediction / predictors.size();
		}
	}

	@Override
	public int classify(Instance instance) {
		if (predictors.size() == 0) {
			// Default: return first class
			return 0;
		} else {
			Map<Integer, Integer> votes = new HashMap<>();
			for (Predictor predictor : predictors) {
				Classifier classifier = (Classifier) predictor;
				int cls = (int) classifier.classify(instance);
				if (!votes.containsKey(cls)) {
					votes.put(cls, 0);
				}
				votes.put(cls, votes.get(cls) + 1);
			}
			int prediction = 0;
			int maxVotes = 0;
			for (int cls : votes.keySet()) {
				int numVotes = votes.get(cls);
				if (numVotes > maxVotes) {
					maxVotes = numVotes;
					prediction = cls;
				}
			}
			return prediction;
		}
	}

	@Override
	public BaggedEnsemble copy() {
		BaggedEnsemble copy = new BaggedEnsemble(predictors.size());
		for (Predictor predictor : predictors) {
			copy.add(predictor.copy());
		}
		return copy;
	}

}
