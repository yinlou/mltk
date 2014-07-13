package mltk.predictor;

import mltk.core.Instance;

/**
 * Interface for classifiers that predicts the class probabilities.
 * 
 * @author Yin Lou
 * 
 */
public interface ProbabilisticClassifier extends Classifier {

	/**
	 * Returns the class probabilities.
	 * 
	 * @param instance the instance to predict.
	 * @return the class probabilities.
	 */
	public double[] predictProbabilities(Instance instance);

}
