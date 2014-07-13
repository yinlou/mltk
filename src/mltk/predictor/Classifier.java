package mltk.predictor;

import mltk.core.Instance;

/**
 * Interface for classfiers.
 * 
 * @author Yin Lou
 * 
 */
public interface Classifier extends Predictor {

	/**
	 * Classifies an instance.
	 * 
	 * @param instance the instance to classify.
	 * @return a classified value.
	 */
	public int classify(Instance instance);

}
