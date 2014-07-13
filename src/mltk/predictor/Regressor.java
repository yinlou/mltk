package mltk.predictor;

import mltk.core.Instance;

/**
 * Interface for regressors.
 * 
 * @author Yin Lou
 * 
 */
public interface Regressor extends Predictor {

	/**
	 * Regresses an instance.
	 * 
	 * @param instance the instance to regress.
	 * @return a regressed value.
	 */
	public double regress(Instance instance);

}
