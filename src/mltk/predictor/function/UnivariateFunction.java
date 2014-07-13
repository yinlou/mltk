package mltk.predictor.function;

/**
 * Interface for univariate functions.
 * 
 * @author Yin Lou
 * 
 */
public interface UnivariateFunction {

	/**
	 * Computes the value for the function.
	 * 
	 * @param x the argument.
	 * @return the value for the function.
	 */
	public double evaluate(double x);

}
