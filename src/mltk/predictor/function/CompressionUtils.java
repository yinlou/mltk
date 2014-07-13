package mltk.predictor.function;

import mltk.predictor.BaggedEnsemble;
import mltk.predictor.BoostedEnsemble;
import mltk.predictor.Predictor;
import mltk.util.tuple.IntPair;

/**
 * Class for utility functions for compressing ensembles of univariate/bivariate functions.
 * 
 * @author Yin Lou
 * 
 */
public class CompressionUtils {

	/**
	 * Compresses a bagged ensemble of 1D functions to a single 1D function.
	 * 
	 * @param attIndex the attribute index of this regressor.
	 * @param baggedEnsemble the bagged ensemble.
	 * @return a single compressed 1D function.
	 */
	public static Function1D compress(int attIndex, BaggedEnsemble baggedEnsemble) {
		Function1D function = Function1D.getConstantFunction(attIndex, 0);
		for (int i = 0; i < baggedEnsemble.size(); i++) {
			Predictor predictor = baggedEnsemble.get(i);
			Function1D func = null;
			if (predictor instanceof Function1D) {
				func = (Function1D) predictor;
			} else {
				throw new IllegalArgumentException();
			}
			function.add(func);
		}
		function.divide(baggedEnsemble.size());
		return function;
	}

	/**
	 * Compresses a boosted ensemble to a single 1D function.
	 * 
	 * @param attIndex the attribute of this regressor.
	 * @param boostedEnsemble the boosted ensemble.
	 * @return a single compressed 1D function.
	 */
	public static Function1D compress(int attIndex, BoostedEnsemble boostedEnsemble) {
		Function1D function = Function1D.getConstantFunction(attIndex, 0);
		for (int i = 0; i < boostedEnsemble.size(); i++) {
			Predictor predictor = boostedEnsemble.get(i);
			Function1D func = null;
			if (predictor instanceof Function1D) {
				func = (Function1D) predictor;
			} else if (predictor instanceof BaggedEnsemble) {
				func = compress(attIndex, (BaggedEnsemble) predictor);
			} else {
				throw new IllegalArgumentException();
			}
			function.add(func);
		}
		return function;
	}

	/**
	 * Converts a 1D function to 1D lookup table.
	 * 
	 * @param n the number of bins.
	 * @param function the 1D function.
	 * @return a 1D lookup table.
	 */
	public static Array1D convert(int n, Function1D function) {
		double[] predictions = new double[n];
		for (int i = 0; i < predictions.length; i++) {
			predictions[i] = function.evaluate(i);
		}
		return new Array1D(function.getAttributeIndex(), predictions);
	}

	/**
	 * Compresses a bagged ensemble of 2D functions to a single 2D function.
	 * 
	 * @param attIndex1 the 1st attribute index.
	 * @param attIndex2 the 2nd attribute index.
	 * @param baggedEnsemble the bagged ensemble.
	 * @return a single compressed 2D functions.
	 */
	public static Function2D compress(int attIndex1, int attIndex2, BaggedEnsemble baggedEnsemble) {
		Function2D function = Function2D.getConstantFunction(attIndex1, attIndex2, 0);
		for (int i = 0; i < baggedEnsemble.size(); i++) {
			Predictor predictor = baggedEnsemble.get(i);
			Function2D func = null;
			if (predictor instanceof Function2D) {
				func = (Function2D) predictor;
			} else {
				throw new IllegalArgumentException();
			}
			function.add(func);
		}
		function.divide(baggedEnsemble.size());
		return function;
	}

	/**
	 * Compresses a boosted ensemble to a single 2D function.
	 * 
	 * @param attIndex1 the 1st attribute index.
	 * @param attIndex2 the 2nd attribute index.
	 * @param boostedEnsemble the boosted ensemble.
	 * @return a single compressed 2D function.
	 */
	public static Function2D compress(int attIndex1, int attIndex2, BoostedEnsemble boostedEnsemble) {
		Function2D function = Function2D.getConstantFunction(attIndex1, attIndex2, 0);
		for (int i = 0; i < boostedEnsemble.size(); i++) {
			Predictor predictor = boostedEnsemble.get(i);
			Function2D func = null;
			if (predictor instanceof Function2D) {
				func = (Function2D) predictor;
			} else if (predictor instanceof BaggedEnsemble) {
				func = compress(attIndex1, attIndex2, (BaggedEnsemble) predictor);
			} else {
				throw new IllegalArgumentException();
			}
			function.add(func);
		}
		return function;
	}

	/**
	 * Converts a 2D function to 2D lookup table.
	 * 
	 * @param n1 the number of bins for 1st attribute.
	 * @param n2 the number of bins for 2nd attribute.
	 * @param function the 2D function.
	 * @return a 2D lookup table.
	 */
	public static Array2D convert(int n1, int n2, Function2D function) {
		double[][] predictions = new double[n1][n2];
		for (int i = 0; i < n1; i++) {
			double[] preds = predictions[i];
			for (int j = 0; j < n2; j++) {
				preds[j] = function.evaluate(i, j);
			}
		}
		IntPair attIndices = function.getAttributeIndices();
		return new Array2D(attIndices.v1, attIndices.v2, predictions);
	}

}
