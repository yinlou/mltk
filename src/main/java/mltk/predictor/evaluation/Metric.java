package mltk.predictor.evaluation;

import java.util.List;

import mltk.core.Instances;
import mltk.util.MathUtils;

/**
 * Class for evaluation metrics.
 * 
 * @author Yin Lou
 *
 */
public abstract class Metric {

	private boolean isLargerBetter;

	/**
	 * Constructor.
	 * 
	 * @param isLargerBetter <code>true</code> if larger value is better.
	 */
	public Metric(boolean isLargerBetter) {
		this.isLargerBetter = isLargerBetter;
	}
	
	/**
	 * Returns <code>true</code> if larger value is better for this metric.
	 * 
	 * @return <code>true</code> if larger value is better for this metric.
	 */
	public boolean isLargerBetter() {
		return isLargerBetter;
	}

	/**
	 * Returns <code>true</code> if the first value is better.
	 * 
	 * @param a the 1st value.
	 * @param b the 2nd value.
	 * @return <code>true</code> if the first value is better.
	 */
	public boolean isFirstBetter(double a, double b) {
		return MathUtils.isFirstBetter(a, b, isLargerBetter);
	}
	
	/**
	 * Returns the worst value of this metric.
	 * 
	 * @return the worst value of this metric.
	 */
	public double worstValue() {
		if (isLargerBetter) {
			return Double.NEGATIVE_INFINITY;
		} else {
			return Double.POSITIVE_INFINITY;
		}
	}
	
	/**
	 * Evaluates predictions on a dataset.
	 * 
	 * @param preds the predictions.
	 * @param instances the dataset.
	 * @return the evaluation measure.
	 */
	public abstract double eval(double[] preds, Instances instances);
	
	/**
	 * Returns the index of best metric value in a list.
	 * 
	 * @param list the list of metric values.
	 * @return the index of best metric value in a list.
	 */
	public int searchBestMetricValueIndex(List<Double> list) {
		double bestSoFar = worstValue();
		int idx = -1;
		for (int i = 0; i < list.size(); i++) {
			if (isFirstBetter(list.get(i), bestSoFar)) {
				bestSoFar = list.get(i);
				idx = i;
			}
		}
		return idx;
	}

}
