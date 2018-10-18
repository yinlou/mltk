package mltk.predictor.evaluation;

import java.util.ArrayList;
import java.util.List;

/**
 * Class for testing convergence given a list of metric values.
 * 
 * @author Yin Lou
 *
 */
public class ConvergenceTester {

	protected int minNumPoints;
	protected int n;
	protected double c;
	protected double bestSoFar;
	protected int bestIdx;
	protected Metric metric;
	protected List<Double> measureList;
	
	/**
	 * Parses the convergence criteria string.
	 * 
	 * @param cc the convergence criteria string.
	 * @return a convergence tester.
	 */
	public static ConvergenceTester parse(String cc) {
		int minNumPoints = -1;
		int n = 0;
		double c = 1.0;
		if (cc != null && !cc.equals("")) {
			String[] strs = cc.split(":");
			if (strs.length > 0) {
				minNumPoints = Integer.parseInt(strs[0]);
			}
			if (strs.length > 1) {
				n = Integer.parseInt(strs[1]);
			}
			if (strs.length > 2) {
				c = Double.parseDouble(strs[2]);
			}
		}
		return new ConvergenceTester(minNumPoints, n, c);
	}
	
	/**
	 * Constructor.
	 * 
	 * @param minNumPoints the minimum number of points to be considered convergence.
	 * @param c a constant factor in [0, 1].
	 */
	public ConvergenceTester(int minNumPoints, double c) {
		this(minNumPoints, 0, c, 1000);
	}
	
	/**
	 * Constructor.
	 * 
	 * @param minNumPoints the minimum number of points to be considered convergence.
	 * @param n the n.
	 */
	public ConvergenceTester(int minNumPoints, int n) {
		this(minNumPoints, n, 1.0, 1000);
	}
	
	/**
	 * Constructor.
	 * 
	 * @param minNumPoints the minimum number of points to be considered convergence.
	 * @param n the n.
	 * @param c a constant factor in [0, 1].
	 */
	public ConvergenceTester(int minNumPoints, int n, double c) {
		this(minNumPoints, n, c, 1000);
	}
	
	/**
	 * Constructor. A list of metric values is viewed as converged if the list
	 * has at least {@code minNumPoints} and the {@code getBestIndex() + n < size() * c}.
	 * 
	 * @param minNumPoints the minimum number of points to be considered convergence.
	 * @param n the n.
	 * @param c a constant factor in [0, 1].
	 * @param capacity the initial capacity.
	 */
	public ConvergenceTester(int minNumPoints, int n, double c, int capacity) {
		if (n < 0) {
			throw new IllegalArgumentException("n has to be non-negative.");
		}
		if (!(c >= 0 && c <= 1)) {
			throw new IllegalArgumentException("c should to be in [0, 1].");
		}
		this.minNumPoints = minNumPoints;
		this.n = n;
		this.c = c;
		measureList = new ArrayList<>(capacity);
	}
	
	/**
	 * Returns the metric.
	 * 
	 * @return the metric.
	 */
	public Metric getMetric() {
		return metric;
	}
	
	/**
	 * Sets the metric. This method also resets internal status of this tester.
	 * 
	 * @param metric the metric to set.
	 */
	public void setMetric(Metric metric) {
		this.metric = metric;
		measureList.clear();
		bestSoFar = metric.worstValue();
		bestIdx = -1;
	}
	
	/**
	 * Adds a measure.
	 * 
	 * @param measure the metric value to add.
	 */
	public void add(double measure) {
		measureList.add(measure);
		
		if (metric.isFirstBetter(measure, bestSoFar)) {
			bestSoFar = measure;
			bestIdx = measureList.size() - 1;
		}
	}
	
	/**
	 * Returns the index of best metric value so far.
	 * 
	 * @return the index of best metric value so far.
	 */
	public int getBestIndex() {
		return bestIdx;
	}
	
	/**
	 * Returns the best measure value so far.
	 * 
	 * @return the best measure value so far.
	 */
	public double getBestMetricValue() {
		return bestSoFar;
	}
	
	/**
	 * Returns the number of points.
	 * 
	 * @return the number of points.
	 */
	public int size() {
		return measureList.size();
	}
	
	/**
	 * Returns the list of metric values.
	 * 
	 * @return the list of metric values.
	 */
	public List<Double> getMeasureList() {
		return measureList;
	}
	
	/**
	 * Returns {@code true} if the series is converged.
	 * 
	 * @return {@code true} if the series is converged.
	 */
	public boolean isConverged() {
		return minNumPoints >= 0 && measureList.size() >= minNumPoints
				&& bestIdx > 0 && bestIdx + n < measureList.size() * c;
	}
	
}
