package mltk.predictor.evaluation;

/**
 * Factory class for creating metrics.
 *  
 * @author Yin Lou
 *
 */
public class MetricFactory {

	/**
	 * Returns the metric.
	 * 
	 * @param name the metric name.
	 * @return the metric.
	 */
	public static Metric getMetric(String name) {
		Metric metric = null;
		if (name.startsWith("a")) {
			metric = new AUC();
		} else if (name.startsWith("c")) {
			metric = new Error();
		} else if (name.startsWith("r")) {
			metric = new RMSE();
		}
		return metric;
	}
	
}
