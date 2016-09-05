package mltk.predictor.evaluation;

import java.util.HashMap;
import java.util.Map;

/**
 * Factory class for creating metrics.
 *  
 * @author Yin Lou
 *
 */
public class MetricFactory {
	
	private static Map<String, Metric> map;
	
	static {
		map = new HashMap<>();
		map.put("auc", new AUC());
		map.put("error", new Error());
		map.put("logisticloss", new LogisticLoss());
		map.put("logloss", new LogLoss(false));
		map.put("logloss_t", new LogLoss(true));
		map.put("mae", new MAE());
		map.put("rmse", new RMSE());
	}

	/**
	 * Returns the metric.
	 * 
	 * @param str the metric string.
	 * @return the metric.
	 */
	public static Metric getMetric(String str) {
		String[] data = str.toLowerCase().split(":");
		String name = data[0];
		if (data.length == 1) {
			if (!map.containsKey(name)) {
				throw new IllegalArgumentException("Unrecognized metric name: " + name);
			} else {
				return map.get(name);
			}
		} else {
			if (name.equals("logloss")) {
				if (data[1].startsWith("t")) {
					return map.get("logloss_t");
				} else {
					return map.get(name);
				}
			} else if (map.containsKey(name)) {
				return map.get(name);
			}
		}
		return null;
	}
	
}
