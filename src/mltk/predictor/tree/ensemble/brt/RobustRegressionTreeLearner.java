package mltk.predictor.tree.ensemble.brt;

import java.util.Collections;
import java.util.List;

import mltk.core.Instance;
import mltk.core.Instances;
import mltk.predictor.tree.RegressionTreeLearner;
import mltk.util.tuple.DoublePair;
import mltk.util.tuple.IntDoublePair;

/**
 * Class for learning regression trees in logit boost algorithms. The splitting criteria ignores the weights when
 * calculating sum of responses.
 * 
 * @author Yin Lou
 *
 */
class RobustRegressionTreeLearner extends RegressionTreeLearner {

	protected boolean getStats(Instances instances, double[] stats) {
		stats[0] = stats[1] = 0;
		if (instances.size() == 0) {
			return true;
		}
		double firstTarget = instances.get(0).getTarget();
		boolean stdIs0 = true;
		for (Instance instance : instances) {
			double weight = instance.getWeight();
			double target = instance.getTarget();
			stats[0] += weight;
			stats[1] += target;
			if (stdIs0 && target != firstTarget) {
				stdIs0 = false;
			}
		}
		stats[1] /= stats[0];
		return stdIs0;
	}

	protected void getHistogram(Instances instances, List<IntDoublePair> pairs, List<Double> uniqueValues, double w,
			double s, List<DoublePair> histogram) {
		if (pairs.size() == 0) {
			return;
		}
		double lastValue = pairs.get(0).v2;
		double totalWeight = instances.get(pairs.get(0).v1).getWeight();
		double sum = instances.get(pairs.get(0).v1).getTarget();

		for (int i = 1; i < pairs.size(); i++) {
			IntDoublePair pair = pairs.get(i);
			double value = pair.v2;
			double weight = instances.get(pairs.get(i).v1).getWeight();
			double resp = instances.get(pairs.get(i).v1).getTarget();
			if (value != lastValue) {
				uniqueValues.add(lastValue);
				histogram.add(new DoublePair(totalWeight, sum));
				lastValue = value;
				totalWeight = weight;
				sum = resp;
			} else {
				totalWeight += weight;
				sum += resp;
			}
		}
		uniqueValues.add(lastValue);
		histogram.add(new DoublePair(totalWeight, sum));

		if (pairs.size() != instances.size()) {
			// Zero entries are present
			double sumWeight = 0;
			double sumTarget = 0;
			for (DoublePair pair : histogram) {
				sumWeight += pair.v1;
				sumTarget += pair.v2;
			}

			double weightOnZero = w - sumWeight;
			double sumOnZero = s - sumTarget;
			int idx = Collections.binarySearch(uniqueValues, ZERO);
			if (idx < 0) {
				// This should always happen
				uniqueValues.add(-idx - 1, ZERO);
				histogram.add(-idx - 1, new DoublePair(weightOnZero, sumOnZero));
			}
		}
	}

}
