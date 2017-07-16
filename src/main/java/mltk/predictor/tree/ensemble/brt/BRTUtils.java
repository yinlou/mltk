package mltk.predictor.tree.ensemble.brt;

import mltk.predictor.tree.TreeLearner;
import mltk.predictor.tree.DecisionTableLearner;
import mltk.predictor.tree.RegressionTreeLearner;

class BRTUtils {

	public static TreeLearner parseTreeLearner(String baseLearner) {
		String[] data = baseLearner.split(":");
		if (data.length != 3) {
			throw new IllegalArgumentException();
		}
		TreeLearner rtLearner = null;
		switch(data[0]) {
			case "rt":
				rtLearner = new RegressionTreeLearner();
				break;
			case "rrt":
				rtLearner = new RobustRegressionTreeLearner();
				break;
			case "dt":
				rtLearner = new DecisionTableLearner();
				break;
			case "rdt":
				rtLearner = new RobustDecisionTableLearner();
				break;
			default:
				System.err.println("Unknown regression tree learner: " + data[0]);
				throw new IllegalArgumentException();
		}
		rtLearner.setParameters(data[1] + ":" + data[2]);
		
		return rtLearner;
	}
	
}
