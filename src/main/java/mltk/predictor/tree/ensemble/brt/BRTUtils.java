package mltk.predictor.tree.ensemble.brt;

import mltk.predictor.tree.TreeLearner;
import mltk.predictor.tree.RegressionTreeLearner.Mode;

class BRTUtils {

	public static TreeLearner parseTreeLearner(String baseLearner) {
		String[] data = baseLearner.split(":");
		if (data.length != 3) {
			throw new IllegalArgumentException();
		}
		TreeLearner rtLearner = null;
		if (data[0].equalsIgnoreCase("rt")) {
			RobustRegressionTreeLearner learner = new RobustRegressionTreeLearner();
			String mode = data[1];
			switch (mode) {
				case "a":
					learner.setConstructionMode(Mode.ALPHA_LIMITED);
					learner.setAlpha(Double.parseDouble(data[2]));
					break;
				case "d":
					learner.setConstructionMode(Mode.DEPTH_LIMITED);
					learner.setMaxDepth(Integer.parseInt(data[2]));
					break;
				case "l":
					learner.setConstructionMode(Mode.NUM_LEAVES_LIMITED);
					learner.setMaxNumLeaves(Integer.parseInt(data[2]));
					break;
				case "s":
					learner.setConstructionMode(Mode.MIN_LEAF_SIZE_LIMITED);
					learner.setMinLeafSize(Integer.parseInt(data[2]));
				default:
					throw new IllegalArgumentException();
			}
			rtLearner = learner;
		} else {
			System.err.println("Unknown regression tree learner");
			throw new IllegalArgumentException();
		}
		
		return rtLearner;
	}
	
}
