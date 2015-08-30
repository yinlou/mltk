package mltk.predictor.gam;

import mltk.core.Instances;
import mltk.predictor.HoldoutValidatedLearner;

public class LambdaGAMLearner extends HoldoutValidatedLearner {

	/**
	 * Builds a ranker.
	 * 
	 * @param trainSet the training set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param maxNumLeaves the number of leaves.
	 * @return a ranker.
	 */
	public GAM buildRanker(Instances trainSet, int maxNumIters, int maxNumLeaves) {
		return null;
	}

	@Override
	public GAM build(Instances instances) {
		return null;
	}
	
}
