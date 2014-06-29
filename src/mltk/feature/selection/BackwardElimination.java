package mltk.feature.selection;

import java.util.ArrayList;
import java.util.List;

import mltk.core.Attribute;
import mltk.core.Instances;
import mltk.predictor.BaggedEnsembleLearner;
import mltk.predictor.Regressor;
import mltk.predictor.evaluation.Evaluator;
import mltk.util.StatUtils;
import mltk.util.tuple.DoublePair;
import mltk.util.tuple.Pair;

/**
 * Class for feature selection using backward elimination.
 *  
 * @author Yin Lou
 *
 */
public class BackwardElimination {

	/**
	 * Selects features using backward elimination.
	 * 
	 * @param trainSet the training set.
	 * @param validSet the validation set.
	 * @param learner the learner to use.
	 * @param numIters the number of iterations to estimate the mean and std for
	 * full complexity models.
	 * @return the list of selected features and <mean, std> pair for full 
	 * complexity models. 
	 */
	public static Pair<List<Attribute>, DoublePair> select(Instances trainSet, 
			Instances validSet, BaggedEnsembleLearner learner, int numIters) {
		List<Attribute> attributes = trainSet.getAttributes();
		List<Attribute> selected = new ArrayList<>(attributes);
		DoublePair perf = null;
		for (;;) {
			if (selected.size() == 0) {
				break;
			}
			boolean changed = false;
			trainSet.setAttributes(selected);
			perf = evaluateModel(trainSet, validSet, learner, numIters);
			System.out.println("Mean: " + perf.v1 + " Std: " + perf.v2);
			int i;
			for (i = 0; i < selected.size(); ) {
				List<Attribute> attList = new ArrayList<>(selected);
				Attribute attr = attList.get(i);
				attList.remove(i);
				trainSet.setAttributes(attList);
				Regressor regressor = (Regressor) learner.build(trainSet);
				double rmse = Evaluator.evalRMSE(regressor, validSet);
				System.out.println("Testing: " + attr.getName() + " RMSE: " + rmse);
				if (perf.v1 - perf.v2 * 3 <= rmse && rmse <= perf.v1 + perf.v2 * 3) {
					// Eliminate feature
					selected.remove(i);
					changed = true;
					System.out.println("Eliminate: " + attr.getName());
				} else {
					i++;
				}
			}
			if (!changed) {
				break;
			}
		}
		trainSet.setAttributes(attributes);
		return new Pair<List<Attribute>, DoublePair>(selected, perf);
	}
	
	private static DoublePair evaluateModel(Instances trainSet, Instances validSet, 
			BaggedEnsembleLearner learner, int numIters) {
		// Estimating std of full complexity model
		double[] rmse = new double[numIters];
		for (int i = 0; i < rmse.length; i++) {
			Regressor regressor = (Regressor) learner.build(trainSet);
			rmse[i] = Evaluator.evalRMSE(regressor, validSet);
		}
		double mean = StatUtils.mean(rmse);
		double std = StatUtils.std(rmse);
		return new DoublePair(mean, std);
	}
	
}
