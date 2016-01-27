package mltk.predictor.tree.ensemble;

import mltk.predictor.Learner;
import mltk.predictor.tree.TreeLearner;

/**
 * Class for learning tree ensembles.
 * 
 * @author Yin Lou
 *
 */
public abstract class TreeEnsembleLearner extends Learner {

	protected TreeLearner treeLearner;
	
	public TreeLearner getTreeLearner() {
		return treeLearner;
	}
	
	public void setTreeLearner(TreeLearner treeLearner) {
		this.treeLearner = treeLearner;
	}
	
}
