package mltk.predictor.tree.ensemble.brt;

import org.junit.Assert;
import org.junit.Test;

import mltk.core.Instances;
import mltk.core.InstancesTestHelper;
import mltk.predictor.evaluation.Evaluator;
import mltk.predictor.evaluation.MetricFactory;
import mltk.predictor.tree.TreeLearner;

public class LogitBoostLearnerTest {

	@Test
	public void testLogitBoostLearner() {
		TreeLearner treeLearner = BRTUtils.parseTreeLearner("rrt:d:3");
		Instances instances = InstancesTestHelper.getInstance().getDenseClassificationDataset();
		
		LogitBoostLearner learner = new LogitBoostLearner();
		learner.setLearningRate(0.1);
		learner.setMetric(MetricFactory.getMetric("auc"));
		learner.setTreeLearner(treeLearner);
		
		BRT brt = learner.buildBinaryClassifier(instances, 10);
		double auc = Evaluator.evalAreaUnderROC(brt, instances);
		Assert.assertTrue(auc > 0.5);
	}
	
}
