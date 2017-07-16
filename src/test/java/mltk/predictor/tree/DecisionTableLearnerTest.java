package mltk.predictor.tree;

import java.util.List;

import org.junit.Assert;
import org.junit.Test;

import mltk.core.Attribute;
import mltk.core.Instances;
import mltk.core.InstancesTestHelper;

public class DecisionTableLearnerTest {

	@Test
	public void testDecisionTableLearner1() {
		DecisionTableLearner rtLearner = new DecisionTableLearner();
		rtLearner.setConstructionMode(DecisionTableLearner.Mode.ONE_PASS_GREEDY);
		rtLearner.setMaxDepth(2);
		
		Instances instances = InstancesTestHelper.getInstance().getDenseRegressionDataset();
		DecisionTable rt = rtLearner.build(instances);
		int[] attributeIndices = rt.getAttributeIndices();
		Assert.assertEquals(2, attributeIndices.length);
		Assert.assertEquals(0, attributeIndices[0]);
	}
	
	@Test
	public void testDecisionTableLearner2() {
		DecisionTableLearner rtLearner = new DecisionTableLearner();
		rtLearner.setConstructionMode(DecisionTableLearner.Mode.ONE_PASS_GREEDY);
		rtLearner.setMaxDepth(2);
		
		Instances instances = InstancesTestHelper.getInstance()
				.getDenseRegressionDataset().copy();
		// Apply feature selection
		List<Attribute> attributes = instances.getAttributes(1);
		instances.setAttributes(attributes);
		DecisionTable rt = rtLearner.build(instances);
		int[] attributeIndices = rt.getAttributeIndices();
		Assert.assertEquals(2, attributeIndices.length);
		Assert.assertEquals(1, attributeIndices[0]);
	}
	
}
