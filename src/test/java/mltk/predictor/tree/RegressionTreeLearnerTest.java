package mltk.predictor.tree;

import java.util.List;

import org.junit.Assert;
import org.junit.Test;

import mltk.core.Attribute;
import mltk.core.Instances;
import mltk.core.InstancesTestHelper;

public class RegressionTreeLearnerTest {

	@Test
	public void testRegressionTreeLearner1() {
		RegressionTreeLearner rtLearner = new RegressionTreeLearner();
		rtLearner.setConstructionMode(RegressionTreeLearner.Mode.DEPTH_LIMITED);
		rtLearner.setMaxDepth(2);
		
		Instances instances = InstancesTestHelper.getInstance().getDenseRegressionDataset();
		RegressionTree rt = rtLearner.build(instances);
		TreeInteriorNode root = (TreeInteriorNode) rt.getRoot();
		Assert.assertEquals(0, root.attIndex);
		Assert.assertTrue(root.getLeftChild() != null);
		Assert.assertTrue(root.getRightChild() != null);
	}
	
	@Test
	public void testRegressionTreeLearner2() {
		RegressionTreeLearner rtLearner = new RegressionTreeLearner();
		rtLearner.setConstructionMode(RegressionTreeLearner.Mode.DEPTH_LIMITED);
		rtLearner.setMaxDepth(2);
		
		Instances instances = InstancesTestHelper.getInstance()
				.getDenseRegressionDataset().copy();
		// Apply feature selection
		List<Attribute> attributes = instances.getAttributes(1);
		instances.setAttributes(attributes);
		RegressionTree rt = rtLearner.build(instances);
		TreeInteriorNode root = (TreeInteriorNode) rt.getRoot();
		Assert.assertEquals(1, root.attIndex);
	}
	
}
