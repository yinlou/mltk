package mltk.predictor.tree.ensemble.brt;

import org.junit.Assert;
import org.junit.Test;

import mltk.predictor.tree.RegressionTreeLearner;
import mltk.predictor.tree.TreeLearner;
import mltk.util.MathUtils;

public class BRTUtilsTest {

	@Test
	public void testParseRegressionTreeLearner1() {
		String baseLearner = "rt:l:100";
		TreeLearner treeLearner = null;
		treeLearner = BRTUtils.parseTreeLearner(baseLearner);
		Assert.assertTrue(treeLearner instanceof RegressionTreeLearner);
		RegressionTreeLearner rtLearner = (RegressionTreeLearner) treeLearner;
		Assert.assertEquals(RegressionTreeLearner.Mode.NUM_LEAVES_LIMITED, rtLearner.getConstructionMode());
		Assert.assertEquals(100, rtLearner.getMaxNumLeaves());
	}
	
	@Test
	public void testParseRegressionTreeLearner2() {
		String baseLearner = "rt:d:5";
		TreeLearner treeLearner = null;
		treeLearner = BRTUtils.parseTreeLearner(baseLearner);
		Assert.assertTrue(treeLearner instanceof RegressionTreeLearner);
		RegressionTreeLearner rtLearner = (RegressionTreeLearner) treeLearner;
		Assert.assertEquals(RegressionTreeLearner.Mode.DEPTH_LIMITED, rtLearner.getConstructionMode());
		Assert.assertEquals(5, rtLearner.getMaxDepth());
	}
	
	@Test
	public void testParseRegressionTreeLearner3() {
		String baseLearner = "rt:a:0.01";
		TreeLearner treeLearner = null;
		treeLearner = BRTUtils.parseTreeLearner(baseLearner);
		Assert.assertTrue(treeLearner instanceof RegressionTreeLearner);
		RegressionTreeLearner rtLearner = (RegressionTreeLearner) treeLearner;
		Assert.assertEquals(RegressionTreeLearner.Mode.ALPHA_LIMITED, rtLearner.getConstructionMode());
		Assert.assertEquals(0.01, rtLearner.getAlpha(), MathUtils.EPSILON);
	}
	
	@Test
	public void testParseRegressionTreeLearner4() {
		String baseLearner = "rt:s:50";
		TreeLearner treeLearner = null;
		treeLearner = BRTUtils.parseTreeLearner(baseLearner);
		Assert.assertTrue(treeLearner instanceof RegressionTreeLearner);
		RegressionTreeLearner rtLearner = (RegressionTreeLearner) treeLearner;
		Assert.assertEquals(RegressionTreeLearner.Mode.MIN_LEAF_SIZE_LIMITED, rtLearner.getConstructionMode());
		Assert.assertEquals(50, rtLearner.getMinLeafSize());
	}
	
	@Test
	public void testParseRobustRegressionTreeLearner1() {
		String baseLearner = "rrt:l:100";
		TreeLearner treeLearner = null;
		treeLearner = BRTUtils.parseTreeLearner(baseLearner);
		Assert.assertTrue(treeLearner instanceof RegressionTreeLearner);
		RegressionTreeLearner rtLearner = (RegressionTreeLearner) treeLearner;
		Assert.assertEquals(RegressionTreeLearner.Mode.NUM_LEAVES_LIMITED, rtLearner.getConstructionMode());
		Assert.assertEquals(100, rtLearner.getMaxNumLeaves());
	}
	
	@Test
	public void testParseRobustRegressionTreeLearner2() {
		String baseLearner = "rrt:d:5";
		TreeLearner treeLearner = null;
		treeLearner = BRTUtils.parseTreeLearner(baseLearner);
		Assert.assertTrue(treeLearner instanceof RegressionTreeLearner);
		RegressionTreeLearner rtLearner = (RegressionTreeLearner) treeLearner;
		Assert.assertEquals(RegressionTreeLearner.Mode.DEPTH_LIMITED, rtLearner.getConstructionMode());
		Assert.assertEquals(5, rtLearner.getMaxDepth());
	}
	
	@Test
	public void testParseRobustRegressionTreeLearner3() {
		String baseLearner = "rrt:a:0.01";
		TreeLearner treeLearner = null;
		treeLearner = BRTUtils.parseTreeLearner(baseLearner);
		Assert.assertTrue(treeLearner instanceof RegressionTreeLearner);
		RegressionTreeLearner rtLearner = (RegressionTreeLearner) treeLearner;
		Assert.assertEquals(RegressionTreeLearner.Mode.ALPHA_LIMITED, rtLearner.getConstructionMode());
		Assert.assertEquals(0.01, rtLearner.getAlpha(), MathUtils.EPSILON);
	}
	
	@Test
	public void testParseRobustRegressionTreeLearner4() {
		String baseLearner = "rrt:s:50";
		TreeLearner treeLearner = null;
		treeLearner = BRTUtils.parseTreeLearner(baseLearner);
		Assert.assertTrue(treeLearner instanceof RegressionTreeLearner);
		RegressionTreeLearner rtLearner = (RegressionTreeLearner) treeLearner;
		Assert.assertEquals(RegressionTreeLearner.Mode.MIN_LEAF_SIZE_LIMITED, rtLearner.getConstructionMode());
		Assert.assertEquals(50, rtLearner.getMinLeafSize());
	}
	
}
