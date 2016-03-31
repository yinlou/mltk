package mltk.util;

import org.junit.Assert;
import org.junit.Test;

public class OptimUtilsTest {

	@Test
	public void testGetProbability() {
		Assert.assertEquals(0.5, OptimUtils.getProbability(0), MathUtils.EPSILON);
	}
	
	@Test
	public void testGetResidual() {
		Assert.assertEquals(-1.0, OptimUtils.getResidual(1.0, 0), MathUtils.EPSILON);
	}
	
	@Test
	public void testGetPseudoResidual() {
		Assert.assertEquals(0.5, OptimUtils.getPseudoResidual(0, 1), MathUtils.EPSILON);
		Assert.assertEquals(-0.5, OptimUtils.getPseudoResidual(0, -1), MathUtils.EPSILON);
	}
	
	@Test
	public void testComputeLogisticLoss() {
		Assert.assertEquals(0.693147181, OptimUtils.computeLogisticLoss(0, 1), MathUtils.EPSILON);
		Assert.assertEquals(0.693147181, OptimUtils.computeLogisticLoss(0, -1), MathUtils.EPSILON);
		Assert.assertEquals(0.006715348, OptimUtils.computeLogisticLoss(5, 1), MathUtils.EPSILON);
		Assert.assertEquals(5.006715348, OptimUtils.computeLogisticLoss(5, -1), MathUtils.EPSILON);
	}
	
	@Test
	public void testIsConverged() {
		Assert.assertTrue(OptimUtils.isConverged(0.100000001, 0.1, 1e-6));
		Assert.assertFalse(OptimUtils.isConverged(0.15, 0.1, 1e-6));
	}
	
}
