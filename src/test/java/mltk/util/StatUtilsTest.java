package mltk.util;

import org.junit.Assert;
import org.junit.Test;

public class StatUtilsTest {
	
	private int[] a = {1, 4, 3, 2};
	private double[] b = {-1.2, 1.2, -5.3, 5.3};

	@Test
	public void testMax() {
		Assert.assertEquals(4, StatUtils.max(a));
		Assert.assertEquals(5.3, StatUtils.max(b), MathUtils.EPSILON);
	}
	
	@Test
	public void testIndexOfMax() {
		Assert.assertEquals(1, StatUtils.indexOfMax(a));
		Assert.assertEquals(3, StatUtils.indexOfMax(b));
	}
	
	@Test
	public void testMin() {
		Assert.assertEquals(1, StatUtils.min(a));
		Assert.assertEquals(-5.3, StatUtils.min(b), MathUtils.EPSILON);
	}
	
	@Test
	public void testIndexOfMin() {
		Assert.assertEquals(0, StatUtils.indexOfMin(a));
		Assert.assertEquals(2, StatUtils.indexOfMin(b));
	}
	
	@Test
	public void testSum() {
		Assert.assertEquals(0, StatUtils.sum(b), MathUtils.EPSILON);
	}
	
	@Test
	public void testSumSq() {
		Assert.assertEquals(59.06, StatUtils.sumSq(b), MathUtils.EPSILON);
		Assert.assertEquals(b[0] * b[0], StatUtils.sumSq(b, 0, 1), MathUtils.EPSILON);
	}
	
	@Test
	public void testMean() {
		Assert.assertEquals(0, StatUtils.mean(b), MathUtils.EPSILON);
	}
	
	@Test
	public void testVariance() {
		Assert.assertEquals(19.686666667, StatUtils.variance(b), MathUtils.EPSILON);
	}
	
	@Test
	public void testStd() {
		Assert.assertEquals(Math.sqrt(19.686666667), StatUtils.std(b), MathUtils.EPSILON);
	}
	
	@Test
	public void testRms() {
		Assert.assertEquals(Math.sqrt(59.06 / b.length), StatUtils.rms(b), MathUtils.EPSILON);
	}
	
}
