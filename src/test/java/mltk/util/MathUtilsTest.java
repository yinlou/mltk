package mltk.util;

import org.junit.Assert;
import org.junit.Test;

public class MathUtilsTest {
	
	@Test
	public void testEquals() {
		Assert.assertTrue(MathUtils.equals(0.1, 0.10000001));
		Assert.assertFalse(MathUtils.equals(0.0, 1.0));
	}

	@Test
	public void testIndicator() {
		Assert.assertEquals(1, MathUtils.indicator(true));
		Assert.assertEquals(0, MathUtils.indicator(false));
	}
	
	@Test
	public void testIsFirstBetter() {
		Assert.assertTrue(MathUtils.isFirstBetter(0.5, 0, true));
		Assert.assertFalse(MathUtils.isFirstBetter(0.5, 0, false));
	}
	
	@Test
	public void testIsInteger() {
		Assert.assertTrue(MathUtils.isInteger(1.0));
		Assert.assertFalse(MathUtils.isInteger(1.1));
	}
	
	@Test
	public void testIsZero() {
		Assert.assertTrue(MathUtils.isZero(MathUtils.EPSILON / 2));
		Assert.assertFalse(MathUtils.isZero(MathUtils.EPSILON * 2));
	}
	
	@Test
	public void testSigmoid() {
		Assert.assertEquals(0.5, MathUtils.sigmoid(0), MathUtils.EPSILON);
	}
	
	@Test
	public void testSign() {
		Assert.assertEquals(1, MathUtils.sign(0.5));
		Assert.assertEquals(0, MathUtils.sign(0.0));
		Assert.assertEquals(-1, MathUtils.sign(-0.5));
		
		Assert.assertEquals(1, MathUtils.sign(2));
		Assert.assertEquals(0, MathUtils.sign(0));
		Assert.assertEquals(-1, MathUtils.sign(-2));
	}
	
}
