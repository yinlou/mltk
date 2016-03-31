package mltk.util;

import org.junit.Assert;
import org.junit.Test;

public class ArrayUtilsTest {

	@Test
	public void testParseDoubleArray() {
		String str = "[1.1, 2.2, 3.3, 4.4]";
		double[] a = {1.1, 2.2, 3.3, 4.4};
		Assert.assertArrayEquals(a, ArrayUtils.parseDoubleArray(str), MathUtils.EPSILON);
	}
	
	@Test
	public void testParseIntArray() {
		String str = "[1, 2, 3, 4]";
		int[] a = {1, 2, 3, 4};
		Assert.assertArrayEquals(a, ArrayUtils.parseIntArray(str));
	}
	
	@Test
	public void testIsConstant() {
		int[] a = {1, 1, 1};
		int[] b = {2, 1, 1};
		Assert.assertTrue(ArrayUtils.isConstant(a, 0, a.length, 1));
		Assert.assertFalse(ArrayUtils.isConstant(b, 0, b.length, 1));
		Assert.assertTrue(ArrayUtils.isConstant(b, 1, b.length, 1));
		
		double[] c = {0.1, 0.1, 0.1, 0.1};
		double[] d = {0.2, 0.1, 0.1, 0.1};
		Assert.assertTrue(ArrayUtils.isConstant(c, 0, c.length, 0.1));
		Assert.assertFalse(ArrayUtils.isConstant(d, 0, d.length, 0.1));
		Assert.assertTrue(ArrayUtils.isConstant(d, 1, d.length, 0.1));
	}
	
	@Test
	public void testGetMedian() {
		double[] a = {0.7, 0.4, 0.3, 0.2, 0.5, 0.6, 0.1};
		Assert.assertEquals(0.4, ArrayUtils.getMedian(a), MathUtils.EPSILON);
	}
	
}
