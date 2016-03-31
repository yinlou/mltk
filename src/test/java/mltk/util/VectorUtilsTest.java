package mltk.util;

import org.junit.Assert;
import org.junit.Test;

public class VectorUtilsTest {

	@Test
	public void testAdd() {
		double[] a = {1, 2, 3, 4};
		double[] b = {2, 3, 4, 5};
		VectorUtils.add(a, 1);
		Assert.assertArrayEquals(b, a, MathUtils.EPSILON);
	}
	
	@Test
	public void testSubtract() {
		double[] a = {1, 2, 3, 4};
		double[] b = {2, 3, 4, 5};
		VectorUtils.subtract(b, 1);
		Assert.assertArrayEquals(a, b, MathUtils.EPSILON);
	}
	
	@Test
	public void testMultiply() {
		double[] a = {1, 2, 3, 4};
		double[] b = {2, 4, 6, 8};
		VectorUtils.multiply(a, 2);
		Assert.assertArrayEquals(b, a, MathUtils.EPSILON);
	}
	
	@Test
	public void testDivide() {
		double[] a = {1, 2, 3, 4};
		double[] b = {2, 4, 6, 8};
		VectorUtils.divide(b, 2);
		Assert.assertArrayEquals(a, b, MathUtils.EPSILON);
	}
	
	@Test
	public void testL2norm() {
		double[] a = {1, 2, 3, 4};
		Assert.assertEquals(5.477225575, VectorUtils.l2norm(a), MathUtils.EPSILON);
	}
	
	@Test
	public void testL1norm() {
		double[] a = {1, -2, 3, -4};
		Assert.assertEquals(10, VectorUtils.l1norm(a), MathUtils.EPSILON);
	}
	
	@Test
	public void testDotProduct() {
		double[] a = {1, 2, 3, 4};
		double[] b = {0, -1, 0, 1};
		Assert.assertEquals(2, VectorUtils.dotProduct(a, b), MathUtils.EPSILON);
	}
	
	@Test
	public void testCorrelation() {
		double[] a = {1, 2, 3, 4};
		double[] b = {2, 4, 6, 8};
		double[] c = {-2, -4, -6, -8};
		Assert.assertEquals(1, VectorUtils.correlation(a, b), MathUtils.EPSILON);
		Assert.assertEquals(-1, VectorUtils.correlation(a, c), MathUtils.EPSILON);
	}
	
}
