package mltk.core;

import org.junit.Assert;
import org.junit.Test;

import mltk.util.MathUtils;

public class BinsTest {

	@Test
	public void testBins() {
		Bins bins = new Bins(new double[] {1, 5, 6}, new double[] {0.5, 2.5, 5.5});
		
		Assert.assertEquals(0, bins.getIndex(-1));
		Assert.assertEquals(0, bins.getIndex(0.3));
		Assert.assertEquals(0, bins.getIndex(1));
		Assert.assertEquals(1, bins.getIndex(1.1));
		Assert.assertEquals(1, bins.getIndex(5));
		Assert.assertEquals(2, bins.getIndex(5.5));
		Assert.assertEquals(2, bins.getIndex(6.5));
		
		Assert.assertEquals(0.5, bins.getValue(0), MathUtils.EPSILON);
		Assert.assertEquals(2.5, bins.getValue(1), MathUtils.EPSILON);
		Assert.assertEquals(5.5, bins.getValue(2), MathUtils.EPSILON);
	}
	
}
