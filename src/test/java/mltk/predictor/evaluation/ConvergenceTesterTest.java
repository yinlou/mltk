package mltk.predictor.evaluation;

import org.junit.Assert;
import org.junit.Test;

public class ConvergenceTesterTest {

	@Test
	public void test1() {
		ConvergenceTester ct = new ConvergenceTester(10, 0, 0.8);
		ct.setMetric(new AUC());
		ct.add(0.55);
		ct.add(0.60);
		ct.add(0.70);
		ct.add(0.75);
		ct.add(0.80);
		ct.add(0.85);
		ct.add(0.90); // Peak
		ct.add(0.85);
		Assert.assertFalse(ct.isConverged());
		ct.add(0.82);
		ct.add(0.81);
		Assert.assertTrue(ct.isConverged());
	}
	
	@Test
	public void test2() {
		ConvergenceTester ct = new ConvergenceTester(10, 2, 1.0);
		ct.setMetric(new AUC());
		ct.add(0.55);
		ct.add(0.60);
		ct.add(0.70);
		ct.add(0.75);
		ct.add(0.80);
		ct.add(0.85);
		ct.add(0.90); // Peak
		ct.add(0.85);
		Assert.assertFalse(ct.isConverged());
		ct.add(0.82);
		ct.add(0.81);
		Assert.assertTrue(ct.isConverged());
	}
	
	@Test
	public void test3() {
		ConvergenceTester ct = new ConvergenceTester(10, 0, 0.8);
		ct.setMetric(new RMSE());
		ct.add(5.00);
		ct.add(4.50);
		ct.add(4.00);
		ct.add(3.50);
		ct.add(3.00);
		ct.add(2.50);
		ct.add(2.00); // Bottom
		ct.add(2.20);
		Assert.assertFalse(ct.isConverged());
		ct.add(2.10);
		ct.add(2.05);
		Assert.assertTrue(ct.isConverged());
	}
	
	@Test
	public void test4() {
		ConvergenceTester ct = new ConvergenceTester(10, 2, 1.0);
		ct.setMetric(new RMSE());
		ct.add(5.00);
		ct.add(4.50);
		ct.add(4.00);
		ct.add(3.50);
		ct.add(3.00);
		ct.add(2.50);
		ct.add(2.00); // Bottom
		ct.add(2.20);
		Assert.assertFalse(ct.isConverged());
		ct.add(2.10);
		ct.add(2.05);
		Assert.assertTrue(ct.isConverged());
	}
	
	@Test
	public void testParse() {
		ConvergenceTester ct = null;

	    // Empty
	    ct = ConvergenceTester.parse("");
	    Assert.assertEquals(ct.minNumPoints, -1);
	    Assert.assertEquals(ct.n, 0);
	    Assert.assertEquals(ct.c, 1.0, 1e-6);

	    // One parameter
	    ct = ConvergenceTester.parse("10");
	    Assert.assertEquals(ct.minNumPoints, 10);
	    Assert.assertEquals(ct.n, 0);
	    Assert.assertEquals(ct.c, 1.0, 1e-6);
	    
	    // Two parameters
	    ct = ConvergenceTester.parse("10:5");
	    Assert.assertEquals(ct.minNumPoints, 10);
	    Assert.assertEquals(ct.n, 5);
	    Assert.assertEquals(ct.c, 1.0, 1e-6);
	    

	    // Three parameters
	    ct = ConvergenceTester.parse("10:5:0.8");
	    Assert.assertEquals(ct.minNumPoints, 10);
	    Assert.assertEquals(ct.n, 5);
	    Assert.assertEquals(ct.c, 0.8, 1e-6);
	}
	
}
