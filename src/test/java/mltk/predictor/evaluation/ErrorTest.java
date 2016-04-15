package mltk.predictor.evaluation;

import mltk.util.MathUtils;

import org.junit.Assert;
import org.junit.Test;

public class ErrorTest {
	
	@Test
	public void testLabel() {
		double[] preds = {1, 0, 1, 0};
		double[] targets = {1, 0, 0, 1};
		Error metric = new Error();
		Assert.assertEquals(0.5, metric.eval(preds, targets), MathUtils.EPSILON);
	}

	@Test
	public void testProbability() {
		double[] preds = {2, -1.5, 0.3, 5};
		double[] targets = {1, 0, 0, 1};
		Error metric = new Error();
		Assert.assertEquals(0.25, metric.eval(preds, targets), MathUtils.EPSILON);
	}
	
}
