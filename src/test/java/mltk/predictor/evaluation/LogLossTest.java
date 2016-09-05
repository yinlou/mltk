package mltk.predictor.evaluation;

import mltk.util.MathUtils;

import org.junit.Assert;
import org.junit.Test;

public class LogLossTest {
	
	@Test
	public void testProb() {
		double[] preds = {0.8, 0.1, 0.05, 0.9};
		double[] targets = {1, 0, 0, 1};
		LogLoss metric = new LogLoss(false);
		Assert.assertEquals(0.485157877, metric.eval(preds, targets), MathUtils.EPSILON);
	}
	
	@Test
	public void testRawScore() {
		double[] preds = {5, -5, -3, 3};
		double[] targets = {1, 0, 0, 1};
		LogLoss metric = new LogLoss(true);
		Assert.assertEquals(0.1106054, metric.eval(preds, targets), MathUtils.EPSILON);
	}
	
}
