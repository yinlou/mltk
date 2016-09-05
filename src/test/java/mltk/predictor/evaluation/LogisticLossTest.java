package mltk.predictor.evaluation;

import mltk.util.MathUtils;

import org.junit.Assert;
import org.junit.Test;

public class LogisticLossTest {
	
	@Test
	public void test() {
		double[] preds = {5, -5, -3, 3};
		double[] targets = {1, 0, 0, 1};
		LogisticLoss metric = new LogisticLoss();
		Assert.assertEquals(0.02765135, metric.eval(preds, targets), MathUtils.EPSILON);
	}
	
}
