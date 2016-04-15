package mltk.predictor.evaluation;

import mltk.util.MathUtils;

import org.junit.Assert;
import org.junit.Test;

public class MAETest {

	@Test
	public void test() {
		double[] preds = {1, 2, 3, 4};
		double[] targets = {0.1, 0.2, 0.3, 0.4};
		MAE metric = new MAE();
		Assert.assertEquals(2.25, metric.eval(preds, targets), MathUtils.EPSILON);
	}
	
}
