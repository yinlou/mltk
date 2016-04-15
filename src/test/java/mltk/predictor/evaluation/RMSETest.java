package mltk.predictor.evaluation;

import mltk.util.MathUtils;

import org.junit.Assert;
import org.junit.Test;

public class RMSETest {

	@Test
	public void test() {
		double[] preds = {1, 2, 3, 4};
		double[] targets = {1.1, 1.9, 3.2, 4};
		RMSE metric = new RMSE();
		Assert.assertEquals(0.122474487, metric.eval(preds, targets), MathUtils.EPSILON);
	}
	
}
