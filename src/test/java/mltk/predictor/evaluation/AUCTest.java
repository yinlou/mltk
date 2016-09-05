package mltk.predictor.evaluation;

import mltk.util.MathUtils;

import org.junit.Assert;
import org.junit.Test;

public class AUCTest {
	
	@Test
	public void test1() {
		double[] preds = {0.8, 0.1, 0.05, 0.9};
		double[] targets = {1, 0, 0, 1};
		AUC metric = new AUC();
		Assert.assertEquals(1, metric.eval(preds, targets), MathUtils.EPSILON);
	}
	
	@Test
	public void test2() {
		double[] preds = {0.5, 0.5, 0.5, 0.5};
		double[] targets = {1, 0, 0, 1};
		AUC metric = new AUC();
		Assert.assertEquals(0.5, metric.eval(preds, targets), MathUtils.EPSILON);
	}
	
}
