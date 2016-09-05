package mltk.predictor.evaluation;

import org.junit.Assert;
import org.junit.Test;

public class MetricFactoryTest {
	
	@Test
	public void test() {
		Assert.assertEquals(AUC.class, MetricFactory.getMetric("AUC").getClass());
		Assert.assertEquals(Error.class, MetricFactory.getMetric("Error").getClass());
		Assert.assertEquals(LogisticLoss.class, MetricFactory.getMetric("LogisticLoss").getClass());
		Assert.assertEquals(LogLoss.class, MetricFactory.getMetric("LogLoss").getClass());
		Assert.assertEquals(LogLoss.class, MetricFactory.getMetric("LogLoss:True").getClass());
		Assert.assertEquals(true, ((LogLoss) MetricFactory.getMetric("LogLoss:True")).isRawScore());
		Assert.assertEquals(MAE.class, MetricFactory.getMetric("MAE").getClass());
		Assert.assertEquals(RMSE.class, MetricFactory.getMetric("RMSE").getClass());
	}
	
}
