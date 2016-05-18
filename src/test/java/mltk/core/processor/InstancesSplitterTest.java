package mltk.core.processor;

import org.junit.Assert;
import org.junit.Test;

import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.InstancesTestHelper;
import mltk.util.Random;

public class InstancesSplitterTest {
	
	@Test
	public void testSamplingTrainValid() {
		Random.getInstance().setSeed(5);
		Instances instances = InstancesTestHelper.getInstance()
				.getDenseRegressionDataset();
		Instances[] datasets = InstancesSplitter.split(instances, 0.8);
		Instances train = datasets[0];
		Instances valid = datasets[1];
		Assert.assertEquals((int) (instances.size() * 0.8), train.size());
		Assert.assertEquals((int) (instances.size() * 0.2), valid.size());
	}
	
	@Test
	public void testSamplingTrainValidTest() {
		Random.getInstance().setSeed(5);
		Instances instances = InstancesTestHelper.getInstance()
				.getDenseRegressionDataset();
		Instances[] datasets = InstancesSplitter.split(instances, 0.7, 0.1, 0.1);
		Instances train = datasets[0];
		Instances valid = datasets[1];
		Instances test = datasets[2];
		Assert.assertEquals((int) (instances.size() * 0.7), train.size());
		Assert.assertEquals((int) (instances.size() * 0.1), valid.size());
		Assert.assertEquals((int) (instances.size() * 0.1), test.size());
	}
	
	@Test
	public void testStratifiedSampling() {
		Random.getInstance().setSeed(5);
		Instances instances = InstancesTestHelper.getInstance()
				.getDenseClassificationDataset();
		Instances[] datasets = InstancesSplitter.split(instances, "target", 0.8, 0.2);
		Instances train = datasets[0];
		Instances valid = datasets[1];
		int numPosTrain = 0;
		for (Instance instance : train) {
			if (instance.getTarget() == 1) {
				numPosTrain++;
			}
		}
		int numPosValid = 0;
		for (Instance instance : valid) {
			if (instance.getTarget() == 1) {
				numPosValid++;
			}
		}
		Assert.assertTrue(numPosTrain < train.size() / 3);
		Assert.assertTrue(numPosValid < valid.size() / 3);
		Assert.assertEquals((int) (instances.size() * 0.8), train.size());
		Assert.assertEquals((int) (instances.size() * 0.2), valid.size());
	}

}
