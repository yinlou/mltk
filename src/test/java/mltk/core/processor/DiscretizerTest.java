package mltk.core.processor;

import org.junit.Assert;
import org.junit.Test;

import mltk.core.BinnedAttribute;
import mltk.core.Instances;
import mltk.core.InstancesTestHelper;
import mltk.util.MathUtils;

public class DiscretizerTest {
	
	@Test
	public void testMissingValue() {
		Instances instances = InstancesTestHelper.getInstance().getDenseClassificationDatasetWMissing().copy();
		Discretizer.discretize(instances, 0, 10);
		Assert.assertTrue(instances.getAttributes().get(0).getClass() == BinnedAttribute.class);
		for (int i = 0; i < 10; i++) {
			Assert.assertTrue(instances.get(i).isMissing(0));
		}
		for (int i = 10; i < 20; i++) {
			Assert.assertFalse(instances.get(i).isMissing(0));
			Assert.assertTrue(MathUtils.isInteger(instances.get(i).getValue(0)));
		}
	}

}
