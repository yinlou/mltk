package mltk.core.io;

import org.junit.Assert;
import org.junit.Test;

import mltk.core.Instance;

public class InstancesReaderTest {

	@Test
	public void testDenseFormat() {
		String[] data = {"0.0", "1.5", "?", "3"};
		Instance instance = InstancesReader.parseDenseInstance(data, 3);
		Assert.assertTrue(instance.isMissing(2));
	}
	
}
