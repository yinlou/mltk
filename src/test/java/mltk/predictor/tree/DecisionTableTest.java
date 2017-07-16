package mltk.predictor.tree;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;

import org.junit.Assert;
import org.junit.Test;

import mltk.core.Instance;
import mltk.core.InstancesTestHelper;
import mltk.predictor.io.PredictorReader;
import mltk.util.MathUtils;

public class DecisionTableTest {

	@Test
	public void testIO() {
		DecisionTable dt = DecisionTableTestHelper.getInstance().getTable1();
		Instance instance = InstancesTestHelper.getInstance().getDenseRegressionDataset().get(0);
		
		try {
			ByteArrayOutputStream boas = new ByteArrayOutputStream();
			PrintWriter out = new PrintWriter(boas);
			dt.write(out);
			out.close();
			
			ByteArrayInputStream bais = new ByteArrayInputStream(boas.toByteArray());
			BufferedReader in = new BufferedReader(new InputStreamReader(bais));
			DecisionTable t = PredictorReader.read(in, DecisionTable.class);
			
			Assert.assertArrayEquals(dt.getAttributeIndices(), t.getAttributeIndices());
			Assert.assertArrayEquals(dt.getSplits(), t.getSplits(), MathUtils.EPSILON);
			Assert.assertEquals(dt.regress(instance), t.regress(instance), MathUtils.EPSILON);
		} catch (Exception e) {
			Assert.fail("Should not see exception: " + e.getMessage());
		}
	}
	
	@Test
	public void testRegress() {
		DecisionTable dt = DecisionTableTestHelper.getInstance().getTable1();
		Instance instance = InstancesTestHelper.getInstance().getDenseRegressionDataset().get(0);
		
		Assert.assertEquals(0.7, dt.regress(instance), MathUtils.EPSILON);
	}

}
