package mltk.predictor.tree.ensemble;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;

import org.junit.Assert;
import org.junit.Test;

import mltk.predictor.tree.DecisionTable;
import mltk.predictor.tree.DecisionTableTestHelper;
import mltk.util.MathUtils;

public class BoostedDTablesTest {

	@Test
	public void testIO() {
		DecisionTable dt1 = DecisionTableTestHelper.getInstance().getTable1();
		DecisionTable dt2 = DecisionTableTestHelper.getInstance().getTable2();
		BoostedDTables bt = new BoostedDTables();
		bt.add(dt1);
		bt.add(dt2);
		
		try {
			ByteArrayOutputStream boas = new ByteArrayOutputStream();
			PrintWriter out = new PrintWriter(boas);
			bt.write(out);
			out.close();
			
			ByteArrayInputStream bais = new ByteArrayInputStream(boas.toByteArray());
			BufferedReader in = new BufferedReader(new InputStreamReader(bais));
			in.readLine();
			
			BoostedDTables ts = new BoostedDTables();
			ts.read(in);
			
			Assert.assertEquals(2, ts.size());
			DecisionTable t1 = ts.get(0);
			Assert.assertArrayEquals(dt1.getAttributeIndices(), t1.getAttributeIndices());
			Assert.assertArrayEquals(dt1.getSplits(), t1.getSplits(), MathUtils.EPSILON);
			
			DecisionTable t2 = ts.get(1);
			Assert.assertArrayEquals(dt2.getAttributeIndices(), t2.getAttributeIndices());
			Assert.assertArrayEquals(dt2.getSplits(), t2.getSplits(), MathUtils.EPSILON);
		} catch (Exception e) {
			Assert.fail("Should not see exception: " + e.getMessage());
		}
	}

}
