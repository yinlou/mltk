package mltk.predictor.tree.ensemble.brt;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;

import org.junit.Assert;
import org.junit.Test;

import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.InstancesTestHelper;
import mltk.predictor.io.PredictorReader;
import mltk.predictor.tree.DecisionTable;
import mltk.predictor.tree.DecisionTableTestHelper;
import mltk.predictor.tree.ensemble.BoostedDTables;
import mltk.predictor.tree.ensemble.BoostedRTrees;
import mltk.util.MathUtils;

public class BDTTest {
	
	private BDT bdt;
	
	public BDTTest() {
		DecisionTable dt1 = DecisionTableTestHelper.getInstance().getTable1();
		DecisionTable dt2 = DecisionTableTestHelper.getInstance().getTable2();
		BoostedRTrees bt = new BoostedRTrees();
		bt.add(dt1);
		bt.add(dt2);
		
		bdt = new BDT(1);
		bdt.tables[0] = new BoostedDTables(bt);
	}

	@Test
	public void testIO() {
		DecisionTable dt1 = DecisionTableTestHelper.getInstance().getTable1();
		DecisionTable dt2 = DecisionTableTestHelper.getInstance().getTable2();
		
		try {
			ByteArrayOutputStream boas = new ByteArrayOutputStream();
			PrintWriter out = new PrintWriter(boas);
			bdt.write(out);
			out.close();
			
			ByteArrayInputStream bais = new ByteArrayInputStream(boas.toByteArray());
			BufferedReader in = new BufferedReader(new InputStreamReader(bais));
			
			BDT b = PredictorReader.read(in, BDT.class);
			BoostedDTables ts = b.getDecisionTreeList(0);
			
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
	
	@Test
	public void testRegress() {
		DecisionTable dt1 = DecisionTableTestHelper.getInstance().getTable1();
		DecisionTable dt2 = DecisionTableTestHelper.getInstance().getTable2();
		BoostedRTrees bt = new BoostedRTrees();
		bt.add(dt1);
		bt.add(dt2);
		
		BRT brt = new BRT(1);
		brt.trees[0] = bt;
		
		Instances instances = InstancesTestHelper.getInstance().getDenseRegressionDataset();
		for (Instance instance : instances) {
			Assert.assertEquals(brt.regress(instance), bdt.regress(instance), MathUtils.EPSILON);
		}
	}

}
