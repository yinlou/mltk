package mltk.predictor.tree;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;

import org.junit.Assert;
import org.junit.Test;

import mltk.predictor.io.PredictorReader;
import mltk.util.MathUtils;

public class RegressionTreeTest {

	@Test
	public void testIO() {
		RegressionTree tree = RegressionTreeTestHelper.getInstance().getTree1();
		
		try {
			ByteArrayOutputStream boas = new ByteArrayOutputStream();
			PrintWriter out = new PrintWriter(boas);
			tree.write(out);
			out.close();
			
			ByteArrayInputStream bais = new ByteArrayInputStream(boas.toByteArray());
			BufferedReader in = new BufferedReader(new InputStreamReader(bais));
			RegressionTree t = PredictorReader.read(in, RegressionTree.class);
			
			Assert.assertTrue(t.root instanceof TreeInteriorNode);
			TreeInteriorNode root = (TreeInteriorNode) t.root;
			Assert.assertEquals(1, root.getSplitAttributeIndex());
			Assert.assertEquals(0.5, root.getSplitPoint(), MathUtils.EPSILON);
			Assert.assertTrue(root.left instanceof RegressionTreeLeaf);
			Assert.assertTrue(root.right instanceof TreeInteriorNode);
			
			RegressionTreeLeaf leaf1 = (RegressionTreeLeaf) root.left;
			Assert.assertEquals(0.4, leaf1.getPrediction(), MathUtils.EPSILON);
			
			TreeInteriorNode right = (TreeInteriorNode) root.right;
			Assert.assertEquals(2, right.getSplitAttributeIndex());
			Assert.assertEquals(-1.5, right.getSplitPoint(), MathUtils.EPSILON);
			Assert.assertTrue(right.left.isLeaf());
			Assert.assertTrue(right.right.isLeaf());
			
			RegressionTreeLeaf leaf2 = (RegressionTreeLeaf) right.left;
			Assert.assertEquals(0.5, leaf2.getPrediction(), MathUtils.EPSILON);
			
			RegressionTreeLeaf leaf3 = (RegressionTreeLeaf) right.right;
			Assert.assertEquals(0.6, leaf3.getPrediction(), MathUtils.EPSILON);
		} catch (Exception e) {
			Assert.fail("Should not see exception: " + e.getMessage());
		}
	}

}
