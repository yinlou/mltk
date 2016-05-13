package mltk.core.io;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

import mltk.core.Attribute;
import mltk.core.Attribute.Type;
import mltk.core.BinnedAttribute;
import mltk.core.NominalAttribute;
import mltk.util.tuple.Pair;

public class AttributesReaderTest {

	@Test
	public void testIO() {
		ByteArrayOutputStream boas = new ByteArrayOutputStream();
		PrintWriter out = new PrintWriter(boas);
		out.println("f1: cont");
		out.println("f2: {a, b, c}");
		out.println("f3: binned (256)");
		out.println("f4: binned (3;[1, 5, 6];[0.5, 2.5, 3])");
		out.println("label: cont (target)");
		out.flush();
		out.close();
		
		ByteArrayInputStream bais = new ByteArrayInputStream(boas.toByteArray());
		BufferedReader br = new BufferedReader(new InputStreamReader(bais));
		Pair<List<Attribute>, Attribute> pair = null;
		try {
			pair = AttributesReader.read(br);
		} catch (IOException e) {
			Assert.fail("Should not see exception: " + e.getMessage());
		}
		
		List<Attribute> attributes = pair.v1;
		Attribute targetAtt = pair.v2;
		Assert.assertEquals("label", targetAtt.getName());
		Assert.assertEquals(4, attributes.size());
		for (int i = 0; i < attributes.size(); i++) {
			Assert.assertEquals(i, attributes.get(i).getIndex());
		}
		Assert.assertEquals(Type.NUMERIC, attributes.get(0).getType());
		Assert.assertEquals(Type.NOMINAL, attributes.get(1).getType());
		Assert.assertEquals(Type.BINNED, attributes.get(2).getType());
		Assert.assertEquals(Type.BINNED, attributes.get(3).getType());
		Assert.assertArrayEquals(new String[] {"a", "b", "c"},
				((NominalAttribute) attributes.get(1)).getStates());
		Assert.assertEquals(256, ((BinnedAttribute) attributes.get(2)).getNumBins());
		Assert.assertEquals(3, ((BinnedAttribute) attributes.get(3)).getNumBins());
	}
	
}
