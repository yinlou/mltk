package mltk.predictor.glm;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;

import mltk.predictor.io.PredictorReader;
import mltk.util.MathUtils;

import org.junit.Assert;
import org.junit.Test;

public class GLMTest {

	@Test
	public void testIO() {
		double[] intercept = {1.0, -1.0};
		double[][] w = {
				{1, 2, 3},
				{-1, -2, -3}
		};
		GLM glm = new GLM(intercept, w);
		
		ByteArrayOutputStream boas = new ByteArrayOutputStream();
		PrintWriter out = new PrintWriter(boas);
		try {
			glm.write(out);
		} catch (Exception e) {
			Assert.fail("Should not see exception: " + e.getMessage());
		}
		out.flush();
		out.close();
		
		ByteArrayInputStream bais = new ByteArrayInputStream(boas.toByteArray());
		BufferedReader br = new BufferedReader(new InputStreamReader(bais));
		try {
			GLM parsedGLM = PredictorReader.read(br, GLM.class);
			Assert.assertEquals(intercept.length, parsedGLM.intercept().length);
			Assert.assertEquals(w.length, parsedGLM.coefficients().length);
			Assert.assertArrayEquals(intercept, parsedGLM.intercept, MathUtils.EPSILON);
			for (int i = 0; i < intercept.length; i++) {
				Assert.assertArrayEquals(w[i], parsedGLM.coefficients(i), MathUtils.EPSILON);
			}
		} catch (Exception e) {
			Assert.fail("Should not see exception: " + e.getMessage());
		}
	}
}
