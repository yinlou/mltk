package mltk.predictor.tree.ensemble.ag;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import mltk.core.Instance;
import mltk.predictor.Regressor;
import mltk.predictor.tree.RegressionTree;

/**
 * Class for Additive Groves.
 * 
 * @author Yin Lou
 * 
 */
public class AdditiveGroves implements Regressor {

	protected List<RegressionTree[]> groves;

	/**
	 * Constructor.
	 */
	public AdditiveGroves() {
		groves = new ArrayList<>();
	}

	@Override
	public void read(BufferedReader in) throws Exception {
		int bn = Integer.parseInt(in.readLine().split(": ")[1]);
		groves = new ArrayList<>();
		for (int i = 0; i < bn; i++) {
			int tn = Integer.parseInt(in.readLine().split(": ")[1]);
			RegressionTree[] grove = new RegressionTree[tn];
			for (int j = 0; j < tn; j++) {
				in.readLine();
				RegressionTree rt = new RegressionTree();
				rt.read(in);
				grove[i] = rt;

				in.readLine();
			}
			groves.add(grove);
			in.readLine();
		}
	}

	@Override
	public void write(PrintWriter out) throws Exception {
		out.printf("[Predictor: %s]\n", this.getClass().getCanonicalName());
		out.println("Bagging: " + groves.size());
		for (RegressionTree[] grove : groves) {
			out.println("Size: " + grove.length);
			for (RegressionTree rt : grove) {
				rt.write(out);
				out.println();
			}
			out.println();
		}
	}

	@Override
	public double regress(Instance instance) {
		if (groves.size() == 0) {
			return 0;
		}
		double pred = 0;
		for (RegressionTree[] grove : groves) {
			for (RegressionTree rt : grove) {
				pred += rt.regress(instance);
			}
		}
		return pred / groves.size();
	}

	@Override
	public AdditiveGroves copy() {
		AdditiveGroves copy = new AdditiveGroves();
		for (RegressionTree[] grove : groves) {
			RegressionTree[] copyGrove = new RegressionTree[grove.length];
			for (int i = 0; i < copyGrove.length; i++) {
				copyGrove[i] = grove[i].copy();
			}
			copy.groves.add(copyGrove);
		}
		return copy;
	}

}
