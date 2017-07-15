package mltk.predictor.tree.ensemble;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.ArrayList;

import mltk.core.Instance;
import mltk.predictor.tree.RTree;

/**
 * Class for boosted regression trees. This is a base class for BRT.
 * 
 * @author Yin Lou
 * 
 */
public class BoostedRTrees extends RTreeList {

	/**
	 * Constructor.
	 */
	public BoostedRTrees() {
		super();
	}

	/**
	 * Constructs a regression tree list of length n. By default each tree is null.
	 * 
	 * @param n the length.
	 */
	public BoostedRTrees(int n) {
		trees = new ArrayList<>(n);
		for (int i = 0; i < n; i++) {
			trees.add(null);
		}
	}

	/**
	 * Regresses an instance.
	 * 
	 * @param instance the instance.
	 * @return the regressed
	 */
	public double regress(Instance instance) {
		double pred = 0;
		for (RTree rt : trees) {
			pred += rt.regress(instance);
		}
		return pred;
	}

	@Override
	public BoostedRTrees copy() {
		BoostedRTrees copy = new BoostedRTrees();
		for (RTree rt : trees) {
			copy.trees.add(rt.copy());
		}
		return copy;
	}

	public void read(BufferedReader in) throws Exception {
		int n = Integer.parseInt(in.readLine().split(": ")[1]);
		for (int j = 0; j < n; j++) {
			String line = in.readLine();
			String predictorName = line.substring(1, line.length() - 1).split(": ")[1];
			Class<?> clazz = Class.forName(predictorName);
			RTree rt = (RTree) clazz.newInstance();
			rt.read(in);
			this.trees.add(rt);

			in.readLine();
		}
	}

	public void write(PrintWriter out) throws Exception {
		out.printf("[Predictor: %s]\n", this.getClass().getCanonicalName());
		out.println("Length: " + trees.size());
		for (RTree rt : trees) {
			rt.write(out);
			out.println();
		}
	}

}
