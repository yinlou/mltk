package mltk.predictor.tree.ensemble.brt;

import java.io.BufferedReader;
import java.io.PrintWriter;

import mltk.core.Instance;
import mltk.predictor.ProbabilisticClassifier;
import mltk.predictor.Regressor;
import mltk.predictor.tree.RTree;
import mltk.predictor.tree.ensemble.BoostedRTrees;
import mltk.util.MathUtils;
import mltk.util.StatUtils;
import mltk.util.VectorUtils;

/**
 * Class for boosted regression trees (BRTs).
 * 
 * @author Yin Lou
 * 
 */
public class BRT implements ProbabilisticClassifier, Regressor {

	protected BoostedRTrees[] trees;

	/**
	 * Constructor.
	 */
	public BRT() {

	}

	/**
	 * Constructor.
	 * 
	 * @param k the number of classes.
	 */
	public BRT(int k) {
		trees = new BoostedRTrees[k];
		for (int i = 0; i < trees.length; i++) {
			trees[i] = new BoostedRTrees();
		}
	}

	/**
	 * Returns the tree list for class k.
	 * 
	 * @param k the class k.
	 * @return the tree list for class k.
	 */
	public BoostedRTrees getRegressionTreeList(int k) {
		return trees[k];
	}

	@Override
	public int classify(Instance instance) {
		double[] prob = predictProbabilities(instance);
		return StatUtils.indexOfMax(prob);
	}

	@Override
	public void read(BufferedReader in) throws Exception {
		int k = Integer.parseInt(in.readLine().split(": ")[1]);
		trees = new BoostedRTrees[k];
		for (int i = 0; i < trees.length; i++) {
			int n = Integer.parseInt(in.readLine().split(": ")[1]);
			trees[i] = new BoostedRTrees();
			for (int j = 0; j < n; j++) {
				String line = in.readLine();
				String predictorName = line.substring(1, line.length() - 1).split(": ")[1];
				Class<?> clazz = Class.forName(predictorName);
				RTree rt = (RTree) clazz.newInstance();
				rt.read(in);
				trees[i].add(rt);

				in.readLine();
			}
			in.readLine();
		}
	}

	@Override
	public void write(PrintWriter out) throws Exception {
		out.printf("[Predictor: %s]\n", this.getClass().getCanonicalName());
		out.println("K: " + trees.length);
		for (BoostedRTrees rtList : trees) {
			out.println("Length: " + rtList.size());
			for (RTree rt : rtList) {
				rt.write(out);
				out.println();
			}
			out.println();
		}
	}

	@Override
	public double regress(Instance instance) {
		return trees[0].regress(instance);
	}

	@Override
	public double[] predictProbabilities(Instance instance) {
		if (trees.length == 1) {
			double[] prob = new double[2];
			double pred = regress(instance);
			prob[1] = MathUtils.sigmoid(pred);
			prob[0] = 1 - prob[1];
			return prob;
		} else {
			double[] prob = new double[trees.length];
			double[] pred = new double[trees.length];
			for (int i = 0; i < trees.length; i++) {
				pred[i] = trees[i].regress(instance);
			}
			double max = StatUtils.max(pred);
			double sum = 0;
			for (int i = 0; i < prob.length; i++) {
				prob[i] = Math.exp(pred[i] - max);
				sum += prob[i];
			}
			VectorUtils.divide(prob, sum);
			return prob;
		}
	}

	@Override
	public BRT copy() {
		BRT copy = new BRT(trees.length);
		for (int i = 0; i < trees.length; i++) {
			BoostedRTrees brts = trees[i];
			for (RTree rt : brts) {
				copy.trees[i].add(rt.copy());
			}
		}
		return copy;
	}

}
