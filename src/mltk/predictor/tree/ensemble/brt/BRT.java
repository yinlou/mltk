package mltk.predictor.tree.ensemble.brt;

import java.io.BufferedReader;
import java.io.PrintWriter;

import mltk.core.Instance;
import mltk.predictor.ProbabilisticClassifier;
import mltk.predictor.Regressor;
import mltk.predictor.tree.RegressionTree;
import mltk.util.StatUtils;
import mltk.util.VectorUtils;

/**
 * Class for boosted regression trees (BRTs).
 * 
 * @author Yin Lou
 * 
 */
public class BRT implements ProbabilisticClassifier, Regressor {

	protected BoostedRegressionTrees[] trees;

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
		trees = new BoostedRegressionTrees[k];
		for (int i = 0; i < trees.length; i++) {
			trees[i] = new BoostedRegressionTrees();
		}
	}

	/**
	 * Returns the tree list for class k.
	 * 
	 * @param k the class k.
	 * @return the tree list for class k.
	 */
	public BoostedRegressionTrees getRegressionTreeList(int k) {
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
		trees = new BoostedRegressionTrees[k];
		for (int i = 0; i < trees.length; i++) {
			int n = Integer.parseInt(in.readLine().split(": ")[1]);
			trees[i] = new BoostedRegressionTrees();
			for (int j = 0; j < n; j++) {
				in.readLine();
				RegressionTree rt = new RegressionTree();
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
		for (BoostedRegressionTrees rtList : trees) {
			out.println("Length: " + rtList.size());
			for (RegressionTree rt : rtList) {
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

	@Override
	public BRT copy() {
		BRT copy = new BRT(trees.length);
		for (int i = 0; i < trees.length; i++) {
			BoostedRegressionTrees brts = trees[i];
			for (RegressionTree rt : brts) {
				copy.trees[i].add(rt.copy());
			}
		}
		return copy;
	}

}
