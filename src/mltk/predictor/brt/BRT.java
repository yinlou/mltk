package mltk.predictor.brt;

import java.io.BufferedReader;
import java.io.PrintWriter;

import mltk.core.Instance;
import mltk.predictor.ProbabilisticClassifier;
import mltk.predictor.Regressor;
import mltk.predictor.tree.RegressionTree;
import mltk.predictor.tree.RegressionTreeList;
import mltk.util.StatUtils;
import mltk.util.VectorUtils;

/**
 * Class for boosted regression trees (BRTs).
 * 
 * @author Yin Lou
 *
 */
public class BRT implements ProbabilisticClassifier, Regressor {
	
	protected RegressionTreeList[] trees;
	
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
		trees = new RegressionTreeList[k];
		for (int i = 0; i < trees.length; i++) {
			trees[i] = new RegressionTreeList();
		}
	}
	
	/**
	 * Returns the tree list for class k.
	 * 
	 * @param k the class k.
	 * @return the tree list for class k.
	 */
	public RegressionTreeList getRegressionTreeList(int k) {
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
		trees = new RegressionTreeList[k];
		for (int i = 0; i < trees.length; i++) {
			trees[i] = new RegressionTreeList();
			int n = Integer.parseInt(in.readLine().split(": ")[1]);
			for (int j = 0; j < n; i++) {
				RegressionTree rt = new RegressionTree();
				rt.read(in);
				trees[i].add(rt);
				
				in.readLine();
			}
		}
	}

	@Override
	public void write(PrintWriter out) throws Exception {
		out.printf("[Predictor: %s]\n", this.getClass().getCanonicalName());
		out.println("K: " + trees.length);
		for (RegressionTreeList rtList : trees) {
			out.println("Length: " + rtList.size());
			for (RegressionTree rt : rtList) {
				rt.write(out);
			}
			out.println();
		}
	}

	@Override
	public double regress(Instance instance) {
		return regress(trees[0], instance);
	}

	@Override
	public double[] predictProbabilities(Instance instance) {
		double[] prob = new double[trees.length];
		double[] pred = new double[trees.length];
		for (int i = 0; i < trees.length; i++) {
			pred[i] = regress(trees[i], instance);
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
	
	protected double regress(RegressionTreeList trees, Instance instance) {
		double pred = 0;
		for (RegressionTree rt : trees) {
			pred += rt.regress(instance);
		}
		return pred;
	}

	@Override
	public BRT copy() {
		BRT copy = new BRT(trees.length);
		for (int i = 0; i < trees.length; i++) {
			RegressionTreeList rtList = trees[i];
			for (RegressionTree rt : rtList) {
				copy.trees[i].add(rt.copy());
			}
		}
		return copy;
	}

}
