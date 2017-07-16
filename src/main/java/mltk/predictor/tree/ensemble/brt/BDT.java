package mltk.predictor.tree.ensemble.brt;

import java.io.BufferedReader;
import java.io.PrintWriter;

import mltk.core.Instance;
import mltk.predictor.ProbabilisticClassifier;
import mltk.predictor.Regressor;
import mltk.predictor.tree.ensemble.BoostedDTables;
import mltk.util.MathUtils;
import mltk.util.StatUtils;
import mltk.util.VectorUtils;

/**
 * Class for boosted decision tables (BDTs).
 * 
 * <p>
 * Reference:<br>
 * Y. Lou and M. Obukhov. BDT: Boosting Decision Tables for High Accuracy and Scoring Efficiency. In <i>Proceedings of the
 * 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)</i>, Halifax, Nova Scotia, Canada, 2017.
 * </p>
 * 
 * @author Yin Lou
 * 
 */
public class BDT implements ProbabilisticClassifier, Regressor {

	protected BoostedDTables[] tables;
	
	/**
	 * Constructs a BDT from a BRT object.
	 * 
	 * @param brt the BRT object.
	 * @return a BDT object.
	 */
	public static BDT constructBDT(BRT brt) {
		int k = brt.trees.length;
		BDT bdt = new BDT();
		bdt.tables = new BoostedDTables[k];
		for (int i = 0; i < bdt.tables.length; i++) {
			bdt.tables[i] = new BoostedDTables(brt.trees[i]);
		}
		
		return bdt;
	}

	/**
	 * Constructor.
	 */
	public BDT() {

	}

	/**
	 * Constructor.
	 * 
	 * @param k the number of classes.
	 */
	public BDT(int k) {
		tables = new BoostedDTables[k];
		for (int i = 0; i < tables.length; i++) {
			tables[i] = new BoostedDTables();
		}
	}

	/**
	 * Returns the table list for class k.
	 * 
	 * @param k the class k.
	 * @return the table list for class k.
	 */
	public BoostedDTables getDecisionTreeList(int k) {
		return tables[k];
	}

	@Override
	public int classify(Instance instance) {
		double[] prob = predictProbabilities(instance);
		return StatUtils.indexOfMax(prob);
	}

	@Override
	public void read(BufferedReader in) throws Exception {
		int k = Integer.parseInt(in.readLine().split(": ")[1]);
		tables = new BoostedDTables[k];
		for (int i = 0; i < tables.length; i++) {
			in.readLine();
			tables[i] = new BoostedDTables();
			tables[i].read(in);
			
			in.readLine();
		}
	}

	@Override
	public void write(PrintWriter out) throws Exception {
		out.printf("[Predictor: %s]\n", this.getClass().getCanonicalName());
		out.println("K: " + tables.length);
		for (BoostedDTables dtList : tables) {
			dtList.write(out);
			out.println();
		}
	}

	@Override
	public double regress(Instance instance) {
		return tables[0].regress(instance);
	}

	@Override
	public double[] predictProbabilities(Instance instance) {
		if (tables.length == 1) {
			double[] prob = new double[2];
			double pred = regress(instance);
			prob[1] = MathUtils.sigmoid(pred);
			prob[0] = 1 - prob[1];
			return prob;
		} else {
			double[] prob = new double[tables.length];
			double[] pred = new double[tables.length];
			for (int i = 0; i < tables.length; i++) {
				pred[i] = tables[i].regress(instance);
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
	public BDT copy() {
		BDT copy = new BDT(tables.length);
		for (int i = 0; i < tables.length; i++) {
			copy.tables[i] = tables[i].copy();
		}
		return copy;
	}

}
