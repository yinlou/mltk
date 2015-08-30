package mltk.predictor.tree.ensemble.rf;

import java.io.BufferedReader;
import java.io.PrintWriter;

import mltk.core.Instance;
import mltk.predictor.Regressor;
import mltk.predictor.tree.RegressionTree;
import mltk.predictor.tree.RegressionTreeList;

/**
 * Class for random forests.
 * 
 * @author Yin Lou
 *
 */
public class RandomForest implements Regressor {
	
	protected RegressionTreeList rtList;
	
	/**
	 * Constructor.
	 */
	public RandomForest() {
		rtList = new RegressionTreeList();
	}
	
	/**
	 * Constructor.
	 * 
	 * @param capacity the capacity of this random forest.
	 */
	public RandomForest(int capacity) {
		rtList = new RegressionTreeList(capacity);
	}

	@Override
	public void read(BufferedReader in) throws Exception {
		int capacity = Integer.parseInt(in.readLine().split(": ")[1]);
		rtList = new RegressionTreeList(capacity);
		in.readLine();
		for (int i = 0; i < capacity; i++) {
			in.readLine();
			RegressionTree rt = new RegressionTree();
			rt.read(in);
			rtList.add(rt);
			
			in.readLine();
		}
	}

	@Override
	public void write(PrintWriter out) throws Exception {
		out.printf("[Predictor: %s]\n", this.getClass().getCanonicalName());
		out.println("Ensemble: " + size());
		out.println();
		for (RegressionTree rt : rtList) {
			rt.write(out);
			out.println();
		}
	}

	@Override
	public RandomForest copy() {
		RandomForest copy = new RandomForest(size());
		for (RegressionTree rt : rtList) {
			copy.add(rt.copy());
		}
		return copy;
	}

	@Override
	public double regress(Instance instance) {
		if (size() == 0) {
			return 0.0;
		} else {
			double prediction = 0.0;
			for (RegressionTree rt : rtList) {
				prediction += rt.regress(instance);
			}
			return prediction / size();
		}
	}
	
	/**
	 * Adds a regression tree to the ensemble.
	 * 
	 * @param rt the regression tree.
	 */
	public void add(RegressionTree rt) {
		rtList.add(rt);
	}
	
	/**
	 * Returns the size of this random forest.
	 * 
	 * @return the size of this random forest.
	 */
	public int size() {
		return rtList.size();
	}

}
