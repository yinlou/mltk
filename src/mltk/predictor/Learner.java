package mltk.predictor;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import mltk.core.Attribute;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.SparseVector;
import mltk.predictor.evaluation.Error;
import mltk.predictor.evaluation.Metric;
import mltk.predictor.evaluation.RMSE;
import mltk.util.MathUtils;
import mltk.util.StatUtils;
import mltk.util.VectorUtils;
import mltk.util.tuple.IntDoublePair;

/**
 * Class for learners.
 * 
 * @author Yin Lou
 * 
 */
public abstract class Learner {
	
	protected boolean verbose;

	/**
	 * Returns <code>true</code> if we output something during the training.
	 * 
	 * @return <code>true</code> if we output something during the training.
	 */
	public boolean isVerbose() {
		return verbose;
	}

	/**
	 * Sets whether we output something during the training.
	 * 
	 * @param verbose the switch if we output things during training.
	 */
	public void setVerbose(boolean verbose) {
		this.verbose = verbose;
	}
	
	/**
	 * Enumeration of learning tasks.
	 * 
	 */
	public enum Task {

		/**
		 * Classification task.
		 */
		CLASSIFICATION("classification"),
		/**
		 * Regression task.
		 */
		REGRESSION("regression");

		String task;

		Task(String task) {
			this.task = task;
		}

		/**
		 * Returns the string representation of learning tasks.
		 */
		public String toString() {
			return task;
		}

		/**
		 * Parses an enumeration from a string.
		 * 
		 * @param task the string.
		 * @return a parsed task.
		 */
		public static Task getEnum(String task) {
			for (Task re : Task.values()) {
				if (re.task.startsWith(task)) {
					return re;
				}
			}
			throw new IllegalArgumentException("Invalid Task value: " + task);
		}
		
		public Metric getDefaultMetric() {
			Metric metric = null;
			switch (this) {
				case CLASSIFICATION:
					metric = new Error();
					break;
				case REGRESSION:
					metric = new RMSE();
					break;
				default:
					break;
			}
			return metric;
		}

	}

	/**
	 * Builds a predictor from training set.
	 * 
	 * @param instances the training set.
	 * @return a predictior.
	 */
	public abstract Predictor build(Instances instances);

	/**
	 * Returns <code>true</code> if the instances are treated as sparse.
	 * 
	 * @param instances the instances to test.
	 * @return <code>true</code> if the instances are treated as sparse.
	 */
	protected boolean isSparse(Instances instances) {
		int numSparseInstances = 0;
		for (Instance instance : instances) {
			if (instance.isSparse()) {
				numSparseInstances++;
			}
		}
		return numSparseInstances > instances.size() / 2;
	}

	/**
	 * Returns the column-oriented format of sparse dataset. This method automatically removes attributes with
	 * close-to-zero variance.
	 * 
	 * @param instances the instances.
	 * @param normalize <code>true</code> if all the columns are normalized.
	 * @return the column-oriented format of sparse dataset.
	 */
	protected SparseDataset getSparseDataset(Instances instances, boolean normalize) {
		List<Attribute> attributes = instances.getAttributes();
		int maxAttrId = attributes.get(attributes.size() - 1).getIndex();
		boolean[] included = new boolean[maxAttrId + 1];
		for (Attribute attribute : attributes) {
			included[attribute.getIndex()] = true;
		}

		final int n = instances.size();
		Map<Integer, List<IntDoublePair>> map = new TreeMap<>();
		double[] y = new double[n];

		for (int i = 0; i < instances.size(); i++) {
			Instance instance = instances.get(i);
			SparseVector vector = (SparseVector) instance.getVector();
			int[] indices = vector.getIndices();
			double[] values = vector.getValues();
			for (int j = 0; j < indices.length; j++) {
				if (included[indices[j]]) {
					if (!map.containsKey(indices[j])) {
						map.put(indices[j], new ArrayList<IntDoublePair>());
					}
					List<IntDoublePair> list = map.get(indices[j]);
					list.add(new IntDoublePair(i, values[j]));
				}
			}
			y[i] = instance.getTarget();
		}

		List<Integer> attrsList = new ArrayList<>(map.size());
		List<int[]> indicesList = new ArrayList<>(map.size());
		List<double[]> valuesList = new ArrayList<>(map.size());
		List<Double> stdList = new ArrayList<>(map.size());
		List<Double> cList = null;
		if (normalize) {
			cList = new ArrayList<>();
		}
		double factor = Math.sqrt(n);
		for (Map.Entry<Integer, List<IntDoublePair>> entry : map.entrySet()) {
			Integer attr = entry.getKey();
			List<IntDoublePair> list = entry.getValue();
			int[] indices = new int[list.size()];
			double[] values = new double[list.size()];
			for (int i = 0; i < list.size(); i++) {
				IntDoublePair pair = list.get(i);
				indices[i] = pair.v1;
				values[i] = pair.v2;
			}
			double std = StatUtils.std(values, n);
			if (std > MathUtils.EPSILON) {
				attrsList.add(attr);
				indicesList.add(indices);
				valuesList.add(values);
				stdList.add(std);
				if (normalize) {
					// Normalize the data
					double c = factor / std;
					VectorUtils.multiply(values, c);
					cList.add(c);
				}
			}
		}

		final int p = attrsList.size();
		int[] attrs = new int[p];
		int[][] indices = new int[p][];
		double[][] values = new double[p][];
		for (int j = 0; j < p; j++) {
			attrs[j] = attrsList.get(j);
			indices[j] = indicesList.get(j);
			values[j] = valuesList.get(j);
		}
		
		double[] std = new double[stdList.size()];
		for (int j = 0; j < std.length; j++) {
			std[j] = stdList.get(j);
		}
		double[] c = null;
		if (cList != null) {
			c = new double[cList.size()];
			for (int j = 0; j < c.length; j++) {
				c[j] = cList.get(j);
			}
		}

		return new SparseDataset(attrs, indices, values, y, std, c);
	}

	/**
	 * Returns the column-oriented format of dense dataset. This method automatically removes attributes with
	 * close-to-zero variance.
	 * 
	 * @param instances the instances.
	 * @param normalize <code>true</code> if all the columns are normalized.
	 * @return the column-oriented format of dense dataset.
	 */
	protected DenseDataset getDenseDataset(Instances instances, boolean normalize) {
		List<Attribute> attributes = instances.getAttributes();
		final int p = instances.dimension();
		final int n = instances.size();

		// Convert to column oriented format
		List<double[]> xList = new ArrayList<>(p);
		double[] y = new double[n];
		for (int i = 0; i < n; i++) {
			y[i] = instances.get(i).getTarget();
		}

		List<Integer> attrsList = new ArrayList<>(p);
		List<Double> stdList = new ArrayList<>(p);
		List<Double> cList = null;
		if (normalize) {
			cList = new ArrayList<>();
		}
		double factor = Math.sqrt(n);
		for (int j = 0; j < p; j++) {
			int attIndex = attributes.get(j).getIndex();
			double[] x = new double[n];
			for (int i = 0; i < n; i++) {
				x[i] = instances.get(i).getValue(attIndex);
			}
			double std = StatUtils.std(x);
			if (std > MathUtils.EPSILON) {
				attrsList.add(attIndex);
				xList.add(x);
				stdList.add(std);
				if (normalize) {
					// Normalize the data
					double c = factor / std;
					VectorUtils.multiply(x, c);
					cList.add(c);
				}
			}
		}

		int[] attrs = new int[attrsList.size()];
		double[][] x = new double[attrsList.size()][];
		for (int j = 0; j < attrs.length; j++) {
			attrs[j] = attrsList.get(j);
			x[j] = xList.get(j);
		}
		
		double[] std = new double[stdList.size()];
		for (int j = 0; j < std.length; j++) {
			std[j] = stdList.get(j);
		}
		double[] c = null;
		if (cList != null) {
			c = new double[cList.size()];
			for (int j = 0; j < c.length; j++) {
				c[j] = cList.get(j);
			}
		}

		return new DenseDataset(attrs, x, y, std, c);
	}

	/**
	 * Class for sparse dataset.
	 *
	 */
	protected class SparseDataset {

		public int[] attrs;
		public int[][] indices;
		public double[][] values;
		public double[] y;
		public double[] stdList;
		public double[] cList;

		SparseDataset(int[] attrs, int[][] indices, double[][] values, double[] y, double[] stdList,
				double[] cList) {
			this.attrs = attrs;
			this.indices = indices;
			this.values = values;
			this.y = y;
			this.stdList = stdList;
			this.cList = cList;
		}

	}

	/**
	 * Class for dense dataset.
	 *
	 */
	protected class DenseDataset {

		public int[] attrs;
		public double[][] x;
		public double[] y;
		public double[] stdList;
		public double[] cList;

		DenseDataset(int[] attrs, double[][] x, double[] y, double[] stdList, double[] cList) {
			this.attrs = attrs;
			this.x = x;
			this.y = y;
			this.stdList = stdList;
			this.cList = cList;
		}

	}

}
