package mltk.predictor.function;

import java.util.List;

import mltk.core.Attribute;
import mltk.core.BinnedAttribute;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.NominalAttribute;
import mltk.predictor.Learner;
import mltk.util.MathUtils;
import mltk.util.tuple.IntDoublePair;
import mltk.util.tuple.Pair;

/**
 * Class for cutting squares.
 * 
 * @author Yin Lou
 *
 */
public class SquareCutter extends Learner {

	static class Table {

		double[][][] resp;
		double[][][] count;

		Table(int n, int m) {
			resp = new double[n][m][4];
			count = new double[n][m][4];
		}
	}

	private int attIndex1;
	private int attIndex2;
	private boolean lineSearch;

	/**
	 * Constructor.
	 */
	public SquareCutter() {

	}

	/**
	 * Constructor.
	 * 
	 * @param lineSearch <code>true</code> if line search is performed in the end.
	 */
	public SquareCutter(boolean lineSearch) {
		this.lineSearch = lineSearch;
	}

	/**
	 * Sets the attribute indices.
	 * 
	 * @param attIndex1 the 1st index of attribute.
	 * @param attIndex2 the 2nd index of attribute.
	 */
	public void setAttIndices(int attIndex1, int attIndex2) {
		this.attIndex1 = attIndex1;
		this.attIndex2 = attIndex2;
	}

	public Function2D build(Instances instances) {
		// Note: Currently only support cutting on binned/nominal features
		List<Attribute> attributes = instances.getAttributes();
		int size1 = 0;
		Attribute f1 = attributes.get(attIndex1);
		if (f1.getType() == Attribute.Type.BINNED) {
			size1 = ((BinnedAttribute) f1).getNumBins();
		} else if (f1.getType() == Attribute.Type.NOMINAL) {
			size1 = ((NominalAttribute) f1).getCardinality();
		}
		int size2 = 0;
		Attribute f2 = attributes.get(attIndex2);
		if (f2.getType() == Attribute.Type.BINNED) {
			size2 = ((BinnedAttribute) f2).getNumBins();
		} else if (f2.getType() == Attribute.Type.NOMINAL) {
			size2 = ((NominalAttribute) f2).getCardinality();
		}
		if (size1 == 1 || size2 == 1) {
			// Not an interaction
			// Recommend: Use LineCutter to shape the non-trivial attribute
			return new Function2D(f1.getIndex(), f2.getIndex(), new double[] { Double.POSITIVE_INFINITY },
					new double[] { Double.POSITIVE_INFINITY }, new double[1][1]);
		}
		Histogram2D hist2d = new Histogram2D(size1, size2);
		for (Instance instance : instances) {
			int idx1 = (int) instance.getValue(f1);
			int idx2 = (int) instance.getValue(f2);
			hist2d.resp[idx1][idx2] += instance.getTarget() * instance.getWeight();
			hist2d.count[idx1][idx2] += instance.getWeight();
		}
		Pair<CHistogram, CHistogram> cHist = hist2d.computeCHistogram();
		Table table = new Table(size1, size2);
		computeTable(hist2d, cHist.v1, cHist.v2, table);

		double bestRSS = Double.POSITIVE_INFINITY;
		double[] predInt1 = new double[4];
		int bestV1 = -1;
		int[] bestV2s = new int[2];
		int[] v2s = new int[2];
		for (int v1 = 0; v1 < size1 - 1; v1++) {
			findCuts(table, v1, v2s);
			getPredictor(table, v1, v2s, predInt1);
			double rss = getRSS(table, v1, v2s, predInt1);
			if (rss < bestRSS) {
				bestRSS = rss;
				bestV1 = v1;
				bestV2s[0] = v2s[0];
				bestV2s[1] = v2s[1];
			}
		}

		boolean cutOnAttr2 = false;

		double[] predInt2 = new double[4];
		int[] bestV1s = new int[2];
		int bestV2 = -1;
		int[] v1s = new int[2];
		for (int v2 = 0; v2 < size2 - 1; v2++) {
			findCuts(table, v1s, v2);
			getPredictor(table, v1s, v2, predInt2);
			double rss = getRSS(table, v1s, v2, predInt2);
			if (rss < bestRSS) {
				bestRSS = rss;
				bestV2 = v2;
				bestV1s[0] = v1s[0];
				bestV1s[1] = v1s[1];
				cutOnAttr2 = true;
			}
		}

		if (cutOnAttr2) {
			// Root cut on attribute 2 is better
			getPredictor(table, bestV1s, bestV2, predInt2);
			if (lineSearch) {
				lineSearch(instances, f2.getIndex(), f1.getIndex(), bestV2, bestV1s[0], bestV1s[1], predInt2);
			}
			return getFunction2D(f1.getIndex(), f2.getIndex(), bestV1s, bestV2, predInt2);
		} else {
			// Root cut on attribute 1 is better
			getPredictor(table, bestV1, bestV2s, predInt1);
			if (lineSearch) {
				lineSearch(instances, f1.getIndex(), f2.getIndex(), bestV1, bestV2s[0], bestV2s[1], predInt1);
			}
			return getFunction2D(f1.getIndex(), f2.getIndex(), bestV1, bestV2s, predInt1);
		}
	}

	protected static void fillTable(Table table, int i, int j, CHistogram cHist1, CHistogram cHist2) {
		table.resp[i][j][1] = cHist1.sum[i] - table.resp[i][j][0];
		table.resp[i][j][2] = cHist2.sum[j] - table.resp[i][j][0];
		table.resp[i][j][3] = cHist1.sum[cHist1.size() - 1] - cHist1.sum[i] - table.resp[i][j][2];

		table.count[i][j][1] = cHist1.count[i] - table.count[i][j][0];
		table.count[i][j][2] = cHist2.count[j] - table.count[i][j][0];
		table.count[i][j][3] = cHist1.count[cHist1.size() - 1] - cHist1.count[i] - table.count[i][j][2];
	}

	protected static void computeTable(Histogram2D hist2d, CHistogram cHist1, CHistogram cHist2, Table table) {
		double sum = 0;
		double count = 0;
		for (int j = 0; j < hist2d.resp[0].length; j++) {
			sum += hist2d.resp[0][j];
			table.resp[0][j][0] = sum;
			count += hist2d.count[0][j];
			table.count[0][j][0] = count;
			fillTable(table, 0, j, cHist1, cHist2);
		}
		for (int i = 1; i < hist2d.resp.length; i++) {
			sum = count = 0;
			for (int j = 0; j < hist2d.resp[i].length; j++) {
				sum += hist2d.resp[i][j];
				table.resp[i][j][0] = table.resp[i - 1][j][0] + sum;
				count += hist2d.count[i][j];
				table.count[i][j][0] = table.count[i - 1][j][0] + count;
				fillTable(table, i, j, cHist1, cHist2);
			}
		}
	}

	protected static void getPredictor(Table table, int v1, int v2, double[] pred) {
		for (int i = 0; i < pred.length; i++) {
			pred[i] = table.count[v1][v2][i] == 0 ? 0 : table.resp[v1][v2][i] / table.count[v1][v2][i];
		}
	}

	protected static double getRSS(Table table, int v1, int v2, double[] pred) {
		double rss = 0;
		double t = 0;
		for (int i = 0; i < pred.length; i++) {
			t += pred[i] * pred[i] * table.count[v1][v2][i];
		}
		rss += t;
		t = 0;
		for (int i = 0; i < pred.length; i++) {
			t += pred[i] * table.resp[v1][v2][i];
		}
		rss -= 2 * t;
		return rss;
	}

	protected static void findCuts(Table table, int v1, int[] v2) {
		// Find upper cut
		double bestEval = Double.POSITIVE_INFINITY;
		for (int i = 0; i < table.resp[v1].length - 1; i++) {
			double sum1 = table.resp[v1][i][0];
			double sum2 = table.resp[v1][i][1];
			double weight1 = table.count[v1][i][0];
			double weight2 = table.count[v1][i][1];
			double eval = -sum1 * sum1 / weight1 - sum2 * sum2 / weight2;
			if (eval < bestEval) {
				bestEval = eval;
				v2[0] = i;
			}
		}

		// Find lower cut
		bestEval = Double.POSITIVE_INFINITY;
		for (int i = 0; i < table.resp[v1].length - 1; i++) {
			double sum1 = table.resp[v1][i][2];
			double sum2 = table.resp[v1][i][3];
			double weight1 = table.count[v1][i][2];
			double weight2 = table.count[v1][i][3];
			double eval = -sum1 * sum1 / weight1 - sum2 * sum2 / weight2;
			if (eval < bestEval) {
				bestEval = eval;
				v2[1] = i;
			}
		}
	}

	protected static void findCuts(Table table, int[] v1, int v2) {
		// Find left cut
		double bestEval = Double.POSITIVE_INFINITY;
		for (int i = 0; i < table.resp.length - 1; i++) {
			double sum1 = table.resp[i][v2][0];
			double sum2 = table.resp[i][v2][2];
			double weight1 = table.count[i][v2][0];
			double weight2 = table.count[i][v2][2];
			double eval = -sum1 * sum1 / weight1 - sum2 * sum2 / weight2;
			if (eval < bestEval) {
				bestEval = eval;
				v1[0] = i;
			}
		}

		// Find right cut
		bestEval = Double.POSITIVE_INFINITY;
		for (int i = 0; i < table.resp.length - 1; i++) {
			double sum1 = table.resp[i][v2][1];
			double sum2 = table.resp[i][v2][3];
			double weight1 = table.count[i][v2][1];
			double weight2 = table.count[i][v2][3];
			double eval = -sum1 * sum1 / weight1 - sum2 * sum2 / weight2;
			if (eval < bestEval) {
				bestEval = eval;
				v1[1] = i;
			}
		}
	}

	protected static void findCut(CHistogram cHist, IntDoublePair cut) {
		cut.v2 = Double.POSITIVE_INFINITY;
		for (int i = 0; i < cHist.size() - 1; i++) {
			double sum1 = cHist.sum[i];
			double sum2 = cHist.sum[cHist.size() - 1] - sum1;
			double weight1 = cHist.count[i];
			double weight2 = cHist.count[cHist.size() - 1] - weight1;
			double eval = -sum1 * sum1 * weight1 - sum2 * sum2 * weight2;
			if (eval < cut.v2) {
				cut.v2 = eval;
				cut.v1 = i;
			}
		}
	}

	protected static void getPredictor(Table table, int v1, int[] v2, double[] pred) {
		int v21 = v2[0];
		int v22 = v2[1];
		pred[0] = table.count[v1][v21][0] == 0 ? 0 : table.resp[v1][v21][0] / table.count[v1][v21][0];
		pred[1] = table.count[v1][v21][1] == 0 ? 0 : table.resp[v1][v21][1] / table.count[v1][v21][1];
		pred[2] = table.count[v1][v22][2] == 0 ? 0 : table.resp[v1][v22][2] / table.count[v1][v22][2];
		pred[3] = table.count[v1][v22][3] == 0 ? 0 : table.resp[v1][v22][3] / table.count[v1][v22][3];
	}

	protected static void getPredictor(Table table, int[] v1, int v2, double[] pred) {
		int v11 = v1[0];
		int v12 = v1[1];
		pred[0] = table.count[v11][v2][0] == 0 ? 0 : table.resp[v11][v2][0] / table.count[v11][v2][0];
		pred[1] = table.count[v11][v2][2] == 0 ? 0 : table.resp[v11][v2][2] / table.count[v11][v2][2];
		pred[2] = table.count[v12][v2][1] == 0 ? 0 : table.resp[v12][v2][1] / table.count[v12][v2][1];
		pred[3] = table.count[v12][v2][3] == 0 ? 0 : table.resp[v12][v2][3] / table.count[v12][v2][3];
	}

	protected static double getRSS(Table table, int v1, int v2[], double[] pred) {
		int v21 = v2[0];
		int v22 = v2[1];
		double rss = 0;
		rss += pred[0] * pred[0] * table.count[v1][v21][0];
		rss += pred[1] * pred[1] * table.count[v1][v21][1];
		rss += pred[2] * pred[2] * table.count[v1][v22][2];
		rss += pred[3] * pred[3] * table.count[v1][v22][3];

		double t = 0;
		t += pred[0] * table.resp[v1][v21][0];
		t += pred[1] * table.resp[v1][v21][1];
		t += pred[2] * table.resp[v1][v22][2];
		t += pred[3] * table.resp[v1][v22][3];
		rss -= 2 * t;
		return rss;
	}

	protected static double getRSS(Table table, int[] v1, int v2, double[] pred) {
		int v11 = v1[0];
		int v12 = v1[1];
		double rss = 0;
		rss += pred[0] * pred[0] * table.count[v11][v2][0];
		rss += pred[1] * pred[1] * table.count[v11][v2][2];
		rss += pred[2] * pred[2] * table.count[v12][v2][1];
		rss += pred[3] * pred[3] * table.count[v12][v2][3];

		double t = 0;
		t += pred[0] * table.resp[v11][v2][0];
		t += pred[1] * table.resp[v11][v2][2];
		t += pred[2] * table.resp[v12][v2][1];
		t += pred[3] * table.resp[v12][v2][3];
		rss -= 2 * t;

		return rss;
	}

	protected static Function2D getFunction2D(int attIndex1, int attIndex2, int v1, int[] v2, double[] predInt) {
		double[] splits1 = new double[] { v1, Double.POSITIVE_INFINITY };
		double[] splits2 = null;
		double[][] predictions = null;
		if (v2[0] < v2[1]) {
			splits2 = new double[] { v2[0], v2[1], Double.POSITIVE_INFINITY };
			predictions = new double[][] { 
					{ predInt[0], predInt[1], predInt[1] },
					{ predInt[2], predInt[2], predInt[3] } 
			};
		} else if (v2[0] > v2[1]) {
			splits2 = new double[] { v2[1], v2[0], Double.POSITIVE_INFINITY };
			predictions = new double[][] { 
					{ predInt[0], predInt[0], predInt[1] },
					{ predInt[2], predInt[3], predInt[3] } 
			};
		} else {
			splits2 = new double[] { v2[0], Double.POSITIVE_INFINITY };
			predictions = new double[][] { 
					{ predInt[0], predInt[1] }, 
					{ predInt[2], predInt[3] }
			};
		}
		return new Function2D(attIndex1, attIndex2, splits1, splits2, predictions);
	}

	protected static Function2D getFunction2D(int attIndex1, int attIndex2, int[] v1, int v2, double[] predInt) {
		double[] splits1 = null;
		double[] splits2 = new double[] { v2, Double.POSITIVE_INFINITY };
		double[][] predictions = null;
		if (v1[0] < v1[1]) {
			splits1 = new double[] { v1[0], v1[1], Double.POSITIVE_INFINITY };
			predictions = new double[][] { 
					{ predInt[0], predInt[2] }, 
					{ predInt[1], predInt[2] },
					{ predInt[1], predInt[3] }
			};
		} else if (v1[0] > v1[1]) {
			splits1 = new double[] { v1[1], v1[0], Double.POSITIVE_INFINITY };
			predictions = new double[][] { 
					{ predInt[0], predInt[2] }, 
					{ predInt[0], predInt[3] },
					{ predInt[1], predInt[3] } 
			};
		} else {
			splits1 = new double[] { v1[0], Double.POSITIVE_INFINITY };
			predictions = new double[][] { 
					{ predInt[0], predInt[2] }, 
					{ predInt[1], predInt[3] } 
			};
		}
		return new Function2D(attIndex1, attIndex2, splits1, splits2, predictions);
	}

	protected static void lineSearch(Instances instances, int attIndex1, int attIndex2, int c1, int c21, int c22,
			double[] predictions) {
		double[] numerator = new double[4];
		double[] denominator = new double[4];
		for (Instance instance : instances) {
			int v1 = (int) instance.getValue(attIndex1);
			int v2 = (int) instance.getValue(attIndex2);
			double target = instance.getTarget();
			double t = Math.abs(target);
			if (v1 <= c1) {
				if (v2 <= c21) {
					numerator[0] += target * instance.getWeight();
					denominator[0] += t * (1 - t) * instance.getWeight();
				} else {
					numerator[1] += target * instance.getWeight();
					denominator[1] += t * (1 - t) * instance.getWeight();
				}
			} else {
				if (v2 <= c22) {
					numerator[2] += target * instance.getWeight();
					denominator[2] += t * (1 - t) * instance.getWeight();
				} else {
					numerator[3] += target * instance.getWeight();
					denominator[3] += t * (1 - t) * instance.getWeight();
				}
			}
		}
		for (int i = 0; i < numerator.length; i++) {
			predictions[i] = Math.abs(denominator[i]) < MathUtils.EPSILON ? 0.0 : numerator[i] / denominator[i];
		}
	}

}
