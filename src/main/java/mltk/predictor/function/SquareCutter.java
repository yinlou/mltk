package mltk.predictor.function;

import java.util.List;

import mltk.core.Attribute;
import mltk.core.BinnedAttribute;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.NominalAttribute;
import mltk.predictor.Learner;
import mltk.util.MathUtils;
import mltk.util.OptimUtils;
import mltk.util.tuple.Pair;

/**
 * Class for cutting squares.
 * 
 * @author Yin Lou
 *
 */
public class SquareCutter extends Learner {

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
	 * @param lineSearch {@code true} if line search is performed in the end.
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
		Histogram2D hist2d = new Histogram2D(size1, size2);
		Histogram2D.computeHistogram2D(instances, f1.getIndex(), f2.getIndex(), hist2d);
		Pair<CHistogram, CHistogram> cHist = hist2d.computeCHistogram();
		
		if ((size1 == 1 && !cHist.v1.hasMissingValue()) || (size2 == 1 && !cHist.v2.hasMissingValue())) {
			// Not an interaction
			// Recommend: Use LineCutter to shape the non-trivial attribute
			return new Function2D(f1.getIndex(), f2.getIndex(), new double[] { Double.POSITIVE_INFINITY },
					new double[] { Double.POSITIVE_INFINITY }, new double[1][1]);
		}
		
		Histogram2D.Table table = Histogram2D.computeTable(hist2d, cHist.v1, cHist.v2);

		double bestRSS = Double.POSITIVE_INFINITY;
		double[] predInt1 = new double[9];
		int bestV1 = -1;
		int[] bestV2s = new int[3];
		int[] v2s = new int[3];
		v2s[2] = -1;
		for (int v1 = 0; v1 < size1 - 1; v1++) {
			findCuts(table, v1, v2s, cHist.v1.hasMissingValue());
			getPredictor(table, v1, v2s, predInt1);
			double rss = getRSS(table, v1, v2s, predInt1);
			if (rss < bestRSS) {
				bestRSS = rss;
				bestV1 = v1;
				bestV2s[0] = v2s[0];
				bestV2s[1] = v2s[1];
				bestV2s[2] = v2s[2];
			}
		}

		boolean cutOnAttr2 = false;

		double[] predInt2 = new double[9];
		int[] bestV1s = new int[3];
		int bestV2 = -1;
		int[] v1s = new int[3];
		v1s[2] = -1;
		for (int v2 = 0; v2 < size2 - 1; v2++) {
			findCuts(table, v1s, v2, cHist.v2.hasMissingValue());
			getPredictor(table, v1s, v2, predInt2);
			double rss = getRSS(table, v1s, v2, predInt2);
			if (rss < bestRSS) {
				bestRSS = rss;
				bestV2 = v2;
				bestV1s[0] = v1s[0];
				bestV1s[1] = v1s[1];
				bestV1s[2] = v1s[2];
				cutOnAttr2 = true;
			}
		}

		if (cutOnAttr2) {
			// Root cut on attribute 2 is better
			getPredictor(table, bestV1s, bestV2, predInt2);
			if (lineSearch) {
				lineSearch(instances, f2.getIndex(), f1.getIndex(), bestV2, bestV1s[0], bestV1s[1], bestV1s[2], predInt2);
			}
			return getFunction2D(f1.getIndex(), f2.getIndex(), bestV1s, bestV2, predInt2);
		} else {
			// Root cut on attribute 1 is better
			getPredictor(table, bestV1, bestV2s, predInt1);
			if (lineSearch) {
				lineSearch(instances, f1.getIndex(), f2.getIndex(), bestV1, bestV2s[0], bestV2s[1], bestV2s[2], predInt1);
			}
			return getFunction2D(f1.getIndex(), f2.getIndex(), bestV1, bestV2s, predInt1);
		}
	}

	protected static void findCuts(Histogram2D.Table table, int v1, int[] v2, boolean hasMissingValue) {
		// Find upper cut
		double bestEval = Double.POSITIVE_INFINITY;
		for (int i = 0; i < table.resp[v1].length - 1; i++) {
			double[] resp = table.resp[v1][i];
			double[] count = table.count[v1][i];
			double sum1 = resp[0];
			double sum2 = resp[1];
			double weight1 = count[0];
			double weight2 = count[1];
			double eval1 = OptimUtils.getGain(sum1, weight1);
			double eval2 = OptimUtils.getGain(sum2, weight2);
			double eval = -(eval1 + eval2);
			if (eval < bestEval) {
				bestEval = eval;
				v2[0] = i;
			}
		}

		// Find lower cut
		bestEval = Double.POSITIVE_INFINITY;
		for (int i = 0; i < table.resp[v1].length - 1; i++) {
			double[] resp = table.resp[v1][i];
			double[] count = table.count[v1][i];
			double sum1 = resp[2];
			double sum2 = resp[3];
			double weight1 = count[2];
			double weight2 = count[3];
			double eval1 = OptimUtils.getGain(sum1, weight1);
			double eval2 = OptimUtils.getGain(sum2, weight2);
			double eval = -(eval1 + eval2);
			if (eval < bestEval) {
				bestEval = eval;
				v2[1] = i;
			}
		}
		
		if (hasMissingValue) {
			// Find cut on missing value
			bestEval = Double.POSITIVE_INFINITY;
			for (int i = 0; i < table.respOnMV1.length; i++) {
				double[] respOnMV1 = table.respOnMV1[i];
				double[] countOnMV1 = table.countOnMV1[i];
				double sum1 = respOnMV1[0];
				double sum2 = respOnMV1[1];
				double weight1 = countOnMV1[0];
				double weight2 = countOnMV1[1];
				double eval1 = OptimUtils.getGain(sum1, weight1);
				double eval2 = OptimUtils.getGain(sum2, weight2);
				double eval = -(eval1 + eval2);
				if (eval < bestEval) {
					bestEval = eval;
					v2[2] = i;
				}
			}
		}
	}

	protected static void findCuts(Histogram2D.Table table, int[] v1, int v2, boolean hasMissingValue) {
		// Find left cut
		double bestEval = Double.POSITIVE_INFINITY;
		for (int i = 0; i < table.resp.length - 1; i++) {
			double[] resp = table.resp[i][v2];
			double[] count = table.count[i][v2];
			double sum1 = resp[0];
			double sum2 = resp[2];
			double weight1 = count[0];
			double weight2 = count[2];
			double eval1 = OptimUtils.getGain(sum1, weight1);
			double eval2 = OptimUtils.getGain(sum2, weight2);
			double eval = -(eval1 + eval2);
			if (eval < bestEval) {
				bestEval = eval;
				v1[0] = i;
			}
		}

		// Find right cut
		bestEval = Double.POSITIVE_INFINITY;
		for (int i = 0; i < table.resp.length - 1; i++) {
			double[] resp = table.resp[i][v2];
			double[] count = table.count[i][v2];
			double sum1 = resp[1];
			double sum2 = resp[3];
			double weight1 = count[1];
			double weight2 = count[3];
			double eval1 = OptimUtils.getGain(sum1, weight1);
			double eval2 = OptimUtils.getGain(sum2, weight2);
			double eval = -(eval1 + eval2);
			if (eval < bestEval) {
				bestEval = eval;
				v1[1] = i;
			}
		}
		
		if (hasMissingValue) {
			// Find cut on missing value
			bestEval = Double.POSITIVE_INFINITY;
			for (int i = 0; i < table.respOnMV2.length; i++) {
				double[] respOnMV2 = table.respOnMV2[i];
				double[] countOnMV2 = table.countOnMV2[i];
				double sum1 = respOnMV2[0];
				double sum2 = respOnMV2[1];
				double weight1 = countOnMV2[0];
				double weight2 = countOnMV2[1];
				double eval1 = OptimUtils.getGain(sum1, weight1);
				double eval2 = OptimUtils.getGain(sum2, weight2);
				double eval = -(eval1 + eval2);
				if (eval < bestEval) {
					bestEval = eval;
					v1[2] = i;
				}
			}
		}
	}

	protected static void getPredictor(Histogram2D.Table table, int v1, int[] v2, double[] pred) {
		int v21 = v2[0];
		int v22 = v2[1];
		int vMV = v2[2];
		double[] resp1 = table.resp[v1][v21];
		double[] count1 = table.count[v1][v21];
		double[] resp2 = table.resp[v1][v22];
		double[] count2 = table.count[v1][v22];
		pred[0] = MathUtils.divide(resp1[0], count1[0], 0);
		pred[1] = MathUtils.divide(resp1[1], count1[1], 0);
		pred[2] = MathUtils.divide(resp2[2], count2[2], 0);
		pred[3] = MathUtils.divide(resp2[3], count2[3], 0);
		
		if (vMV >= 0) {
			double[] respOnMV1 = table.respOnMV1[vMV];
			double[] countOnMV1 = table.countOnMV1[vMV];
			pred[4] = MathUtils.divide(respOnMV1[0], countOnMV1[0], 0);
			pred[5] = MathUtils.divide(respOnMV1[1], countOnMV1[1], 0);
		}
		
		double[] respOnMV2 = table.respOnMV2[v1];
		double[] countOnMV2 = table.countOnMV2[v1];
		pred[6] = MathUtils.divide(respOnMV2[0], countOnMV2[0], 0);
		pred[7] = MathUtils.divide(respOnMV2[1], countOnMV2[1], 0);
		
		pred[8] = MathUtils.divide(table.respOnMV12, table.countOnMV12, 0);
	}

	protected static void getPredictor(Histogram2D.Table table, int[] v1, int v2, double[] pred) {
		int v11 = v1[0];
		int v12 = v1[1];
		int vMV = v1[2];
		double[] resp1 = table.resp[v11][v2];
		double[] count1 = table.count[v11][v2];
		double[] resp2 = table.resp[v12][v2];
		double[] count2 = table.count[v12][v2];
		pred[0] = MathUtils.divide(resp1[0], count1[0], 0);
		pred[1] = MathUtils.divide(resp1[2], count1[2], 0);
		pred[2] = MathUtils.divide(resp2[1], count2[1], 0);
		pred[3] = MathUtils.divide(resp2[3], count2[3], 0);
		
		if (vMV >= 0) {
			double[] respOnMV2 = table.respOnMV2[vMV];
			double[] countOnMV2 = table.countOnMV2[vMV];
			pred[4] = MathUtils.divide(respOnMV2[0], countOnMV2[0], 0);
			pred[5] = MathUtils.divide(respOnMV2[1], countOnMV2[1], 0);
		}
		double[] respOnMV1 = table.respOnMV1[v2];
		double[] countOnMV1 = table.countOnMV1[v2];
		pred[6] = MathUtils.divide(respOnMV1[0], countOnMV1[0], 0);
		pred[7] = MathUtils.divide(respOnMV1[1], countOnMV1[1], 0);
		
		pred[8] = MathUtils.divide(table.respOnMV12, table.countOnMV12, 0);
		
	}

	protected static double getRSS(Histogram2D.Table table, int v1, int v2[], double[] pred) {
		int v21 = v2[0];
		int v22 = v2[1];
		int vMV = v2[2];
		double[] resp1 = table.resp[v1][v21];
		double[] resp2 = table.resp[v1][v22];
		double[] count1 = table.count[v1][v21];
		double[] count2 = table.count[v1][v22];
		
		double rss = 0;
		rss += pred[0] * pred[0] * count1[0];
		rss += pred[1] * pred[1] * count1[1];
		rss += pred[2] * pred[2] * count2[2];
		rss += pred[3] * pred[3] * count2[3];
		if (vMV >= 0) {
			double[] countOnMV1 = table.countOnMV1[vMV];
			rss += pred[4] * pred[4] * countOnMV1[0];
			rss += pred[5] * pred[5] * countOnMV1[1];
		}
		double[] countOnMV2 = table.countOnMV2[v1];
		rss += pred[6] * pred[6] * countOnMV2[0];
		rss += pred[7] * pred[7] * countOnMV2[1];
		rss += pred[8] * pred[8] * table.countOnMV12;

		double t = 0;
		t += pred[0] * resp1[0];
		t += pred[1] * resp1[1];
		t += pred[2] * resp2[2];
		t += pred[3] * resp2[3];
		if (vMV >= 0) {
			double[] respOnMV1 = table.respOnMV1[vMV];
			t += pred[4] * respOnMV1[0];
			t += pred[5] * respOnMV1[1];
		}
		double[] respOnMV2 = table.respOnMV2[v1];
		t += pred[6] * respOnMV2[0];
		t += pred[7] * respOnMV2[1];
		t += pred[8] * table.respOnMV12;
		rss -= 2 * t;
		return rss;
	}

	protected static double getRSS(Histogram2D.Table table, int[] v1, int v2, double[] pred) {
		int v11 = v1[0];
		int v12 = v1[1];
		int vMV = v1[2];
		double[] resp1 = table.resp[v11][v2];
		double[] resp2 = table.resp[v12][v2];
		double[] count1 = table.count[v11][v2];
		double[] count2 = table.count[v12][v2];
		
		double rss = 0;
		rss += pred[0] * pred[0] * count1[0];
		rss += pred[1] * pred[1] * count1[2];
		rss += pred[2] * pred[2] * count2[1];
		rss += pred[3] * pred[3] * count2[3];
		if (vMV >= 0) {
			double[] countOnMV2 = table.countOnMV2[vMV];
			rss += pred[4] * pred[4] * countOnMV2[0];
			rss += pred[5] * pred[5] * countOnMV2[1];
		}
		double[] countOnMV1 = table.countOnMV1[v2];
		rss += pred[6] * pred[6] * countOnMV1[0];
		rss += pred[7] * pred[7] * countOnMV1[1];
		rss += pred[8] * pred[8] * table.countOnMV12;

		double t = 0;
		t += pred[0] * resp1[0];
		t += pred[1] * resp1[2];
		t += pred[2] * resp2[1];
		t += pred[3] * resp2[3];
		if (vMV >= 0) {
			double[] respOnMV2 = table.respOnMV2[vMV];
			t += pred[4] * respOnMV2[0];
			t += pred[5] * respOnMV2[1];
		}
		double[] respOnMV1 = table.respOnMV1[v2];
		t += pred[6] * respOnMV1[0];
		t += pred[7] * respOnMV1[1];
		t += pred[8] * table.respOnMV12;
		rss -= 2 * t;

		return rss;
	}

	protected static Function2D getFunction2D(int attIndex1, int attIndex2, int v1, int[] v2, double[] predInt) {
		double[] splits1 = new double[] { v1, Double.POSITIVE_INFINITY };
		double[] splits2 = null;
		double[][] predictions = null;
		double[] predictionsOnMV1 = null;
		double[] predictionsOnMV2 = new double[] {predInt[6], predInt[7]};
		if (v2[0] < v2[1]) {
			if (v2[2] < 0 || v2[2] == v2[0] || v2[2] == v2[1]) {
				splits2 = new double[] { v2[0], v2[1], Double.POSITIVE_INFINITY };
				predictions = new double[][] { 
						{ predInt[0], predInt[1], predInt[1] },
						{ predInt[2], predInt[2], predInt[3] } 
				};
				if (v2[2] < 0) {
					predictionsOnMV1 = new double[] { 0.0, 0.0, 0.0 };
				} else if (v2[2] == v2[0]) {
					predictionsOnMV1 = new double[] { predInt[4], predInt[5], predInt[5] };
				} else {
					predictionsOnMV1 = new double[] { predInt[4], predInt[4], predInt[5] };
				}
			} else {
				if (v2[2] < v2[0]) {
					splits2 = new double[] { v2[2], v2[0], v2[1], Double.POSITIVE_INFINITY };
					predictions = new double[][] { 
							{ predInt[0], predInt[0], predInt[1], predInt[1] },
							{ predInt[2], predInt[2], predInt[2], predInt[3] } 
					};
					predictionsOnMV1 = new double[] { predInt[4], predInt[5], predInt[5], predInt[5] };
				} else if (v2[2] < v2[1]) {
					splits2 = new double[] { v2[0], v2[2], v2[1], Double.POSITIVE_INFINITY };
					predictions = new double[][] { 
							{ predInt[0], predInt[1], predInt[1], predInt[1] },
							{ predInt[2], predInt[2], predInt[2], predInt[3] } 
					};
					predictionsOnMV1 = new double[] { predInt[4], predInt[4], predInt[5], predInt[5] };
				} else {
					splits2 = new double[] { v2[0], v2[1], v2[2], Double.POSITIVE_INFINITY };
					predictions = new double[][] { 
							{ predInt[0], predInt[1], predInt[1], predInt[1] },
							{ predInt[2], predInt[2], predInt[3], predInt[3] } 
					};
					predictionsOnMV1 = new double[] { predInt[4], predInt[4], predInt[4], predInt[5] };
				}
			}
		} else if (v2[0] > v2[1]) {
			if (v2[2] < 0 || v2[2] == v2[1] || v2[2] == v2[1]) {
				splits2 = new double[] { v2[1], v2[0], Double.POSITIVE_INFINITY };
				predictions = new double[][] { 
						{ predInt[0], predInt[0], predInt[1] },
						{ predInt[2], predInt[3], predInt[3] } 
				};
				if (v2[2] < 0) {
					predictionsOnMV1 = new double[] { 0.0, 0.0, 0.0 };
				} else if (v2[2] == v2[1]) {
					predictionsOnMV1 = new double[] { predInt[4], predInt[5], predInt[5] };
				} else {
					predictionsOnMV1 = new double[] { predInt[4], predInt[4], predInt[5] };
				}
			} else {
				if (v2[2] < v2[1]) {
					splits2 = new double[] { v2[2], v2[1], v2[0], Double.POSITIVE_INFINITY };
					predictions = new double[][] { 
							{ predInt[0], predInt[0], predInt[0], predInt[1] },
							{ predInt[2], predInt[2], predInt[3], predInt[3] } 
					};
					predictionsOnMV1 = new double[] { predInt[4], predInt[5], predInt[5], predInt[5] };
				} else if (v2[2] < v2[0]) {
					splits2 = new double[] { v2[1], v2[2], v2[0], Double.POSITIVE_INFINITY };
					predictions = new double[][] { 
							{ predInt[0], predInt[0], predInt[0], predInt[1] },
							{ predInt[2], predInt[3], predInt[3], predInt[3] } 
					};
					predictionsOnMV1 = new double[] { predInt[4], predInt[4], predInt[5], predInt[5] };
				} else {
					splits2 = new double[] { v2[1], v2[0], v2[2], Double.POSITIVE_INFINITY };
					predictions = new double[][] { 
							{ predInt[0], predInt[0], predInt[1], predInt[1] },
							{ predInt[2], predInt[3], predInt[3], predInt[3] } 
					};
					predictionsOnMV1 = new double[] { predInt[4], predInt[4], predInt[4], predInt[5] };
				}
			}
		} else {
			// v2[0] == v2[1]
			if (v2[2] < 0 || v2[2] == v2[0]) {
				splits2 = new double[] { v2[0], Double.POSITIVE_INFINITY };
				predictions = new double[][] { 
						{ predInt[0], predInt[1] }, 
						{ predInt[2], predInt[3] }
				};
				if (v2[2] < 0) {
					predictionsOnMV1 = new double[] { 0.0, 0.0 };
				} else {
					predictionsOnMV1 = new double[] { predInt[4], predInt[5] };
				}
			} else {
				if (v2[2] < v2[0]) {
					splits2 = new double[] { v2[2], v2[0], Double.POSITIVE_INFINITY };
					predictions = new double[][] { 
							{ predInt[0], predInt[0], predInt[1] },
							{ predInt[2], predInt[2], predInt[3] } 
					};
					predictionsOnMV1 = new double[] { predInt[4], predInt[5], predInt[5] };
				} else {
					splits2 = new double[] { v2[0], v2[2], Double.POSITIVE_INFINITY };
					predictions = new double[][] { 
							{ predInt[0], predInt[1], predInt[1] },
							{ predInt[2], predInt[3], predInt[3] } 
					};
					predictionsOnMV1 = new double[] { predInt[4], predInt[4], predInt[5] };
				}
			}
		}
		return new Function2D(attIndex1, attIndex2, splits1, splits2, predictions,
				predictionsOnMV1, predictionsOnMV2, predInt[8]);
	}

	protected static Function2D getFunction2D(int attIndex1, int attIndex2, int[] v1, int v2, double[] predInt) {
		double[] splits1 = null;
		double[] splits2 = new double[] { v2, Double.POSITIVE_INFINITY };
		double[] predictionsOnMV1 = new double[] {predInt[6], predInt[7]};
		double[] predictionsOnMV2 = null;
		double[][] predictions = null;
		if (v1[0] < v1[1]) {
			if (v1[2] < 0 || v1[2] == v1[0] || v1[2] == v1[1]) {
				splits1 = new double[] { v1[0], v1[1], Double.POSITIVE_INFINITY };
				predictions = new double[][] { 
						{ predInt[0], predInt[2] }, 
						{ predInt[1], predInt[2] },
						{ predInt[1], predInt[3] }
				};
				if (v1[2] < 0) {
					predictionsOnMV2 = new double[] { 0.0, 0.0, 0.0 };
				} else if (v1[2] == v1[0]) {
					predictionsOnMV2 = new double[] { predInt[4], predInt[5], predInt[5] };
				} else {
					predictionsOnMV2 = new double[] { predInt[4], predInt[4], predInt[5] };
				}
			} else {
				if (v1[2] < v1[0]) {
					splits1 = new double[] { v1[2], v1[0], v1[1], Double.POSITIVE_INFINITY };
					predictions = new double[][] { 
						{ predInt[0], predInt[2] }, 
						{ predInt[0], predInt[2] }, 
						{ predInt[1], predInt[2] },
						{ predInt[1], predInt[3] } 
					};
					predictionsOnMV2 = new double[] { predInt[4], predInt[5], predInt[5], predInt[5] };
				} else if (v1[2] < v1[1]) {
					splits1 = new double[] { v1[0], v1[2], v1[1], Double.POSITIVE_INFINITY };
					predictions = new double[][] { 
						{ predInt[0], predInt[2] }, 
						{ predInt[1], predInt[2] }, 
						{ predInt[1], predInt[2] },
						{ predInt[1], predInt[3] } 
					};
					predictionsOnMV2 = new double[] { predInt[4], predInt[4], predInt[5], predInt[5] };
				} else {
					splits1 = new double[] { v1[0], v1[1], v1[2], Double.POSITIVE_INFINITY };
					predictions = new double[][] { 
						{ predInt[0], predInt[2] }, 
						{ predInt[1], predInt[2] }, 
						{ predInt[1], predInt[3] },
						{ predInt[1], predInt[3] } 
					};
					predictionsOnMV2 = new double[] { predInt[4], predInt[4], predInt[4], predInt[5] };
				}
			}
		} else if (v1[0] > v1[1]) {
			if (v1[2] < 0 || v1[2] == v1[0] || v1[2] == v1[1]) {
				splits1 = new double[] { v1[1], v1[0], Double.POSITIVE_INFINITY };
				predictions = new double[][] { 
						{ predInt[0], predInt[2] }, 
						{ predInt[0], predInt[3] },
						{ predInt[1], predInt[3] } 
				};
				if (v1[2] < 0) {
					predictionsOnMV2 = new double[] { 0.0, 0.0, 0.0 };
				} else if (v1[2] == v1[0]) {
					predictionsOnMV2 = new double[] { predInt[4], predInt[4], predInt[5] };
				} else {
					predictionsOnMV2 = new double[] { predInt[4], predInt[5], predInt[5] };
				}
			} else {
				if (v1[2] < v1[1]) {
					splits1 = new double[] { v1[2], v1[1], v1[0], Double.POSITIVE_INFINITY };
					predictions = new double[][] { 
						{ predInt[0], predInt[2] }, 
						{ predInt[0], predInt[2] }, 
						{ predInt[0], predInt[3] },
						{ predInt[1], predInt[3] } 
					};
					predictionsOnMV2 = new double[] { predInt[4], predInt[5], predInt[5], predInt[5] };
				} else if (v1[2] < v1[0]) {
					splits1 = new double[] { v1[1], v1[2], v1[0], Double.POSITIVE_INFINITY };
					predictions = new double[][] { 
						{ predInt[0], predInt[2] }, 
						{ predInt[0], predInt[3] }, 
						{ predInt[0], predInt[3] },
						{ predInt[1], predInt[3] } 
					};
					predictionsOnMV2 = new double[] { predInt[4], predInt[4], predInt[5], predInt[5] };
				} else {
					splits1 = new double[] { v1[1], v1[0], v1[2], Double.POSITIVE_INFINITY };
					predictions = new double[][] { 
						{ predInt[0], predInt[2] }, 
						{ predInt[0], predInt[2] }, 
						{ predInt[0], predInt[3] },
						{ predInt[1], predInt[3] } 
					};
					predictionsOnMV2 = new double[] { predInt[4], predInt[4], predInt[4], predInt[5] };
				}
			}
		} else {
			// v1[0] ==  v1[1]
			if (v1[2] < 0 || v1[2] == v1[0]) {
				splits1 = new double[] { v1[0], Double.POSITIVE_INFINITY };
				predictions = new double[][] { 
						{ predInt[0], predInt[2] }, 
						{ predInt[1], predInt[3] } 
				};
				if (v1[2] < 0) {
					predictionsOnMV2 = new double[] { 0.0, 0.0 };
				} else {
					predictionsOnMV2 = new double[] { predInt[4], predInt[5] };
				}
			} else {
				if (v1[2] < v1[0]) {
					splits1 = new double[] { v1[2], v1[0], Double.POSITIVE_INFINITY };
					predictions = new double[][] { 
							{ predInt[0], predInt[2] }, 
							{ predInt[0], predInt[2] }, 
							{ predInt[1], predInt[3] } 
					};
					predictionsOnMV2 = new double[] { predInt[4], predInt[5], predInt[5] };
				} else {
					splits1 = new double[] { v1[2], v1[0], Double.POSITIVE_INFINITY };
					predictions = new double[][] { 
							{ predInt[0], predInt[2] }, 
							{ predInt[1], predInt[3] }, 
							{ predInt[1], predInt[3] } 
					};
					predictionsOnMV2 = new double[] { predInt[4], predInt[4], predInt[5] };
				}
			}
		}
		return new Function2D(attIndex1, attIndex2, splits1, splits2, predictions,
				predictionsOnMV1, predictionsOnMV2, predInt[8]);
	}

	protected static void lineSearch(Instances instances, int attIndex1, int attIndex2, int c1, int c21, int c22, int cMV, 
			double[] predictions) {
		double[] numerator = new double[9];
		double[] denominator = new double[9];
		for (Instance instance : instances) {
			final double target = instance.getTarget();
			final double weight = instance.getWeight();
			final double t = Math.abs(target);
			final double num = target * weight;
			final double den = t * (1 - t) * weight;
			if (!instance.isMissing(attIndex1) && !instance.isMissing(attIndex2)) {
				int v1 = (int) instance.getValue(attIndex1);
				int v2 = (int) instance.getValue(attIndex2);
				if (v1 <= c1) {
					if (v2 <= c21) {
						numerator[0] += num;
						denominator[0] += den;
					} else {
						numerator[1] += num;
						denominator[1] += den;
					}
				} else {
					if (v2 <= c22) {
						numerator[2] += num;
						denominator[2] += den;
					} else {
						numerator[3] += num;
						denominator[3] += den;
					}
				}
			} else if (instance.isMissing(attIndex1) && !instance.isMissing(attIndex2)) {
				int v2 = (int) instance.getValue(attIndex2);
				if (cMV >= 0) {
					if (v2 <= cMV) {
						numerator[4] += num;
						denominator[4] += den;
					} else {
						numerator[5] += num;
						denominator[5] += den;
					}
				} else {
					throw new RuntimeException("Something went wrong");
				}
			} else if (!instance.isMissing(attIndex1) && instance.isMissing(attIndex2)) {
				int v1 = (int) instance.getValue(attIndex1);
				if (v1 <= c1) {
					numerator[6] += num;
					denominator[6] += den;
				} else {
					numerator[7] += num;
					denominator[7] += den;
				}
			} else {
				numerator[8] += num;
				denominator[8] += den;
			}
		}
		for (int i = 0; i < numerator.length; i++) {
			predictions[i] = MathUtils.divide(numerator[i], denominator[i], 0);
		}
	}

}
