package mltk.predictor.function;

import mltk.util.tuple.Pair;

/**
 * Class for 2D histograms.
 * 
 * @author Yin Lou
 * 
 */
public class Histogram2D {

	public double[][] resp;
	public double[][] count;

	/**
	 * Constructor.
	 * 
	 * @param n the size of the 1st dimension.
	 * @param m the size of the 2nd dimension.
	 */
	public Histogram2D(int n, int m) {
		resp = new double[n][m];
		count = new double[n][m];
	}

	/**
	 * Computes the cumulative histograms on the margin.
	 * 
	 * @return the cumulative histograms.
	 */
	public Pair<CHistogram, CHistogram> computeCHistogram() {
		CHistogram cHist1 = new CHistogram(resp.length);
		CHistogram cHist2 = new CHistogram(resp[0].length);

		for (int i = 0; i < resp.length; i++) {
			double[] r = resp[i];
			double[] c = count[i];
			for (int j = 0; j < r.length; j++) {
				cHist1.sum[i] += r[j];
				cHist1.count[i] += c[j];
			}
		}

		for (int i = 0; i < resp.length; i++) {
			double[] r = resp[i];
			double[] c = count[i];
			for (int j = 0; j < r.length; j++) {
				cHist2.sum[j] += r[j];
				cHist2.count[j] += c[j];
			}
		}

		for (int i = 1; i < cHist1.size(); i++) {
			cHist1.sum[i] += cHist1.sum[i - 1];
			cHist1.count[i] += cHist1.count[i - 1];
		}

		for (int i = 1; i < cHist2.size(); i++) {
			cHist2.sum[i] += cHist2.sum[i - 1];
			cHist2.count[i] += cHist2.count[i - 1];
		}

		return new Pair<CHistogram, CHistogram>(cHist1, cHist2);
	}

}
