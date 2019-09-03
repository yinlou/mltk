package mltk.predictor.function;

import mltk.core.Instance;
import mltk.core.Instances;
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
	public double[] respOnMV1;
	public double[] countOnMV1;
	public double[] respOnMV2;
	public double[] countOnMV2;
	public double respOnMV12;
	public double countOnMV12;
	
	public static class Table {

		public double[][][] resp;
		public double[][][] count;
		public double[][] respOnMV1;
		public double[][] countOnMV1;
		public double[][] respOnMV2;
		public double[][] countOnMV2;
		public double respOnMV12;
		public double countOnMV12;
		

		public Table(int n, int m) {
			resp = new double[n][m][4];
			count = new double[n][m][4];
			respOnMV1 = new double[m][2];
			countOnMV1 = new double[m][2];
			respOnMV2 = new double[n][2];
			countOnMV2 = new double[n][2];
			respOnMV12 = 0.0;
			countOnMV12 = 0.0;
		}
	}
	
	/**
	 * Computes 2D histogram given (f1, f2).
	 * 
	 * @param instances the data set.
	 * @param f1 the 1st feature.
	 * @param f2 the 2nd feature.
	 * @param hist2d the histogram to compute.
	 */
	public static void computeHistogram2D(Instances instances, int f1, int f2, Histogram2D hist2d) {
		for (Instance instance : instances) {
			double resp = instance.getTarget() * instance.getWeight();
			double weight = instance.getWeight();
			if (!instance.isMissing(f1) && !instance.isMissing(f2)) {
				int idx1 = (int) instance.getValue(f1);
				int idx2 = (int) instance.getValue(f2);
				hist2d.resp[idx1][idx2] += resp;
				hist2d.count[idx1][idx2] += weight;
			} else if (instance.isMissing(f1) && !instance.isMissing(f2)) {
				int idx2 = (int) instance.getValue(f2);
				hist2d.respOnMV1[idx2] += resp;
				hist2d.countOnMV1[idx2] += weight;
			} else if (!instance.isMissing(f1) && instance.isMissing(f2)) {
				int idx1 = (int) instance.getValue(f1);
				hist2d.respOnMV2[idx1] += resp;
				hist2d.countOnMV2[idx1] += weight;
			} else {
				hist2d.respOnMV12 += resp;
				hist2d.countOnMV12 += weight;
			}
		}
	}
	
	/**
	 * Computes auxiliary data structure given 2D histogram and cumulative 1D histograms.
	 * 
	 * @param hist2d the 2D histogram.
	 * @param cHist1 the cumulative histogram for the 1st feature.
	 * @param cHist2 the cumulative histogram for the 2nd feature.
	 * @return table auxiliary data structure.
	 */
	public static Table computeTable(Histogram2D hist2d, CHistogram cHist1, CHistogram cHist2) {
		Table table = new Table(hist2d.resp.length, hist2d.resp[0].length);
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
		
		double respOnMV1 = 0;
		double countOnMV1 = 0;
		for (int j = 0; j < hist2d.respOnMV1.length; j++) {
			respOnMV1 += hist2d.respOnMV1[j];
			countOnMV1 += hist2d.countOnMV1[j];
			table.respOnMV1[j][0] = respOnMV1;
			table.respOnMV1[j][1] = cHist1.sumOnMV - respOnMV1;
			table.countOnMV1[j][0] = countOnMV1;
			table.countOnMV1[j][1] = cHist1.countOnMV - countOnMV1;
		}
		
		double respOnMV2 = 0;
		double countOnMV2 = 0;
		for (int i = 0; i < hist2d.respOnMV2.length; i++) {
			respOnMV2 += hist2d.respOnMV2[i];
			countOnMV2 += hist2d.countOnMV2[i];
			table.respOnMV2[i][0] = respOnMV2;
			table.respOnMV2[i][1] = cHist2.sumOnMV - respOnMV2;
			table.countOnMV2[i][0] = countOnMV2;
			table.countOnMV2[i][1] = cHist2.countOnMV - countOnMV2;
		}
		
		table.respOnMV12 = hist2d.respOnMV12;
		table.countOnMV12 = hist2d.countOnMV12;
		
		return table;
	}

	protected static void fillTable(Table table, int i, int j, CHistogram cHist1, CHistogram cHist2) {
		double[] count = table.count[i][j];
		double[] resp = table.resp[i][j];
		resp[1] = cHist1.sum[i] - resp[0];
		resp[2] = cHist2.sum[j] - resp[0];
		resp[3] = cHist1.sum[cHist1.size() - 1] - cHist1.sum[i] - resp[2];

		count[1] = cHist1.count[i] - count[0];
		count[2] = cHist2.count[j] - count[0];
		count[3] = cHist1.count[cHist1.size() - 1] - cHist1.count[i] - count[2];
	}

	/**
	 * Constructor.
	 * 
	 * @param n the size of the 1st dimension.
	 * @param m the size of the 2nd dimension.
	 */
	public Histogram2D(int n, int m) {
		resp = new double[n][m];
		count = new double[n][m];
		respOnMV1 = new double[m];
		countOnMV1 = new double[m];
		respOnMV2 = new double[n];
		countOnMV2 = new double[n];
		respOnMV12 = 0.0;
		countOnMV12 = 0.0;
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
				
				cHist2.sum[j] += r[j];
				cHist2.count[j] += c[j];
			}
		}

		for (int i = 1; i < cHist1.size(); i++) {
			cHist1.sum[i] += cHist1.sum[i - 1];
			cHist1.count[i] += cHist1.count[i - 1];
		}

		for (int j = 1; j < cHist2.size(); j++) {
			cHist2.sum[j] += cHist2.sum[j - 1];
			cHist2.count[j] += cHist2.count[j - 1];
		}
		
		for (int j = 0; j < respOnMV1.length; j++) {
			cHist1.sumOnMV += respOnMV1[j];
			cHist1.countOnMV += countOnMV1[j];
		}
		cHist1.sumOnMV += respOnMV12;
		cHist1.countOnMV += countOnMV12;
		
		for (int i = 0; i < respOnMV2.length; i++) {
			cHist2.sumOnMV += respOnMV2[i];
			cHist2.countOnMV += countOnMV2[i];
		}
		cHist2.sumOnMV += respOnMV12;
		cHist2.countOnMV += countOnMV12;

		return new Pair<CHistogram, CHistogram>(cHist1, cHist2);
	}

}
