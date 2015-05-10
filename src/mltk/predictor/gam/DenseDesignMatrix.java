package mltk.predictor.gam;

import java.util.HashSet;
import java.util.Set;

import mltk.predictor.function.CubicSpline;
import mltk.util.StatUtils;
import mltk.util.VectorUtils;

class DenseDesignMatrix {

	double[][][] x;
	double[][] knots;
	double[][] std;

	DenseDesignMatrix(double[][][] x, double[][] knots, double[][] std) {
		this.x = x;
		this.knots = knots;
		this.std = std;
	}

	static DenseDesignMatrix createCubicSplineDesignMatrix(double[][] dataset, double[] stdList, int numKnots) {
		final int n = dataset[0].length;
		final int p = dataset.length;
		double[][][] x = new double[p][][];
		double[][] knots = new double[p][];
		double[][] std = new double[p][];
		double factor = Math.sqrt(n);
		for (int j = 0; j < dataset.length; j++) {
			Set<Double> uniqueValues = new HashSet<>();
			double[] x1 = dataset[j];
			for (int i = 0; i < n; i++) {
				uniqueValues.add(x1[i]);
			}
			int nKnots = uniqueValues.size() <= numKnots ? 0 : numKnots;
			knots[j] = new double[nKnots];
			if (nKnots != 0) {
				x[j] = new double[nKnots + 3][];
				std[j] = new double[nKnots + 3];
			} else {
				x[j] = new double[1][];
				std[j] = new double[1];
			}
			double[][] t = x[j];
			t[0] = x1;
			std[j][0] = stdList[j] / factor;
			if (nKnots != 0) {
				double[] x2 = new double[n];
				for (int i = 0; i < n; i++) {
					x2[i] = x1[i] * x1[i];
				}
				t[1] = x2;
				double[] x3 = new double[n];
				for (int i = 0; i < n; i++) {
					x3[i] = x2[i] * x1[i];
				}
				t[2] = x3;

				std[j][1] = StatUtils.std(x2) / factor;
				std[j][2] = StatUtils.std(x3) / factor;

				double max = StatUtils.max(x1);
				double min = StatUtils.min(x1);
				double stepSize = (max - min) / nKnots;
				for (int k = 0; k < nKnots; k++) {
					knots[j][k] = min + stepSize * k;
					double[] basis = new double[n];
					for (int i = 0; i < n; i++) {
						basis[i] = CubicSpline.h(x1[i], knots[j][k]);
					}
					std[j][k + 3] = StatUtils.std(basis) / factor;
					t[k + 3] = basis;
				}
			}
		}
		
		// Normalize the inputs
		for (int j = 0; j < p; j++) {
			double[][] block = x[j];
			double[] s = std[j];
			for (int i = 0; i < block.length; i++) {
				VectorUtils.divide(block[i], s[i]);
			}
		}
		return new DenseDesignMatrix(x, knots, std);
	}

}
