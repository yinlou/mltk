package mltk.predictor.gam;

import java.util.HashSet;
import java.util.Set;

import mltk.predictor.function.CubicSpline;
import mltk.util.StatUtils;
import mltk.util.VectorUtils;

class SparseDesignMatrix {

	int[][] indices;
	double[][][] values;
	double[][] knots;
	double[][] std;

	SparseDesignMatrix(int[][] indices, double[][][] values, double[][] knots, double[][] std) {
		this.indices = indices;
		this.values = values;
		this.knots = knots;
		this.std = std;
	}

	static SparseDesignMatrix createCubicSplineDesignMatrix(int n, int[][] indices, double[][] values,
			double[] stdList, int numKnots) {
		final int p = indices.length;
		double[][][] x = new double[p][][];
		double[][] knots = new double[p][];
		double[][] std = new double[p][];
		double factor = Math.sqrt(n);
		for (int j = 0; j < values.length; j++) {
			Set<Double> uniqueValues = new HashSet<>();
			double[] x1 = values[j];
			for (int i = 0; i < values[j].length; i++) {
				uniqueValues.add(x1[i]);
			}
			int nKnots = uniqueValues.size() + 1 <= numKnots ? 0 : numKnots;
			knots[j] = new double[nKnots];
			if (nKnots != 0) {
				x[j] = new double[nKnots + 3][];
				std[j] = new double[nKnots + 3];
			} else {
				x[j] = new double[1][];
				std[j] = new double[1];
			}
			double[][] tX = x[j];
			tX[0] = x1;
			std[j][0] = stdList[j] / factor;
			if (nKnots != 0) {
				double[] x2 = new double[x1.length];
				for (int i = 0; i < x2.length; i++) {
					x2[i] = x1[i] * x1[i];
				}
				tX[1] = x2;
				double[] x3 = new double[x1.length];
				for (int i = 0; i < x3.length; i++) {
					x3[i] = x2[i] * x1[i];
				}
				tX[2] = x3;

				std[j][1] = StatUtils.std(x2, n) / factor;
				std[j][2] = StatUtils.std(x3, n) / factor;

				double max = Math.max(StatUtils.max(x1), 0);
				double min = Math.min(StatUtils.min(x1), 0);
				double stepSize = (max - min) / nKnots;
				for (int k = 0; k < nKnots; k++) {
					knots[j][k] = min + stepSize * k;
					double[] basis = new double[x1.length];
					double zero = CubicSpline.h(0, knots[j][k]);
					for (int i = 0; i < basis.length; i++) {
						basis[i] = CubicSpline.h(x1[i], knots[j][k]) - zero;
					}
					std[j][k + 3] = StatUtils.std(basis, n) / factor;
					tX[k + 3] = basis;
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
		return new SparseDesignMatrix(indices, x, knots, std);
	}

}
