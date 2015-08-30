package mltk.predictor.tree.ensemble.ag;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.cmdline.options.LearnerOptions;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.io.InstancesReader;
import mltk.predictor.Bagging;
import mltk.predictor.Learner;
import mltk.predictor.evaluation.AUC;
import mltk.predictor.evaluation.Metric;
import mltk.predictor.evaluation.RMSE;
import mltk.predictor.io.PredictorWriter;
import mltk.predictor.tree.RegressionTree;
import mltk.predictor.tree.RegressionTreeLearner;
import mltk.predictor.tree.RegressionTreeLearner.Mode;
import mltk.util.Random;
import mltk.util.tuple.IntPair;

/**
 * Class for learning Additive Groves. This class currently only supports layered training.
 *
 * <p>
 * Reference:<br>
 * D. Sorokina, R. Caruana and M. Riedewald. Additive Groves of Regression Trees. In <i>Proceedings of the 18th European
 * Conference on Machine Learning (ECML)</i>, Warsaw, Poland, 2007.
 * </p>
 *
 * @author Yin Lou
 *
 */
public class AdditiveGrovesLearner extends Learner {
	
	static class Options extends LearnerOptions {

		@Argument(name = "-v", description = "valid set path", required = true)
		String validPath = null;

		@Argument(name = "-o", description = "output model path")
		String outputModelPath = null;

		@Argument(name = "-e", description = "AUC (a), RMSE (r) (default: r)")
		String metric = null;

		@Argument(name = "-b", description = "bagging iterations (default: 60)")
		int baggingIters = 60;

		@Argument(name = "-n", description = "number of trees in a grove (default: 6)")
		int n = 6;

		@Argument(name = "-a", description = "minimum alpha (default: 0.01)")
		double a = 0.01;

		@Argument(name = "-s", description = "seed of the random number generator (default: 0)")
		long seed = 0L;

	}

	/**
	 * <p>
	 * 
	 * <pre>
	 * Usage: mltk.predictor.tree.ensemble.ag.AdditiveGrovesLearner
	 * -t	train set path
	 * -v	valid set path
	 * [-r]	attribute file path
	 * [-o]	output model path
	 * [-V]	verbose (default: true)
	 * [-o]	output model path
	 * [-e]	AUC (a), RMSE (r) (default: r)
	 * [-b]	bagging iterations (default: 60)
	 * [-n]	number of trees in a grove (default: 6)
	 * [-a]	minimum alpha (default: 0.01)
	 * [-s]	seed of the random number generator (default: 0)
	 * </pre>
	 * 
	 * </p>
	 * 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(AdditiveGrovesLearner.class, opts);
		Metric metric = null;
		try {
			parser.parse(args);
			if ("rmse".startsWith(opts.metric)) {
				metric = new RMSE();
			} else if ("auc".startsWith(opts.metric)) {
				metric = new AUC();
			}
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}

		Random.getInstance().setSeed(opts.seed);

		Instances trainSet = InstancesReader.read(opts.attPath, opts.trainPath);
		Instances validSet = InstancesReader.read(opts.attPath, opts.validPath);

		AdditiveGrovesLearner learner = new AdditiveGrovesLearner();
		learner.setBaggingIters(opts.baggingIters);
		learner.setNumTrees(opts.n);
		learner.setMinAlpha(opts.a);
		learner.setMetric(metric);
		learner.setVerbose(opts.verbose);

		long start = System.currentTimeMillis();
		AdditiveGroves ag = learner.buildRegressor(trainSet, validSet);
		long end = System.currentTimeMillis();
		System.out.println("Time: " + (end - start) / 1000.0);

		if (opts.outputModelPath != null) {
			PredictorWriter.write(ag, opts.outputModelPath);
		}
	}

	class PerformanceMatrix {

		Metric metric;
		double[][][] perf;

		PerformanceMatrix(int maxNumTrees, int numAlphas, int baggingIters, Metric metric) {
			perf = new double[maxNumTrees][numAlphas][baggingIters];
			this.metric = metric;
		}

		void expand(int maxNumTrees, int numAlphas, int baggingIters) {
			// TODO make sure all dimensions are increasing only
			double[][][] newPerf = new double[maxNumTrees][numAlphas][baggingIters];
			for (int i = 0; i < perf.length; i++) {
				double[][] t1Old = perf[i];
				double[][] t1New = newPerf[i];
				for (int j = 0; j < t1Old.length; j++) {
					double[] t2Old = t1Old[j];
					double[] t2New = t1New[j];
					for (int k = 0; k < t2Old.length; k++) {
						t2New[k] = t2Old[k];
					}
				}
			}
			perf = newPerf;
		}

		void eval(int t, int a, int b, double[] preds, double[] targets) {
			perf[t][a][b] = metric.eval(preds, targets);
		}

		IntPair getBestParameters() {
			int bestT = 0;
			int bestA = 0;
			double bestPerf = metric.worstValue();
			if (verbose) {
				System.out.println("Perf Matrix:");
				for (int i = 0; i < perf.length; i++) {
					double[][] tOutter = perf[i];
					for (int j = 0; j < tOutter.length; j++) {
						double[] tInner = tOutter[j];
						double p = tInner[tInner.length - 1];
						System.out.print(p + " ");
						if (metric.isFirstBetter(p, bestPerf)) {
							bestT = i;
							bestA = j;
							bestPerf = p;
						}
					}
					System.out.println();
				}
				System.out.println("Best perf on validation set = " + bestPerf);
			}
			return new IntPair(bestT, bestA);
		}

		/**
		 * Returns <code>true</code> if the bagging converges.
		 *
		 * @param t the number of trees.
		 * @param a the index of alpha.
		 * @return <code>true</code> if the bagging converges.
		 */
		boolean analyzeBagging(int t, int a) {
			return Bagging.analyzeBagging(perf[t][a], metric);
		}

	}

	class ModelMatrix {

		AdditiveGroves[][] groves;

		ModelMatrix(int maxNumTrees, int numAlphas) {
			groves = new AdditiveGroves[maxNumTrees][numAlphas];
		}

		void expand(int maxNumTrees, int numAlphas, int baggingIters) {
			AdditiveGroves[][] newGroves = new AdditiveGroves[maxNumTrees][numAlphas];
			for (int i = 0; i < groves.length; i++) {
				for (int j = 0; j < groves[i].length; j++) {
					newGroves[i][j] = groves[i][j];
				}
			}
			groves = newGroves;
		}

		void add(int t, int a, RegressionTree[] grove) {
			if (groves[t][a] == null) {
				groves[t][a] = new AdditiveGroves();
			}
			groves[t][a].groves.add(grove);
		}

	}

	class PredictionMatrix {

		double[][][] sumPrediction;
		int n;

		PredictionMatrix(int tn, int an, int n) {
			sumPrediction = new double[tn][an][n];
			this.n = n;
		}

		void expand(int tn, int an) {
			double[][][] newSumPrediction = new double[tn][an][n];
			for (int t = 0; t < sumPrediction.length; t++) {
				double[][] tSrc = sumPrediction[t];
				double[][] tDes = newSumPrediction[t];
				for (int a = 0; a < tSrc.length; a++) {
					System.arraycopy(tSrc[a], 0, tDes[a], 0, n);
				}
			}
			sumPrediction = newSumPrediction;
		}

	}

	private int bestNumTrees;
	private int bestBaggingIters;
	private double bestAlpha;

	private int numTrees;
	private int baggingIters;
	private double minAlpha;
	private Metric metric;

	/**
	 * Constructor.
	 */
	public AdditiveGrovesLearner() {
		verbose = false;
		numTrees = 6;
		baggingIters = 60;
		minAlpha = 0.01;
		metric = new RMSE();
	}

	/**
	 * Returns the metric.
	 * 
	 * @return the metric.
	 */
	public Metric getMetric() {
		return metric;
	}

	/**
	 * Sets the metric. Currently only support {@link mltk.predictor.evaluation.RMSE RMSE} and
	 * {@link mltk.predictor.evaluation.AUC AUC}.
	 * 
	 * @param metric the metric.
	 */
	public void setMetric(Metric metric) {
		if (metric instanceof RMSE || metric instanceof AUC) {
			this.metric = metric;
		}
	}

	/**
	 * Returns the best number of trees in a grove from latest run.
	 * 
	 * @return the best number of trees in a grove from latest run.
	 */
	public int getBestNumTrees() {
		return bestNumTrees;
	}

	/**
	 * Returns the best bagging iterations from latest run.
	 * 
	 * @return the best bagging iterations from latest run.
	 */
	public int getBestBaggingIters() {
		return bestBaggingIters;
	}

	/**
	 * Returns the best alpha from latest run.
	 * 
	 * @return the best alpha from latest run.
	 */
	public double getBestAlpha() {
		return bestAlpha;
	}

	/**
	 * Returns the number of trees in a grove. The number of trees may be adjusted during the training.
	 * 
	 * @return the number of trees in a grove.
	 */
	public int getNumTrees() {
		return numTrees;
	}

	/**
	 * Sets the number of trees in a grove. The number of trees may be adjusted during the training.
	 * 
	 * @param numTrees the number of trees in a grove.
	 */
	public void setNumTrees(int numTrees) {
		this.numTrees = numTrees;
	}

	/**
	 * Returns the minimum alpha. The minimum alpha may be adjusted during the training.
	 * 
	 * @return the minimum alpha.
	 */
	public double getMinAlpha() {
		return minAlpha;
	}

	/**
	 * Sets the minimum alpha. The minimum alpha may be adjusted during the training.
	 * 
	 * @param minAlpha the minimum alpha.
	 */
	public void setMinAlpha(double minAlpha) {
		this.minAlpha = minAlpha;
	}

	/**
	 * Returns the number of bagging iterations. The number of bagging iterations may be adjusted during the training.
	 *
	 * @return the number of bagging iterations.
	 */
	public int getBaggingIters() {
		return baggingIters;
	}

	/**
	 * Sets the number of bagging iterations. The number of bagging iterations may be adjusted during the training.
	 *
	 * @param baggingIters the bagging iterations.
	 */
	public void setBaggingIters(int baggingIters) {
		this.baggingIters = baggingIters;
	}

	/**
	 * Builds additive groves.
	 * 
	 * @param trainSet the training set.
	 * @param validSet the validation set.
	 * @return a regressor.
	 */
	public AdditiveGroves buildRegressor(Instances trainSet, Instances validSet) {
		int bn = baggingIters;
		int tn = numTrees;
		int an = 6;
		List<Double> alphas = new ArrayList<>();
		for (int a = 0; a < an; a++) {
			alphas.add(getAlpha(a));
		}

		int prevBN = 0;
		int prevTN = 0;
		int prevAN = 0;

		// Backup targets
		double[] targetTrain = new double[trainSet.size()];
		for (int i = 0; i < targetTrain.length; i++) {
			targetTrain[i] = trainSet.get(i).getTarget();
		}
		double[] targetValid = new double[validSet.size()];
		for (int i = 0; i < targetValid.length; i++) {
			targetValid[i] = validSet.get(i).getTarget();
		}

		PerformanceMatrix perfMatrix = new PerformanceMatrix(tn, an, bn, metric);
		ModelMatrix modelMatrix = new ModelMatrix(tn, an);
		PredictionMatrix predMatrix = new PredictionMatrix(tn, an, validSet.size());

		IntPair bestParams = null;

		for (;;) {
			// Running bagging iterations
			runLayeredTraining(trainSet, validSet, prevBN, bn, 0, tn, 0, an, alphas, perfMatrix, modelMatrix,
					predMatrix, targetTrain, targetValid);

			bestParams = perfMatrix.getBestParameters();
			boolean converged = true;

			prevBN = bn;
			prevTN = tn;
			prevAN = an;

			// Expand alpha
			if (bestParams.v2 == an - 1 && alphas.get(an - 1) > 1.0 / trainSet.size()) {
				converged = false;
				an += 3;
				for (int a = prevAN; a < an; a++) {
					alphas.add(getAlpha(a));
				}
				System.out.println(an);

				predMatrix.expand(tn, an);
				perfMatrix.expand(tn, an, bn);
				modelMatrix.expand(tn, an, bn);

				runLayeredTraining(trainSet, validSet, 0, bn, 0, tn, prevAN, an, alphas, perfMatrix, modelMatrix,
						predMatrix, targetTrain, targetValid);
			}

			// Expand number of trees
			if (bestParams.v1 == tn - 1) {
				converged = false;
				tn += 3;

				predMatrix.expand(tn, an);
				perfMatrix.expand(tn, an, bn);
				modelMatrix.expand(tn, an, bn);

				runLayeredTraining(trainSet, validSet, 0, bn, prevTN, tn, 0, an, alphas, perfMatrix, modelMatrix,
						predMatrix, targetTrain, targetValid);
			}

			// Expand bagging
			if (!perfMatrix.analyzeBagging(bestParams.v1, bestParams.v2)) {
				converged = false;
				bn += 40;

				predMatrix.expand(tn, an);
				perfMatrix.expand(tn, an, bn);
				modelMatrix.expand(tn, an, bn);
			}

			if (converged) {
				break;
			}
		}

		// Restore targets
		for (int i = 0; i < targetTrain.length; i++) {
			trainSet.get(i).setTarget(targetTrain[i]);
		}

		System.out.println("Best model:");
		System.out.println("Alpha = " + alphas.get(bestParams.v2));
		System.out.println("N = " + (bestParams.v1 + 1));
		System.out.println("b = " + bn);
		bestBaggingIters = bn;
		bestNumTrees = bestParams.v1 + 1;
		bestAlpha = getAlpha(bestParams.v2);
		return modelMatrix.groves[bestParams.v1][bestParams.v2];
	}

	/**
	 * Builds additive groves using layered training.
	 * 
	 * @param trainSet the training set.
	 * @param baggingIters the number of bagging iterations.
	 * @param numTrees the number of trees in a grove.
	 * @param alpha the alpha.
	 * @return a regressor.
	 */
	public AdditiveGroves runLayeredTraining(Instances trainSet, int baggingIters, int numTrees, double alpha) {
		final int n = trainSet.size();

		// Backup targets
		double[] targetTrain = new double[trainSet.size()];
		for (int i = 0; i < targetTrain.length; i++) {
			targetTrain[i] = trainSet.get(i).getTarget();
		}

		int bn = baggingIters;
		int tn = numTrees;
		int an = getAlphaIdx(alpha, trainSet.size()) + 1;
		AdditiveGroves ag = new AdditiveGroves();
		for (int b = 0; b < bn; b++) {
			// The most recent predictions for regression trees
			double[][] rtPreds = new double[tn][n];
			// The most recent residuals
			double[] residualTrain = new double[n];
			for (int i = 0; i < n; i++) {
				residualTrain[i] = targetTrain[i];
			}

			if (verbose) {
				System.out.println("Iteration " + (b + 1) + " out of " + bn);
			}

			for (int a = 0; a < an; a++) {
				double currAlpha = getAlpha(a);

				if (verbose) {
					System.out.println("\tBuilding models with alpha = " + currAlpha);
				}

				RegressionTree[] grove = new RegressionTree[tn];
				backfit(trainSet, currAlpha, grove, rtPreds, residualTrain);
				ag.groves.add(grove);
			}
		}

		// Restore targets
		for (int i = 0; i < targetTrain.length; i++) {
			trainSet.get(i).setTarget(targetTrain[i]);
		}

		return ag;
	}

	@Override
	public AdditiveGroves build(Instances instances) {
		Instances trainSet = new Instances(instances.getAttributes(), instances.getTargetAttribute());
		Instances validSet = new Instances(instances.getAttributes(), instances.getTargetAttribute());
		int nTrain = instances.size() / 5 * 4;
		for (int i = 0; i < nTrain; i++) {
			trainSet.add(instances.get(i));
		}
		for (int i = nTrain; i < instances.size(); i++) {
			validSet.add(instances.get(i));
		}
		return buildRegressor(trainSet, validSet);
	}

	protected double getAlpha(int an) {
		double alpha = 1;
		if (an % 3 == 0) {
			alpha = 5;
		} else if (an % 3 == 1) {
			alpha = 2;
		}
		for (int i = 0; i < an / 3 + 1; i++) {
			alpha /= 10;
		}
		return alpha;
	}

	protected int getAlphaIdx(double alpha, int n) {
		int idx = 0;
		double min = 1.0 / n;
		while (alpha < getAlpha(idx) && min < alpha) {
			idx++;
		}
		return idx;
	}

	protected void backfit(Instances trainSet, double alpha, RegressionTree[] grove, double[][] rtPreds,
			double[] residualTrain) {
		Map<Integer, Integer> bagIndices = new HashMap<>();
		List<Integer> oobIndices = new ArrayList<>();
		Bagging.createBootstrapSample(trainSet, bagIndices, oobIndices);
		Instances bag = new Instances(trainSet.getAttributes(), trainSet.getTargetAttribute(), bagIndices.size());
		for (Integer idx : bagIndices.keySet()) {
			int weight = bagIndices.get(idx);
			Instance instance = trainSet.get(idx).clone();
			instance.setWeight(weight);
			bag.add(instance);
		}

		RegressionTreeLearner rtLearner = new RegressionTreeLearner();
		rtLearner.setConstructionMode(Mode.ALPHA_LIMITED);
		rtLearner.setAlpha(alpha);

		double prevRMSE = evalRMSE(oobIndices, residualTrain);
		for (;;) {
			for (int iter = 0; iter < grove.length; iter++) {
				int treeIdx = (iter + grove.length - 1) % grove.length;

				double[] treePreds = rtPreds[treeIdx];
				for (int i = 0; i < residualTrain.length; i++) {
					residualTrain[i] += treePreds[i];
					trainSet.get(i).setTarget(residualTrain[i]);
				}

				RegressionTree rt = rtLearner.build(bag);
				grove[treeIdx] = rt;

				for (int i = 0; i < residualTrain.length; i++) {
					double pred = rt.regress(trainSet.get(i));
					treePreds[i] = pred;
					residualTrain[i] -= pred;
				}
			}

			double currRMSE = evalRMSE(oobIndices, residualTrain);
			if (currRMSE == 0 || (prevRMSE - currRMSE) / prevRMSE <= 0.002) {
				break;
			} else {
				prevRMSE = currRMSE;
			}
		}

	}

	protected double regress(RegressionTree[] trees, Instance instance) {
		double pred = 0;
		for (RegressionTree rt : trees) {
			pred += rt.regress(instance);
		}
		return pred;
	}

	protected void runLayeredTraining(Instances trainSet, Instances validSet, int bStart, int bEnd, int tStart,
			int tEnd, int aStart, int aEnd, List<Double> alphas, PerformanceMatrix perfMatrix, ModelMatrix modelMatrix,
			PredictionMatrix predMatrix, double[] targetTrain, double[] targetValid) {
		final int n = trainSet.size();
		final int tLen = tEnd - tStart;
		double[] predictionValid = new double[validSet.size()];

		for (int b = bStart; b < bEnd; b++) {
			// The most recent predictions for regression trees
			double[][][] rtPreds = new double[tLen][tEnd][n];
			// The most recent residuals
			double[][] residualTrain = new double[tLen][n];
			for (int t = 0; t < tLen; t++) {
				double[] residual = residualTrain[t];
				for (int i = 0; i < n; i++) {
					residual[i] = targetTrain[i];
				}
			}

			if (aStart != 0) {
				for (int t = 0; t < tEnd; t++) {
					RegressionTree[] grove = modelMatrix.groves[t][aStart - 1].groves.get(b);
					update(trainSet, grove, rtPreds, residualTrain, t);
				}
			}

			if (verbose) {
				System.out.println("Iteration " + (b + 1) + " out of " + bEnd);
			}

			for (int a = aStart; a < aEnd; a++) {
				double alpha = alphas.get(a);

				if (verbose) {
					System.out.println("\tBuilding models with alpha = " + alpha);
				}

				for (int t = tStart; t < tEnd; t++) {
					int numTrees = t + 1;
					int tIdx = t - tStart;

					RegressionTree[] grove = new RegressionTree[numTrees];
					backfit(trainSet, alpha, grove, rtPreds[tIdx], residualTrain[tIdx]);
					modelMatrix.add(t, a, grove);

					// Update predictions
					double[] sumPredictionValid = predMatrix.sumPrediction[t][a];
					for (int i = 0; i < sumPredictionValid.length; i++) {
						sumPredictionValid[i] += regress(grove, validSet.get(i));
						predictionValid[i] = sumPredictionValid[i] / (b + 1);
					}
					perfMatrix.eval(t, a, b, predictionValid, targetValid);
				}
			}
		}
	}

	protected double evalRMSE(List<Integer> indices, double[] residual) {
		double rmse = 0;
		for (Integer idx : indices) {
			double d = residual[idx];
			rmse += d * d;
		}
		rmse = Math.sqrt(rmse / indices.size());
		return rmse;
	}

	protected void update(Instances trainSet, RegressionTree[] grove, double[][][] rtPreds, double[][] residualTrain,
			int maxTN) {
		for (int t = 0; t < maxTN; t++) {
			RegressionTree rt = grove[t];
			for (int i = 0; i < trainSet.size(); i++) {
				double pred = rt.regress(trainSet.get(i));
				rtPreds[maxTN][t][i] = pred;
				residualTrain[maxTN][i] -= pred;
			}
		}
	}

}
