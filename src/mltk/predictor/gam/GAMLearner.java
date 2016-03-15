package mltk.predictor.gam;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.cmdline.options.HoldoutValidatedLearnerWithTaskOptions;
import mltk.core.Attribute;
import mltk.core.BinnedAttribute;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.NominalAttribute;
import mltk.core.Sampling;
import mltk.core.Attribute.Type;
import mltk.core.io.InstancesReader;
import mltk.predictor.BaggedEnsemble;
import mltk.predictor.BaggedEnsembleLearner;
import mltk.predictor.BoostedEnsemble;
import mltk.predictor.HoldoutValidatedLearner;
import mltk.predictor.Regressor;
import mltk.predictor.evaluation.Metric;
import mltk.predictor.evaluation.MetricFactory;
import mltk.predictor.evaluation.SimpleMetric;
import mltk.predictor.function.CompressionUtils;
import mltk.predictor.function.Function1D;
import mltk.predictor.function.LineCutter;
import mltk.predictor.io.PredictorWriter;
import mltk.util.OptimUtils;
import mltk.util.Random;

/**
 * Class for learning GAMs via gradient tree boosting.
 *
 * <p>
 * Reference:<br>
 * Y. Lou, R. Caruana and J. Gehrke. Intelligible models for classification and regression. In <i>Proceedings of the
 * 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)</i>, Beijing, China, 2012.
 * </p>
 *
 * @author Yin Lou
 *
 */
public class GAMLearner extends HoldoutValidatedLearner {

	static class Options extends HoldoutValidatedLearnerWithTaskOptions {

		@Argument(name = "-b", description = "base learner (default: tr:3:100)")
		String baseLearner = "tr:3:100";

		@Argument(name = "-m", description = "maximum number of iterations", required = true)
		int maxNumIters = -1;

		@Argument(name = "-s", description = "seed of the random number generator (default: 0)")
		long seed = 0L;

		@Argument(name = "-l", description = "learning rate (default: 0.01)")
		double learningRate = 0.01;

		@Argument(name = "-f", description = "feature importances path")
		String featPath = null;

		@Argument(name = "-w", description = "selected features path")
		String featPathOut = null;
	}

	/**
	 * <p>
	 *
	 * <pre>
	 * Usage: mltk.predictor.gam.GAMLearner
	 * -m	maximum number of iterations
	 * [-t]	train set path
	 * [-g]	task between classification (c) and regression (r) (default: r)
	 * [-v]	valid set path
	 * [-e]	evaluation metric (default: default metric of task)
	 * [-r]	attribute file path
	 * [-o]	output model path
	 * [-V]	verbose (default: true)
	 * [-b]	base learner (default: tr:3:100)
	 * [-s]	seed of the random number generator (default: 0)
	 * [-l]	learning rate (default: 0.01)
	 * [-f] path to the features info if feature selection (ex: featInfo.txt:N:500)
	 * [-w] path to save the selected features
	 * </pre>
	 *
	 * </p>
	 *
	 * @param args the command line arguments.
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(GAMLearner.class, opts);
		Task task = null;
		Metric metric = null;
		try {
			parser.parse(args);
			task = Task.get(opts.task);
			if (opts.metric == null) {
				metric = task.getDefaultMetric();
			} else {
				metric = MetricFactory.getMetric(opts.metric);
			}
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}

		Random.getInstance().setSeed(opts.seed);

		GAMLearner learner = new GAMLearner();

		Instances trainSet = InstancesReader.read(opts.attPath, opts.trainPath);
		if(opts.featPath != null) {
			System.out.println("Reading feature file...");
			String[] opts_feats = (opts.featPath).split(":");
			BufferedReader br = new BufferedReader(new FileReader(opts_feats[0]));
			Set<Integer> feats = new HashSet<Integer>();

			if(opts_feats[1].equals("N")) {
				feats = getFeaturesByRank(br, Integer.parseInt(opts_feats[2]));
			}
			else if(opts_feats[1].equals("T")) {
				feats = getFeaturesByValue(br, Double.parseDouble(opts_feats[2]));
			}
			else {
				System.out.println("ERROR: Could not read featInfo argument");
				feats = null;
			}

			System.out.println("Using " + feats.size() + " variables...");
			br.close();
			learner.setFeatures(feats);

			if(opts.featPathOut != null) {
				PrintWriter featOut = new PrintWriter(opts.featPathOut);
				StringBuilder sbuilder1 = new StringBuilder(feats.toString());
				// remove brackets
				sbuilder1.deleteCharAt(0);
				sbuilder1.deleteCharAt(sbuilder1.length()-1);
				featOut.println(sbuilder1.toString());

				List<Attribute> attributes = trainSet.getAttributes();
				StringBuilder sbuilder2 = new StringBuilder();
				for(int i : feats) {
					sbuilder2.append(attributes.get(i).getName() + ",");
				}
				// remove last ','
				sbuilder2.deleteCharAt(sbuilder2.length()-1);
				featOut.println(sbuilder2.toString());

				featOut.flush();
				featOut.close();
				System.out.println("Selected features saved");
			}
		}

		learner.setBaseLearner(opts.baseLearner);
		learner.setMaxNumIters(opts.maxNumIters);
		learner.setLearningRate(opts.learningRate);
		learner.setTask(task);
		learner.setMetric(metric);
		learner.setVerbose(opts.verbose);

		if (opts.validPath != null) {
			Instances validSet = InstancesReader.read(opts.attPath, opts.validPath);
			learner.setValidSet(validSet);
		}

		long start = System.currentTimeMillis();
		GAM gam = learner.build(trainSet);
		long end = System.currentTimeMillis();
		System.out.println("Time: " + (end - start) / 1000.0);

		if (opts.outputModelPath != null) {
			PredictorWriter.write(gam, opts.outputModelPath);
		}
	}

	private int baggingIters;
	private int maxNumIters;
	private int maxNumLeaves;
	private Task task;
	private double learningRate;
	private Set<Integer> features;

	/**
	 * Constructor.
	 */
	public GAMLearner() {
		verbose = false;
		baggingIters = 100;
		maxNumIters = -1;
		maxNumLeaves = 3;
		learningRate = 1;
		task = Task.REGRESSION;
		metric = task.getDefaultMetric();
		features = null;
	}

	/**
	 * Returns the first k features in the file given by br.
	 * Assume features are sorted in the file.
	 * @param br A BufferReader of the featureInfo file.
	 * @param k The number of features to read.
	 * @return The set of feature index.
	 * @throws IOException
	 */
	public static Set<Integer> getFeaturesByRank(BufferedReader br, int k) throws IOException {
		Set<Integer> feats = new HashSet<Integer>();
		// read first line: features
		String line = br.readLine();
		String[] data1 = line.split(",");

		for(int i=0; i < k; i++) {
			feats.add(Integer.parseInt(data1[i]));
		}
		return feats;
	}

	/**
	 * Returns all features that have a minimum importance in the file given by br.
	 * @param br A BufferReader of the featureInfo file.
	 * @param impThreshold The threshold below which features are disregarded.
	 * @return The set of feature index.
	 * @throws IOException
	 */
	public static Set<Integer> getFeaturesByValue(BufferedReader br, double impThreshold) throws IOException {
		Set<Integer> feats = new HashSet<Integer>();
		// read first line: features
		String line = br.readLine();
		String[] data1 = line.split(",");

		// read second line: importance
		line = br.readLine();
		String[] data2 = line.split(",");

		for(int k=0; k < data1.length; k++) {
			if(Double.parseDouble(data2[k]) > impThreshold) {
				feats.add(Integer.parseInt(data1[k]));
			}
		}
		return feats;
	}

	/**
	 * Returns the number of bagging iterations.
	 *
	 * @return the number of bagging iterations.
	 */
	public int getBaggingIters() {
		return baggingIters;
	}

	/**
	 * Sets the number of bagging iterations.
	 *
	 * @param baggingIters the bagging iterations.
	 */
	public void setBaggingIters(int baggingIters) {
		this.baggingIters = baggingIters;
	}

	/**
	 * Returns the maximum number of iterations.
	 *
	 * @return the maximum number of iterations.
	 */
	public int getMaxNumIters() {
		return maxNumIters;
	}

	/**
	 * Sets the maximum number of iterations.
	 *
	 * @param maxNumIters the maximum number of iterations.
	 */
	public void setMaxNumIters(int maxNumIters) {
		this.maxNumIters = maxNumIters;
	}

	/**
	 * Returns the maximum number of leaves.
	 *
	 * @return the maximum number of leaves.
	 */
	public int getMaxNumLeaves() {
		return maxNumLeaves;
	}

	/**
	 * Sets the maximum number of leaves.
	 *
	 * @param maxNumLeaves the maximum number of leaves.
	 */
	public void setMaxNumLeaves(int maxNumLeaves) {
		this.maxNumLeaves = maxNumLeaves;
	}

	/**
	 * Returns the learning rate.
	 *
	 * @return the learning rate.
	 */
	public double getLearningRate() {
		return learningRate;
	}

	/**
	 * Sets the learning rate.
	 *
	 * @param learningRate the learning rate.
	 */
	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	/**
	 * Returns the task of this learner.
	 *
	 * @return the task of this learner.
	 */
	public Task getTask() {
		return task;
	}

	/**
	 * Sets the task of this learner.
	 *
	 * @param task the task of this learner.
	 */
	public void setTask(Task task) {
		this.task = task;
	}

	/**
	 * Sets the features to use for this learner.
	 */
	public void setFeatures(Set<Integer> feats) {
		this.features = feats;
	}

	/**
	 * Sets the base learner.<br>
	 *
	 * @param option the option string.
	 */
	public void setBaseLearner(String option) {
		String[] opts = option.split(":");
		switch (opts[0]) {
			case "tr":
				int maxNumLeaves = Integer.parseInt(opts[1]);
				int baggingIters = Integer.parseInt(opts[2]);
				setMaxNumLeaves(maxNumLeaves);
				setBaggingIters(baggingIters);
				break;
			case "cs":
				break;
			default:
				break;
		}
	}

	/**
	 * Builds a classifier.
	 *
	 * @param trainSet the training set.
	 * @param validSet the validation set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param maxNumLeaves the maximum number of leaves.
	 * @return a classifier.
	 */
	public GAM buildClassifier(Instances trainSet, Instances validSet, int maxNumIters, int maxNumLeaves) {
		GAM gam = new GAM();

		// Backup targets
		double[] target = new double[trainSet.size()];
		for (int i = 0; i < target.length; i++) {
			target[i] = trainSet.get(i).getTarget();
		}

		List<Attribute> attributes = trainSet.getAttributes();
		List<BoostedEnsemble> regressors = new ArrayList<>(attributes.size());
		for (int i = 0; i < attributes.size(); i++) {
			regressors.add(new BoostedEnsemble());
		}

		// Create bags
		Instances[] bags = Sampling.createBags(trainSet, baggingIters);

		LineCutter lineCutter = new LineCutter(true);
		lineCutter.setNumIntervals(maxNumLeaves);
		BaggedEnsembleLearner learner = new BaggedEnsembleLearner(bags.length, lineCutter);

		// Initialize predictions and residuals
		double[] pTrain = new double[trainSet.size()];
		double[] rTrain = new double[trainSet.size()];
		OptimUtils.computePseudoResidual(pTrain, target, rTrain);
		double[] pValid = new double[validSet.size()];

		List<Double> measureList = new ArrayList<>(maxNumIters * attributes.size());

		// Gradient boosting
		for (int iter = 0; iter < maxNumIters; iter++) {
			for (int j = 0; j < attributes.size(); j++) {
				if(features == null || features.contains(j)) {
					// Derivitive to attribute k
					// Minimizes the loss function: log(1 + exp(-yF))
					for (int i = 0; i < trainSet.size(); i++) {
						trainSet.get(i).setTarget(rTrain[i]);
					}

					BoostedEnsemble boostedEnsemble = regressors.get(j);

					// Train model
					lineCutter.setAttributeIndex(j);
					BaggedEnsemble baggedEnsemble = learner.build(bags);
					Function1D func = CompressionUtils.compress(attributes.get(j).getIndex(), baggedEnsemble);
					if (learningRate != 1) {
						func.multiply(learningRate);
					}
					boostedEnsemble.add(func);
					baggedEnsemble = null;

					// Update predictions
					for (int i = 0; i < trainSet.size(); i++) {
						Instance instance = trainSet.get(i);
						double pred = func.regress(instance);
						pTrain[i] += pred;
						rTrain[i] = OptimUtils.getPseudoResidual(pTrain[i], target[i]);
					}
					for (int i = 0; i < validSet.size(); i++) {
						Instance instance = validSet.get(i);
						double pred = func.regress(instance);
						pValid[i] += pred;
					}

					double measure = metric.eval(pValid, validSet);
					measureList.add(measure);
					if (verbose) {
						System.out.println("Iteration " + iter + " Feature " + j + ": " + measure);
					}
				}
			}
		}

		// Search the best model on validation set
		int idx = metric.searchBestMetricValueIndex(measureList);

		// Remove trees
		int n = idx / attributes.size();
		int m = idx % attributes.size();
		for (int k = 0; k < regressors.size(); k++) {
			BoostedEnsemble boostedEnsemble = regressors.get(k);
			for (int i = boostedEnsemble.size(); i > n + 1; i--) {
				boostedEnsemble.removeLast();
			}
			if (k > m) {
				boostedEnsemble.removeLast();
			}
		}

		// Restore targets
		for (int i = 0; i < target.length; i++) {
			trainSet.get(i).setTarget(target[i]);
		}

		// Compress model
		for (int i = 0; i < regressors.size(); i++) {
			BoostedEnsemble boostedEnsemble = regressors.get(i);
			Attribute attribute = attributes.get(i);
			int attIndex = attribute.getIndex();
			Function1D function = CompressionUtils.compress(attIndex, boostedEnsemble);
			Regressor regressor = function;
			if (attribute.getType() == Type.BINNED) {
				int l = ((BinnedAttribute) attribute).getNumBins();
				regressor = CompressionUtils.convert(l, function);
			} else if (attribute.getType() == Type.NOMINAL) {
				int l = ((NominalAttribute) attribute).getCardinality();
				regressor = CompressionUtils.convert(l, function);
			}
			gam.add(new int[] { attIndex }, regressor);
		}

		return gam;
	}

	/**
	 * Builds a classifier.
	 *
	 * @param trainSet the training set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param maxNumLeaves the maximum number of leaves.
	 * @return a classifier.
	 */
	public GAM buildClassifier(Instances trainSet, int maxNumIters, int maxNumLeaves) {
		GAM gam = new GAM();
		SimpleMetric simpleMetric = (SimpleMetric) metric;

		// Backup targets
		double[] target = new double[trainSet.size()];
		for (int i = 0; i < target.length; i++) {
			target[i] = trainSet.get(i).getTarget();
		}

		List<Attribute> attributes = trainSet.getAttributes();
		List<BoostedEnsemble> regressors = new ArrayList<>(attributes.size());
		for (int i = 0; i < attributes.size(); i++) {
			regressors.add(new BoostedEnsemble());
		}

		// Create bags
		Instances[] bags = Sampling.createBags(trainSet, baggingIters);

		LineCutter lineCutter = new LineCutter(true);
		lineCutter.setNumIntervals(maxNumLeaves);
		BaggedEnsembleLearner learner = new BaggedEnsembleLearner(bags.length, lineCutter);

		// Initialize predictions and residuals
		double[] pTrain = new double[trainSet.size()];
		double[] rTrain = new double[trainSet.size()];
		OptimUtils.computePseudoResidual(pTrain, target, rTrain);

		// Gradient boosting
		for (int iter = 0; iter < maxNumIters; iter++) {
			for (int j = 0; j < attributes.size(); j++) {
				// Derivitive to attribute k
				// Minimizes the loss function: log(1 + exp(-yF))
				for (int i = 0; i < trainSet.size(); i++) {
					trainSet.get(i).setTarget(rTrain[i]);
				}

				BoostedEnsemble boostedEnsemble = regressors.get(j);

				// Train model
				lineCutter.setAttributeIndex(j);
				BaggedEnsemble baggedEnsemble = learner.build(bags);
				Function1D func = CompressionUtils.compress(attributes.get(j).getIndex(), baggedEnsemble);
				if (learningRate != 1) {
					func.multiply(learningRate);
				}
				boostedEnsemble.add(func);
				baggedEnsemble = null;

				// Update predictions
				for (int i = 0; i < trainSet.size(); i++) {
					Instance instance = trainSet.get(i);
					double pred = func.regress(instance);
					pTrain[i] += pred;
					rTrain[i] = OptimUtils.getPseudoResidual(pTrain[i], target[i]);
				}

				double measure = simpleMetric.eval(pTrain, target);
				if (verbose) {
					System.out.println("Iteration " + iter + " Feature " + j + ": " + measure);
				}
			}
		}

		// Restore targets
		for (int i = 0; i < target.length; i++) {
			trainSet.get(i).setTarget(target[i]);
		}

		// Compress model
		for (int i = 0; i < regressors.size(); i++) {
			BoostedEnsemble boostedEnsemble = regressors.get(i);
			Attribute attribute = attributes.get(i);
			int attIndex = attribute.getIndex();
			Function1D function = CompressionUtils.compress(attIndex, boostedEnsemble);
			Regressor regressor = function;
			if (attribute.getType() == Type.BINNED) {
				int l = ((BinnedAttribute) attribute).getNumBins();
				regressor = CompressionUtils.convert(l, function);
			} else if (attribute.getType() == Type.NOMINAL) {
				int l = ((NominalAttribute) attribute).getCardinality();
				regressor = CompressionUtils.convert(l, function);
			}
			gam.add(new int[] { attIndex }, regressor);
		}

		return gam;
	}

	/**
	 * Builds a regressor.
	 *
	 * @param trainSet the training set.
	 * @param validSet the validation set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param maxNumLeaves the maximum number of leaves.
	 * @return a regressor.
	 */
	public GAM buildRegressor(Instances trainSet, Instances validSet, int maxNumIters, int maxNumLeaves) {
		GAM gam = new GAM();

		List<Attribute> attributes = trainSet.getAttributes();
		List<BoostedEnsemble> regressors = new ArrayList<>(attributes.size());
		for (int i = 0; i < attributes.size(); i++) {
			regressors.add(new BoostedEnsemble());
		}

		// Backup targets
		double[] target = new double[trainSet.size()];
		for (int i = 0; i < target.length; i++) {
			target[i] = trainSet.get(i).getTarget();
		}

		// Create bags
		Instances[] bags = Sampling.createBags(trainSet, baggingIters);

		LineCutter lineCutter = new LineCutter();
		lineCutter.setNumIntervals(maxNumLeaves);
		BaggedEnsembleLearner learner = new BaggedEnsembleLearner(bags.length, lineCutter);

		// Initialize predictions and residuals
		double[] rTrain = new double[trainSet.size()];
		double[] pValid = new double[validSet.size()];
		double[] rValid = new double[validSet.size()];
		for (int i = 0; i < trainSet.size(); i++) {
			Instance instance = trainSet.get(i);
			rTrain[i] = instance.getTarget();
		}
		for (int i = 0; i < validSet.size(); i++) {
			Instance instance = validSet.get(i);
			rValid[i] = instance.getTarget();
		}

		List<Double> measureList = new ArrayList<>(maxNumIters * attributes.size());

		// Gradient boosting
		for (int iter = 0; iter < maxNumIters; iter++) {
			for (int j = 0; j < attributes.size(); j++) {
				// Derivative to attribute k
				// Equivalent to residual
				BoostedEnsemble boostedEnsemble = regressors.get(j);
				// Prepare training set
				for (int i = 0; i < rTrain.length; i++) {
					trainSet.get(i).setTarget(rTrain[i]);
				}
				// Train model
				lineCutter.setAttributeIndex(j);
				BaggedEnsemble baggedEnsemble = learner.build(bags);
				Function1D func = CompressionUtils.compress(attributes.get(j).getIndex(), baggedEnsemble);
				if (learningRate != 1) {
					func.multiply(learningRate);
				}
				boostedEnsemble.add(func);
				baggedEnsemble = null;

				// Update residuals
				for (int i = 0; i < rTrain.length; i++) {
					Instance instance = trainSet.get(i);
					double pred = func.regress(instance);
					rTrain[i] -= pred;
				}
				for (int i = 0; i < rValid.length; i++) {
					Instance instance = validSet.get(i);
					double pred = func.regress(instance);
					pValid[i] += pred;
					rValid[i] -= pred;
				}

				double measure = metric.eval(pValid, validSet);
				measureList.add(measure);
				if (verbose) {
					System.out.println("Iteration " + iter + " Feature " + j + ": " + measure);
				}
			}
		}

		// Search the best model on validation set
		int idx = metric.searchBestMetricValueIndex(measureList);

		// Prune tree ensembles
		int n = idx / attributes.size();
		int m = idx % attributes.size();
		for (int k = 0; k < regressors.size(); k++) {
			BoostedEnsemble boostedEnsemble = regressors.get(k);
			for (int i = boostedEnsemble.size(); i > n + 1; i--) {
				boostedEnsemble.removeLast();
			}
			if (k > m) {
				boostedEnsemble.removeLast();
			}
		}

		// Restore targets
		for (int i = 0; i < target.length; i++) {
			trainSet.get(i).setTarget(target[i]);
		}

		// Compress model
		for (int i = 0; i < regressors.size(); i++) {
			BoostedEnsemble boostedEnsemble = regressors.get(i);
			Attribute attribute = attributes.get(i);
			int attIndex = attribute.getIndex();
			Function1D function = CompressionUtils.compress(attIndex, boostedEnsemble);
			Regressor regressor = function;
			if (attribute.getType() == Type.BINNED) {
				int l = ((BinnedAttribute) attribute).getNumBins();
				regressor = CompressionUtils.convert(l, function);
			} else if (attribute.getType() == Type.NOMINAL) {
				int l = ((NominalAttribute) attribute).getCardinality();
				regressor = CompressionUtils.convert(l, function);
			}
			gam.add(new int[] { attIndex }, regressor);
		}

		return gam;
	}

	/**
	 * Builds a regressor.
	 *
	 * @param trainSet the training set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param maxNumLeaves the maximum number of leaves.
	 * @return a regressor.
	 */
	public GAM buildRegressor(Instances trainSet, int maxNumIters, int maxNumLeaves) {
		GAM gam = new GAM();
		SimpleMetric simpleMetric = (SimpleMetric) metric;

		List<Attribute> attributes = trainSet.getAttributes();
		List<BoostedEnsemble> regressors = new ArrayList<>(attributes.size());
		for (int i = 0; i < attributes.size(); i++) {
			regressors.add(new BoostedEnsemble());
		}

		// Backup targets
		double[] target = new double[trainSet.size()];
		for (int i = 0; i < target.length; i++) {
			target[i] = trainSet.get(i).getTarget();
		}

		// Create bags
		Instances[] bags = Sampling.createBags(trainSet, baggingIters);

		LineCutter lineCutter = new LineCutter();
		lineCutter.setNumIntervals(maxNumLeaves);
		BaggedEnsembleLearner learner = new BaggedEnsembleLearner(bags.length, lineCutter);

		// Initialize predictions and residuals
		double[] pTrain = new double[trainSet.size()];
		double[] rTrain = new double[trainSet.size()];
		for (int i = 0; i < trainSet.size(); i++) {
			Instance instance = trainSet.get(i);
			rTrain[i] = instance.getTarget();
		}

		// Gradient boosting
		for (int iter = 0; iter < maxNumIters; iter++) {
			for (int j = 0; j < attributes.size(); j++) {
				// Derivative to attribute k
				// Equivalent to residual
				BoostedEnsemble boostedEnsemble = regressors.get(j);
				// Prepare training set
				for (int i = 0; i < rTrain.length; i++) {
					trainSet.get(i).setTarget(rTrain[i]);
				}
				// Train model
				lineCutter.setAttributeIndex(j);
				BaggedEnsemble baggedEnsemble = learner.build(bags);
				Function1D func = CompressionUtils.compress(attributes.get(j).getIndex(), baggedEnsemble);
				if (learningRate != 1) {
					func.multiply(learningRate);
				}
				boostedEnsemble.add(func);
				baggedEnsemble = null;

				// Update residuals
				for (int i = 0; i < rTrain.length; i++) {
					Instance instance = trainSet.get(i);
					double pred = func.regress(instance);
					pTrain[i] += pred;
					rTrain[i] -= pred;
				}

				double measure = simpleMetric.eval(pTrain, target);
				if (verbose) {
					System.out.println("Iteration " + iter + " Feature " + j + ": " + measure);
				}
			}
		}

		// Restore targets
		for (int i = 0; i < target.length; i++) {
			trainSet.get(i).setTarget(target[i]);
		}

		// Compress model
		for (int i = 0; i < regressors.size(); i++) {
			BoostedEnsemble boostedEnsemble = regressors.get(i);
			Attribute attribute = attributes.get(i);
			int attIndex = attribute.getIndex();
			Function1D function = CompressionUtils.compress(attIndex, boostedEnsemble);
			Regressor regressor = function;
			if (attribute.getType() == Type.BINNED) {
				int l = ((BinnedAttribute) attribute).getNumBins();
				regressor = CompressionUtils.convert(l, function);
			} else if (attribute.getType() == Type.NOMINAL) {
				int l = ((NominalAttribute) attribute).getCardinality();
				regressor = CompressionUtils.convert(l, function);
			}
			gam.add(new int[] { attIndex }, regressor);
		}

		return gam;
	}

	@Override
	public GAM build(Instances instances) {
		GAM gam = null;
		if (maxNumIters < 0) {
			maxNumIters = 20;
		}
		if (metric == null) {
			metric = task.getDefaultMetric();
		}
		switch (task) {
			case REGRESSION:
				if (validSet != null) {
					gam = buildRegressor(instances, validSet, maxNumIters, maxNumLeaves);
				} else {
					gam = buildRegressor(instances, maxNumIters, maxNumLeaves);
				}
				break;
			case CLASSIFICATION:
				if (validSet != null) {
					gam = buildClassifier(instances, validSet, maxNumIters, maxNumLeaves);
				} else {
					gam = buildClassifier(instances, maxNumIters, maxNumLeaves);
				}
				break;
			default:
				break;
		}
		return gam;
	}

}
