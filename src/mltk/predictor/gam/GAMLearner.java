package mltk.predictor.gam;

import java.util.ArrayList;
import java.util.List;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.core.Attribute;
import mltk.core.BinnedAttribute;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.NominalAttribute;
import mltk.core.Attribute.Type;
import mltk.core.io.InstancesReader;
import mltk.predictor.BaggedEnsemble;
import mltk.predictor.BaggedEnsembleLearner;
import mltk.predictor.Bagging;
import mltk.predictor.BoostedEnsemble;
import mltk.predictor.Learner;
import mltk.predictor.Regressor;
import mltk.predictor.function.CompressionUtils;
import mltk.predictor.function.Function1D;
import mltk.predictor.function.LineCutter;
import mltk.predictor.io.PredictorWriter;
import mltk.util.OptimUtils;
import mltk.util.Random;
import mltk.util.StatUtils;

/**
 * Class for learning GAMs via gradient tree boosting.
 * 
 * <p>
 * Reference:<br>
 * Y. Lou, R. Caruana and J. Gehrke. Intelligible models for classification and 
 * regression. In <i>Proceedings of the 18th ACM SIGKDD International Conference 
 * on Knowledge Discovery and Data Mining (KDD)</i>, Beijing, China, 2012.
 * </p>
 * @author Yin Lou
 *
 */
public class GAMLearner extends Learner {
	
	private boolean verbose;
	private int baggingIters;
	private int maxNumIters;
	private int maxNumLeaves;
	private Task task;
	private double learningRate;
	private Instances validSet;
	
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
	}

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
	 * Returns the validation set.
	 * 
	 * @return the validation set.
	 */
	public Instances getValidSet() {
		return validSet;
	}
	
	/**
	 * Sets the validation set.
	 * 
	 * @param validSet the validation set.
	 */
	public void setValidSet(Instances validSet) {
		this.validSet = validSet;
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
	public GAM buildClassifier(Instances trainSet, Instances validSet, 
			int maxNumIters, int maxNumLeaves) {
		GAM gam = new GAM();

		// Backup targets
		int[] target = new int[trainSet.size()];
		for (int i = 0; i < target.length; i++) {
			target[i] = (int) trainSet.get(i).getTarget();
		}

		List<Attribute> attributes = trainSet.getAttributes();
		List<BoostedEnsemble> regressors = new ArrayList<>(attributes.size());
		for (int i = 0; i < attributes.size(); i++) {
			regressors.add(new BoostedEnsemble());
		}
		
		// Create bags
		Instances[] bags = Bagging.createBags(trainSet, baggingIters);
		
		LineCutter lineCutter = new LineCutter(true);
		lineCutter.setNumIntervals(maxNumLeaves);
		BaggedEnsembleLearner learner = 
				new BaggedEnsembleLearner(bags.length, lineCutter);

		// Initialize predictions
		double[] predictionTrain = new double[trainSet.size()];
		double[] predictionValid = new double[validSet.size()];
		
		List<Double> errorList = new ArrayList<>(maxNumIters);

		// Gradient boosting
		for (int iter = 0; iter < maxNumIters; iter++) {
			int k = iter % attributes.size();
			// Derivitive to attribute k
			// Minimizes the loss function: log(1 + exp(-yF))
			for (int i = 0; i < trainSet.size(); i++) {
				double r = OptimUtils.getPseudoResidual(predictionTrain[i], target[i]);
				trainSet.get(i).setTarget(r);
			}

			BoostedEnsemble boostedEnsemble = regressors.get(k);

			// Train model
			lineCutter.setAttributeIndex(k);
			BaggedEnsemble baggedEnsemble = learner.build(bags);
			if (learningRate != 1) {
				for (int i = 0; i < baggedEnsemble.size(); i++) {
					Function1D func = (Function1D) baggedEnsemble.get(i);
					func.multiply(learningRate);
				}
			}
			boostedEnsemble.add(baggedEnsemble);

			// Update predictions
			for (int i = 0; i < trainSet.size(); i++) {
				Instance instance = trainSet.get(i);
				double pred = baggedEnsemble.regress(instance);
				predictionTrain[i] += pred;
			}
			for (int i = 0; i < validSet.size(); i++) {
				Instance instance = validSet.get(i);
				double pred = baggedEnsemble.regress(instance);
				predictionValid[i] += pred;
			}

			double error = 0.0;
			for (int i = 0; i < validSet.size(); i++) {
				int cls = (int) validSet.get(i).getTarget();
				int pred = predictionValid[i] >= 0 ? 1 : 0;
				if (pred != cls) {
					error++;
				}
			}
			error /= validSet.size();
			errorList.add(error);
			if (verbose) {
				System.out.println("Iteration " + iter + " Feature " + k + ": " + error);
			}
		}

		// Search the best model on validation set
		double min = Double.POSITIVE_INFINITY;
		int idx = -1;
		for (int i = 0; i < errorList.size(); i++) {
			if (errorList.get(i) < min) {
				min = errorList.get(i);
				idx = i;
			}
		}
		
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
			gam.add(new int[] {attIndex}, regressor);
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

		// Backup targets
		int[] target = new int[trainSet.size()];
		for (int i = 0; i < target.length; i++) {
			target[i] = (int) trainSet.get(i).getTarget();
		}

		List<Attribute> attributes = trainSet.getAttributes();
		List<BoostedEnsemble> regressors = new ArrayList<>(attributes.size());
		for (int i = 0; i < attributes.size(); i++) {
			regressors.add(new BoostedEnsemble());
		}
		
		// Create bags
		Instances[] bags = Bagging.createBags(trainSet, baggingIters);
		
		LineCutter lineCutter = new LineCutter(true);
		lineCutter.setNumIntervals(maxNumLeaves);
		BaggedEnsembleLearner learner = 
				new BaggedEnsembleLearner(bags.length, lineCutter);

		// Initialize predictions
		double[] predictionTrain = new double[trainSet.size()];

		// Gradient boosting
		for (int iter = 0; iter < maxNumIters; iter++) {
			int k = iter % attributes.size();
			// Derivitive to attribute k
			// Minimizes the loss function: log(1 + exp(-2yF))
			for (int i = 0; i < trainSet.size(); i++) {
				double r = OptimUtils.getPseudoResidual(predictionTrain[i], target[i]);
				trainSet.get(i).setTarget(r);
			}

			BoostedEnsemble boostedEnsemble = regressors.get(k);

			// Train model
			lineCutter.setAttributeIndex(k);
			BaggedEnsemble baggedEnsemble = learner.build(bags);
			if (learningRate != 1) {
				for (int i = 0; i < baggedEnsemble.size(); i++) {
					Function1D func = (Function1D) baggedEnsemble.get(i);
					func.multiply(learningRate);
				}
			}
			boostedEnsemble.add(baggedEnsemble);

			// Update predictions
			for (int i = 0; i < trainSet.size(); i++) {
				Instance instance = trainSet.get(i);
				double pred = baggedEnsemble.regress(instance);
				predictionTrain[i] += pred;
			}

			double error = 0.0;
			for (int i = 0; i < target.length; i++) {
				int cls = (int) target[i];
				int pred = predictionTrain[i] >= 0 ? 1 : 0;
				if (pred != cls) {
					error++;
				}
			}
			error /= trainSet.size();
			if (verbose) {
				System.out.println("Iteration " + iter + " Feature " + k + ": " + error);
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
			gam.add(new int[] {attIndex}, regressor);
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
	public GAM buildRegressor(Instances trainSet, Instances validSet, 
			int maxNumIters, int maxNumLeaves) {
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
		Instances[] bags = Bagging.createBags(trainSet, baggingIters);
		
		LineCutter lineCutter = new LineCutter();
		lineCutter.setNumIntervals(maxNumLeaves);
		BaggedEnsembleLearner learner = 
				new BaggedEnsembleLearner(bags.length, lineCutter);
		
		// Initialize residuals
		double[] residualTrain = new double[trainSet.size()];
		double[] residualValid = new double[validSet.size()];
		for (int i = 0; i < trainSet.size(); i++) {
			Instance instance = trainSet.get(i);
			residualTrain[i] = instance.getTarget();
		}
		for (int i = 0; i < validSet.size(); i++) {
			Instance instance = validSet.get(i);
			residualValid[i] = instance.getTarget();
		}
		
		List<Double> rmseList = new ArrayList<>(maxNumIters);
		
		// Gradient boosting
		for (int iter = 0; iter < maxNumIters; iter++) {
			int k = iter % attributes.size();
			// Derivative to attribute k
			// Equivalent to residual
			BoostedEnsemble boostedEnsemble = regressors.get(k);
			// Prepare training set
			for (int i = 0; i < residualTrain.length; i++) {
				trainSet.get(i).setTarget(residualTrain[i]);
			}
			// Train model
			lineCutter.setAttributeIndex(k);
			BaggedEnsemble baggedEnsemble = learner.build(bags);
			if (learningRate != 1) {
				for (int i = 0; i < baggedEnsemble.size(); i++) {
					Function1D func = (Function1D) baggedEnsemble.get(i);
					func.multiply(learningRate);
				}
			}
			boostedEnsemble.add(baggedEnsemble);
			
			// Update residuals
			for (int j = 0; j < residualTrain.length; j++) {
				Instance instance = trainSet.get(j);
				double pred = baggedEnsemble.regress(instance);
				residualTrain[j] -= pred;
			}
			for (int j = 0; j < residualValid.length; j++) {
				Instance instance = validSet.get(j);
				double pred = baggedEnsemble.regress(instance);
				residualValid[j] -= pred;
			}
			
			double rmse = StatUtils.rms(residualValid);
			rmseList.add(rmse);
			if (verbose) {
				System.out.println("Iteration " + iter + " Feature " + k + ": " 
								+ StatUtils.rms(residualTrain) + " " + rmse);
			}
		}
	
		// Search the best model on validation set
		double min = Double.POSITIVE_INFINITY;
		int idx = -1;
		for (int i = 0; i < rmseList.size(); i++) {
			if (rmseList.get(i) < min) {
				min = rmseList.get(i);
				idx = i;
			}
		}
		
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
			gam.add(new int[] {attIndex}, regressor);
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
		Instances[] bags = Bagging.createBags(trainSet, baggingIters);
		
		LineCutter lineCutter = new LineCutter();
		lineCutter.setNumIntervals(maxNumLeaves);
		BaggedEnsembleLearner learner = 
				new BaggedEnsembleLearner(bags.length, lineCutter);
		
		// Initialize residuals
		double[] residualTrain = new double[trainSet.size()];
		for (int i = 0; i < trainSet.size(); i++) {
			Instance instance = trainSet.get(i);
			residualTrain[i] = instance.getTarget();
		}
		
		// Gradient boosting
		for (int iter = 0; iter < maxNumIters; iter++) {
			int k = iter % attributes.size();
			// Derivative to attribute k
			// Equivalent to residual
			BoostedEnsemble boostedEnsemble = regressors.get(k);
			// Prepare training set
			for (int i = 0; i < residualTrain.length; i++) {
				trainSet.get(i).setTarget(residualTrain[i]);
			}
			// Train model
			lineCutter.setAttributeIndex(k);
			BaggedEnsemble baggedEnsemble = learner.build(bags);
			if (learningRate != 1) {
				for (int i = 0; i < baggedEnsemble.size(); i++) {
					Function1D func = (Function1D) baggedEnsemble.get(i);
					func.multiply(learningRate);
				}
			}
			boostedEnsemble.add(baggedEnsemble);
			
			// Update residuals
			for (int i = 0; i < residualTrain.length; i++) {
				Instance instance = trainSet.get(i);
				double pred = baggedEnsemble.regress(instance);
				residualTrain[i] -= pred;	
			}
			
			double rmse = StatUtils.rms(residualTrain);
			if (verbose) {
				System.out.println("Iteration " + iter + " Feature " + k + ": " + rmse);
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
			gam.add(new int[] {attIndex}, regressor);
		}
		
		return gam;
	}

	@Override
	public GAM build(Instances instances) {
		GAM gam = null;
		if (maxNumIters < 0) {
			maxNumIters = instances.getAttributes().size() * 20;
		}
		switch (task) {
			case REGRESSION:
				if (validSet != null) {
					gam = buildRegressor(instances, validSet, maxNumIters, 
							maxNumLeaves);
				} else {
					gam = buildRegressor(instances, maxNumIters, maxNumLeaves);
				}
				break;
			case CLASSIFICATION:
				if (validSet != null) {
					gam = buildClassifier(instances, validSet, maxNumIters, 
							maxNumLeaves);
				} else {
					gam = buildClassifier(instances, maxNumIters, maxNumLeaves);
				}
				break;
			default:
				break;
		}
		return gam;
	}
	
	static class Options {
		
		@Argument(name = "-r", description = "attribute file path")
		String attPath = null;
		
		@Argument(name = "-t", description = "train set path", required = true)
		String trainPath = null;
		
		@Argument(name = "-v", description = "valid set path")
		String validPath = null;
		
		@Argument(name = "-o", description = "output model path")
		String outputModelPath = null;
		
		@Argument(name = "-g", description = "task between classification (c) and regression (r) (default: r)")
		String task = "r";
		
		@Argument(name = "-b", description = "base learner (default: tr:3:100)")
		String baseLearner = "tr:3:100";
		
		@Argument(name = "-m", description = "maximum number of iterations", required = true)
		int maxNumIters = -1;
		
		@Argument(name = "-s", description = "seed of the random number generator (default: 0)")
		long seed = 0L;
		
		@Argument(name = "-l", description = "learning rate (default: 0.01)")
		double learningRate = 0.01;
		
	}
	
	/**
	 * <p>
	 * <pre>
	 * Usage: GAMLearner
	 * -t	train set path
	 * -m	maximum number of iterations
	 * [-r]	attribute file path
	 * [-v]	valid set path
	 * [-o]	output model path
	 * [-g]	task between classification (c) and regression (r) (default: r)
	 * [-b]	base learner (default: tr:3:100)
	 * [-s]	seed of the random number generator (default: 0)
	 * [-l]	learning rate (default: 0.01)
	 * </pre>
	 * </p>
	 * 
	 * @param args the command line arguments.
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(GAMLearner.class, opts);
		Task task = null;
		try {
			parser.parse(args);
			task = Task.getEnum(opts.task);
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}
		
		Random.getInstance().setSeed(opts.seed);

		Instances trainSet = InstancesReader.read(opts.attPath, opts.trainPath);
		
		GAMLearner gamLearner = new GAMLearner();
		gamLearner.setBaseLearner(opts.baseLearner);
		gamLearner.setMaxNumIters(opts.maxNumIters);
		gamLearner.setLearningRate(opts.learningRate);
		gamLearner.setTask(task);
		gamLearner.setVerbose(true);
		
		if (opts.validPath != null) {
			Instances validSet = InstancesReader.read(opts.attPath, opts.validPath);
			gamLearner.setValidSet(validSet);
		}
		
		long start = System.currentTimeMillis();
		GAM gam = gamLearner.build(trainSet);
		long end = System.currentTimeMillis();
		System.out.println("Time: " + (end - start) / 1000.0);
		
		if (opts.outputModelPath != null) {
			PredictorWriter.write(gam, opts.outputModelPath);
		}
	}

}
