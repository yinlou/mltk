package mltk.predictor.brt;

import java.util.Arrays;
import java.util.List;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.core.Attribute;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.NominalAttribute;
import mltk.core.io.InstancesReader;
import mltk.predictor.Learner;
import mltk.predictor.io.PredictorWriter;
import mltk.predictor.tree.RegressionTree;
import mltk.predictor.tree.RegressionTreeLearner;
import mltk.predictor.tree.RegressionTreeLearner.Mode;
import mltk.util.MathUtils;
import mltk.util.Permutation;
import mltk.util.Random;
import mltk.util.StatUtils;
import mltk.util.VectorUtils;

/**
 * Class for logit boost learner.
 * 
 * @author Yin Lou
 *
 */
public class LogitBoostLearner extends Learner {
	
	private boolean verbose;
	private int maxNumIters;
	private int maxNumLeaves;
	private double learningRate;
	private double alpha;
	
	/**
	 * Constructor.
	 */
	public LogitBoostLearner() {
		verbose = false;
		maxNumIters = 3500;
		maxNumLeaves = 100;
		learningRate = 1;
		alpha = 1;
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
	 * Builds a classifier.
	 * 
	 * @param trainSet the training set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param maxNumLeaves the maximum number of leaves.
	 * @return a classifier.
	 */
	public BRT buildClassifier(Instances trainSet, int maxNumIters, int maxNumLeaves) {
		Attribute classAttribute = trainSet.getTargetAttribute();
		if (classAttribute.getType() != Attribute.Type.NOMINAL) {
			throw new IllegalArgumentException("Class attribute must be nominal.");
		}
		NominalAttribute clazz = (NominalAttribute) classAttribute;
		final int numClasses = clazz.getStates().length;
		final int n = trainSet.size();
		final int baseClass = numClasses - 1;
		
		BRT brt = new BRT(numClasses);
		
		List<Attribute> attributes = trainSet.getAttributes();
		int limit = (int) (attributes.size() * alpha);
		int[] indices = new int[limit];
		Permutation perm = new Permutation(attributes.size());
		if (alpha < 1) {
			perm.permute();
		}
		
		// Backup targets and weights
		double[] target = new double[n];
		double[] weight = new double[n];
		for (int i = 0; i < n; i++) {
			Instance instance = trainSet.get(i);
			target[i] = instance.getTarget();
			weight[i] = instance.getWeight();
		}
		
		// Initialization
		double[][] pred = new double[n][numClasses];
		double[][] prob = new double[n][numClasses];
		int[][] r = new int[n][numClasses];
		for (int i = 0; i < n; i++) {
			for (int k = 0; k < numClasses; k++) {
				r[i][k] = MathUtils.indicator(target[i] == k);
				prob[i][k] = 1.0 / numClasses;
			}
		}
		
		RegressionTreeLearner rtLearner = new RegressionTreeLearner();
		rtLearner.setConstructionMode(Mode.NUM_LEAVES_LIMITED);
		rtLearner.setMaxNumLeaves(maxNumLeaves);
		
		for (int iter = 0; iter < maxNumIters; iter++) {
			// Prepare attributes
			if (alpha < 1) {
				int[] a = perm.getPermutation();
				for (int i = 0; i < indices.length; i++) {
					indices[i] = a[i];
				}
				Arrays.sort(indices);
				List<Attribute> attList = trainSet.getAttributes(indices);
				trainSet.setAttributes(attList);
			}
			
			for (int k = 0; k < numClasses - 1; k++) {
				// Prepare training set
				for (int i = 0; i < n; i++) {
					Instance instance = trainSet.get(i);
					double pk = prob[i][k];
					double pb = prob[i][baseClass];
					double t = r[i][k] - pk - (r[i][baseClass] - pb);
					double w = pb * (1 - pb) + pk * (1 - pk) + 2 * pk * pb;
					instance.setTarget(t);
					instance.setWeight(w);
				}
				
				RegressionTree rt = rtLearner.build(trainSet);
				if (learningRate != 1) {
					rt.multiply(learningRate);
				}
				brt.trees[k].add(rt);
				
				for (int i = 0; i < n; i++) {
					double p = rt.regress(trainSet.get(i));
					pred[i][k] += p;
				}
			}
			
			if (alpha < 1) {
				// Restore attributes
				trainSet.setAttributes(attributes);
			}
			
			// Update probabilities
			for (int i = 0; i < n; i++) {
				predictProbabilities(pred[i], prob[i]);
			}
			
			if (verbose) {
				double error = 0;
				for (int i = 0; i < n; i++) {
					double p = StatUtils.indexOfMax(prob[i]);
					if (p != target[i]) {
						error++;
					}
				}
				error /= n;
				//double rmse = StatUtils.rms(residualTrain);
				System.out.println("Iteration " + iter + ": " + error);
			}
		}
		
		// Restore targets and weights
		for (int i = 0; i < n; i++) {
			Instance instance = trainSet.get(i);
			instance.setTarget(target[i]);
			instance.setWeight(weight[i]);
		}
		
		return brt;
	}
	
	protected void predictProbabilities(double[] pred, double[] prob) {
		double max = StatUtils.max(pred);
		double sum = 0;
		for (int i = 0; i < prob.length; i++) {
			prob[i] = Math.exp(pred[i] - max);
			sum += prob[i];
		}
		VectorUtils.divide(prob, sum);
	}

	@Override
	public BRT build(Instances instances) {
		return buildClassifier(instances, maxNumIters, maxNumLeaves);
	}
	
	static class Options {
		
		@Argument(name = "-r", description = "attribute file path")
		String attPath = null;
		
		@Argument(name = "-t", description = "train set path", required = true)
		String trainPath = null;
		
		@Argument(name = "-o", description = "output model path")
		String outputModelPath = null;
		
		@Argument(name = "-c", description = "max number of leaves (default: 100)")
		int maxNumLeaves = 100;
		
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
	 * Usage: LogitBoostLearner
	 * -t	train set path
	 * -m	maximum number of iterations
	 * [-r]	attribute file path
	 * [-o]	output model path
	 * [-c]	max number of leaves (default: 100)
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
		CmdLineParser parser = new CmdLineParser(LogitBoostLearner.class, opts);
		try {
			parser.parse(args);
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}
		
		Random.getInstance().setSeed(opts.seed);
		
		Instances trainSet = InstancesReader.read(opts.attPath, opts.trainPath);
		
		LogitBoostLearner logitBoostLearner = new LogitBoostLearner();
		logitBoostLearner.setLearningRate(opts.learningRate);
		logitBoostLearner.setMaxNumIters(opts.maxNumIters);
		logitBoostLearner.setMaxNumLeaves(opts.maxNumLeaves);
		logitBoostLearner.setVerbose(true);
		
		long start = System.currentTimeMillis();
		BRT brt = logitBoostLearner.build(trainSet);
		long end = System.currentTimeMillis();
		System.out.println("Time: " + (end - start) / 1000.0);
		
		if (opts.outputModelPath != null) {
			PredictorWriter.write(brt, opts.outputModelPath);
		}
	}

}
