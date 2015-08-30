package mltk.predictor.gam;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.cmdline.options.LearnerWithTaskOptions;
import mltk.core.Instances;
import mltk.core.io.InstancesReader;
import mltk.core.processor.OneHotEncoder;
import mltk.predictor.Learner;
import mltk.predictor.glm.GLM;
import mltk.predictor.glm.RidgeLearner;
import mltk.predictor.io.PredictorWriter;

/**
 * Class for learning scorecard models. Scorecard models are a special kind of
 * generalized additive models where scores for each state of the feature are 
 * learned. 
 * 
 * @author Yin Lou
 *
 */
public class ScorecardModelLearner extends Learner {
	
	static class Options extends LearnerWithTaskOptions {

		@Argument(name = "-m", description = "maximum number of iterations", required = true)
		int maxNumIters = -1;

		@Argument(name = "-l", description = "lambda (default: 0)")
		double lambda = 0;

	}

	/**
	 * <p>
	 * 
	 * <pre>
	 * Usage: mltk.predictor.gam.ScorecardModelLearner
	 * -t	train set path
	 * -m	maximum number of iterations
	 * [-g]	task between classification (c) and regression (r) (default: r)
	 * [-r]	attribute file path
	 * [-o]	output model path
	 * [-V]	verbose (default: true)
	 * [-l]	lambda (default: 0)
	 * </pre>
	 * 
	 * </p>
	 * 
	 * @param args the command line arguments.
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(ScorecardModelLearner.class, opts);
		Task task = null;
		try {
			parser.parse(args);
			task = Task.getEnum(opts.task);
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}

		Instances trainSet = InstancesReader.read(opts.attPath, opts.trainPath);

		ScorecardModelLearner learner = new ScorecardModelLearner();
		learner.setMaxNumIters(opts.maxNumIters);
		learner.setLambda(opts.lambda);
		learner.setTask(task);
		learner.setVerbose(opts.verbose);

		long start = System.currentTimeMillis();
		GAM gam = learner.build(trainSet);
		long end = System.currentTimeMillis();
		System.out.println("Time: " + (end - start) / 1000.0);

		if (opts.outputModelPath != null) {
			PredictorWriter.write(gam, opts.outputModelPath);
		}
	}
	
	private int maxNumIters;
	private double lambda;
	private Task task;
	private OneHotEncoder encoder;
	
	/**
	 * Constructor.
	 */
	public ScorecardModelLearner() {
		verbose = false;
		maxNumIters = -1;
		encoder = new OneHotEncoder();
		lambda = 0;
		task = Task.REGRESSION;
	}
	
	/**
	 * Returns the lambda.
	 * 
	 * @return the lambda.
	 */
	public double getLambda() {
		return lambda;
	}
	
	/**
	 * Sets the lambda.
	 * 
	 * @param lambda the lambda.
	 */
	public void setLambda(double lambda) {
		this.lambda = lambda;
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
	 * Builds a classifier.
	 * 
	 * @param trainSet the training set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the L2 regularization parameter.
	 * @return a classifier.
	 */
	public GAM buildClassifier(Instances trainSet, int maxNumIters, double lambda) {
		Instances trainSetNew = encoder.process(trainSet);
		
		RidgeLearner learner = new RidgeLearner();
		learner.setTask(Task.CLASSIFICATION);
		learner.setLambda(lambda);
		learner.setVerbose(verbose);
		learner.setMaxNumIters(maxNumIters);
		
		GLM glm = learner.build(trainSetNew);
		
		return GAMUtils.getGAM(glm, trainSet.getAttributes());
	}
	
	/**
	 * Builds a regressor.
	 * 
	 * @param trainSet the training set.
	 * @param maxNumIters the maximum number of iterations.
	 * @param lambda the L2 regularization parameter.
	 * @return a regressor.
	 */
	public GAM buildRegressor(Instances trainSet, int maxNumIters, double lambda) {
		Instances trainSetNew = encoder.process(trainSet);
		
		RidgeLearner learner = new RidgeLearner();
		learner.setTask(Task.REGRESSION);
		learner.setLambda(lambda);
		learner.setVerbose(verbose);
		learner.setMaxNumIters(maxNumIters);
		
		GLM glm = learner.build(trainSetNew);
		
		return GAMUtils.getGAM(glm, trainSet.getAttributes());
	}

	@Override
	public GAM build(Instances instances) {
		GAM gam = null;
		if (maxNumIters < 0) {
			maxNumIters = instances.dimension() * 20;
		}
		switch (task) {
			case REGRESSION:
				gam = buildRegressor(instances, maxNumIters, lambda);
				break;
			case CLASSIFICATION:
				gam = buildClassifier(instances, maxNumIters, lambda);
				break;
			default:
				break;
		}
		return gam;
	}

}
