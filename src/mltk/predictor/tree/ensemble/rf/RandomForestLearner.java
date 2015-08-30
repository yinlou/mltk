package mltk.predictor.tree.ensemble.rf;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.cmdline.options.LearnerOptions;
import mltk.core.Instances;
import mltk.core.io.InstancesReader;
import mltk.predictor.Bagging;
import mltk.predictor.Learner;
import mltk.predictor.io.PredictorWriter;
import mltk.predictor.tree.RegressionTreeLearner;
import mltk.predictor.tree.RegressionTreeLearner.Mode;

/**
 * Class for learning random forests.
 * 
 * @author Yin Lou
 *
 */
public class RandomForestLearner extends Learner {

	static class Options extends LearnerOptions {

		@Argument(name = "-m", description = "construction mode:parameter. Construction mode can be alpha limited (a), depth limited (d), number of leaves limited (l) and minimum leaf size limited (s) (default: a:0.001)")
		String mode = "a:0.001";

		@Argument(name = "-f", description = "number of features to consider (default: 1/3 of total number of features)")
		int numFeatures = -1;

		@Argument(name = "-b", description = "bagging iterations (default: 100)")
		int baggingIters = 100;

	}

	/**
	 * Trains a random forest of regression trees.
	 *
	 * When bagging is turned off (b = 0), this procedure generates a single random regression tree. When the number of
	 * features to consider is the number of total features, this procedure builds bagged tree.
	 *
	 * <p>
	 *
	 * <pre>
	 * Usage: mltk.predictor.tree.ensemble.rf.RandomForestLearner
	 * -t	train set path
	 * [-r]	attribute file path
	 * [-o]	output model path
	 * [-V]	verbose (default: true)
	 * [-m]	construction mode:parameter. Construction mode can be alpha limited (a), depth limited (d), number of leaves limited (l) and minimum leaf size limited (s) (default: a:0.001)
	 * [-f]	number of features to consider
	 * [-b]	bagging iterations (default: 100)
	 * </pre>
	 *
	 * </p>
	 *
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(RandomForestLearner.class, opts);
		RandomRegressionTreeLearner rtLearner = new RandomRegressionTreeLearner();
		try {
			parser.parse(args);
			String[] data = opts.mode.split(":");
			if (data.length != 2) {
				throw new IllegalArgumentException();
			}
			switch (data[0]) {
				case "a":
					rtLearner.setConstructionMode(Mode.ALPHA_LIMITED);
					rtLearner.setAlpha(Double.parseDouble(data[1]));
					break;
				case "d":
					rtLearner.setConstructionMode(Mode.DEPTH_LIMITED);
					rtLearner.setMaxDepth(Integer.parseInt(data[1]));
					break;
				case "l":
					rtLearner.setConstructionMode(Mode.NUM_LEAVES_LIMITED);
					rtLearner.setMaxNumLeaves(Integer.parseInt(data[1]));
					break;
				case "s":
					rtLearner.setConstructionMode(Mode.MIN_LEAF_SIZE_LIMITED);
					rtLearner.setMinLeafSize(Integer.parseInt(data[1]));
				default:
					throw new IllegalArgumentException();
			}
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}

		Instances trainSet = InstancesReader.read(opts.attPath, opts.trainPath);

		rtLearner.setNumFeatures(opts.numFeatures);
		RandomForestLearner rfLearner = new RandomForestLearner();
		rfLearner.setBaggingIterations(opts.baggingIters);
		rfLearner.setRegressionTreeLearner(rtLearner);
		rfLearner.setVerbose(opts.verbose);
		
		long start = System.currentTimeMillis();
		RandomForest rf = rfLearner.build(trainSet);
		long end = System.currentTimeMillis();
		System.out.println("Time: " + (end - start) / 1000.0 + " (s).");

		if (opts.outputModelPath != null) {
			PredictorWriter.write(rf, opts.outputModelPath);
		}
	}
	
	private int baggingIters;
	private RegressionTreeLearner rtLearner;

	@Override
	public RandomForest build(Instances instances) {
		// Create bags
		Instances[] bags = Bagging.createBags(instances, baggingIters);
		
		RandomForest rf = new RandomForest(baggingIters);
		for (Instances bag : bags) {
			rf.add(rtLearner.build(bag));
		}
		return rf;
	}
	
	/**
	 * Constructor.
	 */
	public RandomForestLearner() {
		verbose = false;
		baggingIters = 100;
		rtLearner = new RandomRegressionTreeLearner();
	}
	
	/**
	 * Returns the number of bagging iterations.
	 * 
	 * @return the number of bagging iterations.
	 */
	public int getBaggingIterations() {
		return baggingIters;
	}

	/**
	 * Sets the number of bagging iterations.
	 * 
	 * @param baggingIters the number of bagging iterations.
	 */
	public void setBaggingIterations(int baggingIters) {
		this.baggingIters = baggingIters;
	}
	
	/**
	 * Returns the regression tree learner.
	 * 
	 * @return the regression tree learner.
	 */
	public RegressionTreeLearner getRegressionTreeLearner() {
		return rtLearner;
	}
	
	/**
	 * Sets the regression tree learner.
	 * 
	 * @param rtLearner the regression tree learner.
	 */
	public void setRegressionTreeLearner(RegressionTreeLearner rtLearner) {
		this.rtLearner = rtLearner;
	}

}
