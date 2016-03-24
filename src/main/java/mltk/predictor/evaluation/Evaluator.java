package mltk.predictor.evaluation;

import java.util.List;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.io.InstancesReader;
import mltk.predictor.ProbabilisticClassifier;
import mltk.predictor.Classifier;
import mltk.predictor.Regressor;
import mltk.predictor.io.PredictorReader;

/**
 * Class for making evaluations.
 * 
 * @author Yin Lou
 * 
 */
public class Evaluator {

	/**
	 * Returns the area under ROC curve.
	 * 
	 * @param classifier a classifier that outputs probability.
	 * @param instances the instances.
	 * @return the area under ROC curve.
	 */
	public static double evalAreaUnderROC(ProbabilisticClassifier classifier, Instances instances) {
		double[] probs = new double[instances.size()];
		double[] targets = new double[instances.size()];
		for (int i = 0; i < probs.length; i++) {
			Instance instance = instances.get(i);
			probs[i] = classifier.predictProbabilities(instance)[1];
			targets[i] = instance.getTarget();
		}
		return new AUC().eval(probs, targets);
	}

	/**
	 * Returns the root mean squared error.
	 * 
	 * @param preds the predictions.
	 * @param targets the targets.
	 * @return the root mean squared error.
	 */
	public static double evalRMSE(List<Double> preds, List<Double> targets) {
		double rmse = 0;
		for (int i = 0; i < preds.size(); i++) {
			double d = targets.get(i) - preds.get(i);
			rmse += d * d;
		}
		rmse = Math.sqrt(rmse / preds.size());
		return rmse;
	}

	/**
	 * Returns the root mean squared error.
	 * 
	 * @param regressor the regressor.
	 * @param instances the instances.
	 * @return the root mean squared error.
	 */
	public static double evalRMSE(Regressor regressor, Instances instances) {
		double rmse = 0;
		for (int i = 0; i < instances.size(); i++) {
			Instance instance = instances.get(i);
			double target = instance.getTarget();
			double pred = regressor.regress(instance);
			double d = target - pred;
			rmse += d * d;
		}
		rmse = Math.sqrt(rmse / instances.size());
		return rmse;
	}

	/**
	 * Returns the classification error.
	 * 
	 * @param classifier the classifier.
	 * @param instances the instances.
	 * @return the classification error.
	 */
	public static double evalError(Classifier classifier, Instances instances) {
		double error = 0;
		for (int i = 0; i < instances.size(); i++) {
			Instance instance = instances.get(i);
			double target = instance.getTarget();
			double pred = classifier.classify(instance);
			if (target != pred) {
				error++;
			}
		}
		error /= instances.size();
		return error;
	}

	static class Options {

		@Argument(name = "-r", description = "attribute file path")
		String attPath = null;

		@Argument(name = "-d", description = "data set path", required = true)
		String dataPath = null;

		@Argument(name = "-m", description = "model path", required = true)
		String modelPath = null;

		@Argument(name = "-e", description = "AUC (a), Error (c), RMSE (r) (default: r)")
		String task = "r";

	}

	/**
	 * <p>
	 * 
	 * <pre>
	 * Usage: Evaluator
	 * -d	data set path
	 * -m	model path
	 * [-r]	attribute file path
	 * [-e]	AUC (a), Error (c), RMSE (r) (default: r)
	 * </pre>
	 * 
	 * </p>
	 * 
	 * @param args the command line arguments.
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(Evaluator.class, opts);
		try {
			parser.parse(args);
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}

		Instances instances = InstancesReader.read(opts.attPath, opts.dataPath);
		mltk.predictor.Predictor predictor = PredictorReader.read(opts.modelPath);

		switch (opts.task) {
			case "a":
				ProbabilisticClassifier probClassifier = (ProbabilisticClassifier) predictor;
				double auc = Evaluator.evalAreaUnderROC(probClassifier, instances);
				System.out.println("AUC: " + auc);
				break;
			case "c":
				Classifier classifier = (Classifier) predictor;
				double error = Evaluator.evalError(classifier, instances);
				System.out.println("Error: " + error);
				break;
			case "r":
				Regressor regressor = (Regressor) predictor;
				double rmse = Evaluator.evalRMSE(regressor, instances);
				System.out.println("RMSE: " + rmse);
				break;
			default:
				break;
		}
	}

}
