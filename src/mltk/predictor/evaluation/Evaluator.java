package mltk.predictor.evaluation;

import java.util.Arrays;
import java.util.Comparator;
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
import mltk.util.tuple.DoublePair;

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
	 * @param probs probability of positive class
	 * @param targets 0/1 targets
	 * @return the area under ROC curve.
	 */
	public static double evalAreaUnderROC(double[] probs, double[] targets) {
		DoublePair[] a = new DoublePair[probs.length];
		for (int i = 0; i < probs.length; i++) {
			a[i] = new DoublePair(probs[i], targets[i]);
		}
		Arrays.sort(a, new Comparator<DoublePair>() {

			@Override
			public int compare(DoublePair o1, DoublePair o2) {
				if (o1.v1 < o2.v1) {
					return -1;
				} else if (o1.v1 > o2.v1) {
					return 1;
				} else {
					if (o1.v2 < o2.v2) {
						return -1;
					} else if (o1.v2 > o2.v2) {
						return 1;
					} else {
						return 0;
					}
				}
			}

		});

		double[] fraction = new double[a.length];
		for (int idx = 0; idx < fraction.length;) {
			int begin = idx;
			double pos = 0;
			for (; idx < fraction.length && a[idx].v1 == a[begin].v1; idx++) {
				pos += a[idx].v2;
			}
			double frac = pos / (idx - begin);
			for (int i = begin; i < idx; i++) {
				fraction[i] = frac;
			}
		}

		double tt = 0;
		double tf = 0;
		double ft = 0;
		double ff = 0;

		for (int i = 0; i < a.length; i++) {
			tf += a[i].v2;
			ff += 1 - a[i].v2;
		}

		double area = 0;
		double tpfPrev = 0;
		double fpfPrev = 0;

		for (int i = a.length - 1; i >= 0; i--) {
			tt += fraction[i];
			tf -= fraction[i];
			ft += 1 - fraction[i];
			ff -= 1 - fraction[i];
			double tpf = tt / (tt + tf);
			double fpf = 1.0 - ff / (ft + ff);
			area += 0.5 * (tpf + tpfPrev) * (fpf - fpfPrev);
			tpfPrev = tpf;
			fpfPrev = fpf;
		}

		return area;
	}

	/**
	 * Returns the area under ROC curve.
	 * 
	 * @param probs probability of positive class
	 * @param targets 0/1 targets
	 * @return the area under ROC curve.
	 */
	public static double evalAreaUnderROC(List<Double> probs, List<Double> targets) {
		DoublePair[] a = new DoublePair[probs.size()];
		for (int i = 0; i < probs.size(); i++) {
			a[i] = new DoublePair(probs.get(i), targets.get(i));
		}
		Arrays.sort(a, new Comparator<DoublePair>() {

			@Override
			public int compare(DoublePair o1, DoublePair o2) {
				if (o1.v1 < o2.v1) {
					return -1;
				} else if (o1.v1 > o2.v1) {
					return 1;
				} else {
					if (o1.v2 < o2.v2) {
						return -1;
					} else if (o1.v2 > o2.v2) {
						return 1;
					} else {
						return 0;
					}
				}
			}

		});

		double[] fraction = new double[a.length];
		for (int idx = 0; idx < fraction.length;) {
			int begin = idx;
			double pos = 0;
			for (; idx < fraction.length && a[idx].v1 == a[begin].v1; idx++) {
				pos += a[idx].v2;
			}
			double frac = pos / (idx - begin);
			for (int i = begin; i < idx; i++) {
				fraction[i] = frac;
			}
		}

		double tt = 0;
		double tf = 0;
		double ft = 0;
		double ff = 0;

		for (int i = 0; i < a.length; i++) {
			tf += a[i].v2;
			ff += 1 - a[i].v2;
		}

		double area = 0;
		double tpfPrev = 0;
		double fpfPrev = 0;

		for (int i = a.length - 1; i >= 0; i--) {
			tt += fraction[i];
			tf -= fraction[i];
			ft += 1 - fraction[i];
			ff -= 1 - fraction[i];
			double tpf = tt / (tt + tf);
			double fpf = 1.0 - ff / (ft + ff);
			area += 0.5 * (tpf + tpfPrev) * (fpf - fpfPrev);
			tpfPrev = tpf;
			fpfPrev = fpf;
		}

		return area;
	}

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
		return evalAreaUnderROC(probs, targets);
	}

	/**
	 * Returns the root mean squared error.
	 * 
	 * @param preds the predictions.
	 * @param targets the targets.
	 * @return the root mean squared error.
	 */
	public static double evalRMSE(double[] preds, double[] targets) {
		double rmse = 0;
		for (int i = 0; i < preds.length; i++) {
			double d = targets[i] - preds[i];
			rmse += d * d;
		}
		rmse = Math.sqrt(rmse / preds.length);
		return rmse;
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
