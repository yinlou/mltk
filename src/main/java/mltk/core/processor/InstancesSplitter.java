package mltk.core.processor;

import java.io.File;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.io.InstancesReader;
import mltk.core.io.InstancesWriter;
import mltk.util.Random;

/**
 * Class for cross validation.
 * 
 * @author Yin Lou
 * 
 */
public class InstancesSplitter {

	/**
	 * Splits the dataset according to the ratio. This method returns two instances objects, the size of the first one
	 * is 100% * ratio of the orignal dataset while the size of the second one is 100% * (1 - ratio) of the orignal
	 * dataset.
	 * 
	 * @param instances the dataset.
	 * @param ratio the ratio.
	 * @return two smaller datasets.
	 */
	public static Instances[] split(Instances instances, double ratio) {
		Instances dataset = new Instances(instances);
		dataset.shuffle();
		Instances train = new Instances(dataset.getAttributes(), dataset.getTargetAttribute());
		Instances test = new Instances(dataset.getAttributes(), dataset.getTargetAttribute());
		int nTrain = (int) (dataset.size() * ratio);
		for (int i = 0; i < nTrain; i++) {
			train.add(dataset.get(i));
		}
		for (int i = nTrain; i < dataset.size(); i++) {
			test.add(dataset.get(i));
		}
		return new Instances[] { train, test };
	}

	/**
	 * Splits the dataset into k equi-sized datasets.
	 * 
	 * @param instances the dataset.
	 * @param k the number of datasets to return.
	 * @return k equi-sized datasets.
	 */
	public static Instances[] split(Instances instances, int k) {
		Instances dataset = new Instances(instances);
		dataset.shuffle();
		Instances[] datasets = new Instances[k];
		for (int i = 0; i < datasets.length; i++) {
			datasets[i] = new Instances(dataset.getAttributes(), dataset.getTargetAttribute());
		}
		for (int i = 0; i < dataset.size(); i++) {
			datasets[i % datasets.length].add(dataset.get(i));
		}
		return datasets;
	}

	/**
	 * Creates cross validation folds from a dataset. For each cross validation fold contains a training set and a test
	 * set.
	 * 
	 * @param instances the dataset.
	 * @param k the number of cross validation folds.
	 * @return the cross validation datasets.
	 */
	public static Instances[][] createCrossValidationFolds(Instances instances, int k) {
		Instances[] datasets = split(instances, k);
		Instances[][] folds = new Instances[k][2];
		for (int i = 0; i < k; i++) {
			folds[i][1] = datasets[i];
			folds[i][0] = new Instances(instances.getAttributes(), instances.getTargetAttribute());
			for (int j = 0; j < k; j++) {
				if (i == j) {
					continue;
				}
				for (Instance instance : datasets[j]) {
					folds[i][0].add(instance);
				}
			}
		}
		return folds;
	}

	/**
	 * Creates cross validation folds from a dataset. For each cross validation fold contains a training set, a
	 * validation set and a test set.
	 * 
	 * @param instances the dataset.
	 * @param k the number of cross validation folds.
	 * @param ratio the ratio that controls how many points in the training set for each fold.
	 * @return the cross validation datasets.
	 */
	public static Instances[][] createCrossValidationFolds(Instances instances, int k, double ratio) {
		Instances[] datasets = split(instances, k);
		Instances[][] folds = new Instances[k][3];
		for (int i = 0; i < k; i++) {
			folds[i][2] = datasets[i];
			Instances trainSet = new Instances(instances.getAttributes(), instances.getTargetAttribute());
			for (int j = 0; j < k; j++) {
				if (i == j) {
					continue;
				}
				for (Instance instance : datasets[j]) {
					trainSet.add(instance);
				}
			}
			Instances[] tmp = split(trainSet, ratio);
			folds[i][0] = tmp[0];
			folds[i][1] = tmp[1];
		}
		return folds;
	}

	static class Options {

		@Argument(name = "-r", description = "attribute file path")
		String attPath = null;

		@Argument(name = "-i", description = "input dataset path", required = true)
		String inputPath = null;

		@Argument(name = "-o", description = "output directory path", required = true)
		String outputDirPath = null;

		@Argument(name = "-m", description = "splitting mode:parameter. Splitting mode can be split (s) and cross validation (c) (default: c:5)")
		String crossValidationMode = "c:5";

		@Argument(name = "-s", description = "seed of the random number generator (default: 0)")
		long seed = 0L;
	}

	/**
	 * <p>
	 * 
	 * <pre>
	 * Usage: mltk.core.processor.InstancesSplitter
	 * -r	attribute file path
	 * -i	input dataset path
	 * -o	output directory path
	 * [-c]	splitting mode:parameter. Splitting mode can be split (s) and cross validation (c) (default: c:5)
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
		CmdLineParser parser = new CmdLineParser(InstancesSplitter.class, opts);
		String[] data = null;
		try {
			parser.parse(args);
			data = opts.crossValidationMode.split(":");
			if (data.length < 2) {
				throw new IllegalArgumentException();
			}
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}

		Random.getInstance().setSeed(opts.seed);

		Instances instances = InstancesReader.read(opts.attPath, opts.inputPath);

		File attFile = new File(opts.attPath);
		String prefix = attFile.getName().split("\\.")[0];

		File dir = new File(opts.outputDirPath);
		if (!dir.exists()) {
			dir.mkdir();
		}

		switch (data[0]) {
			case "c":
				int k = Integer.parseInt(data[1]);
				if (data.length == 2) {
					Instances[][] folds = InstancesSplitter.createCrossValidationFolds(instances, k);
					for (int i = 0; i < folds.length; i++) {
						String path = opts.outputDirPath + File.separator + "cv." + i;
						File directory = new File(path);
						if (!directory.exists()) {
							directory.mkdir();
						}
						InstancesWriter.write(folds[i][0], path + File.separator + prefix + ".attr", path
								+ File.separator + prefix + ".train.all");
						InstancesWriter.write(folds[i][1], path + File.separator + prefix + ".test");
					}
				} else {
					double ratio = Double.parseDouble(data[2]);
					Instances[][] folds = InstancesSplitter.createCrossValidationFolds(instances, k, ratio);
					for (int i = 0; i < folds.length; i++) {
						String path = opts.outputDirPath + File.separator + "cv." + i;
						File directory = new File(path);
						if (!directory.exists()) {
							directory.mkdir();
						}
						InstancesWriter.write(folds[i][0], path + File.separator + prefix + ".attr", path
								+ File.separator + prefix + ".train");
						InstancesWriter.write(folds[i][1], path + File.separator + prefix + ".valid");
						InstancesWriter.write(folds[i][2], path + File.separator + prefix + ".test");
					}
				}
				break;
			case "s":
				double ratio = Double.parseDouble(data[1]);
				Instances[] datasets = InstancesSplitter.split(instances, ratio);
				InstancesWriter.write(datasets[0], opts.outputDirPath + File.separator + prefix + ".attr",
						opts.outputDirPath + File.separator + prefix + ".train");
				InstancesWriter.write(datasets[1], opts.outputDirPath + File.separator + prefix + ".valid");
				break;
			default:
				break;
		}
	}

}
