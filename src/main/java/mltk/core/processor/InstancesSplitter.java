package mltk.core.processor;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.core.Attribute;
import mltk.core.Attribute.Type;
import mltk.core.BinnedAttribute;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.NominalAttribute;
import mltk.core.io.InstancesReader;
import mltk.core.io.InstancesWriter;
import mltk.util.MathUtils;
import mltk.util.Random;
import mltk.util.StatUtils;

/**
 * Class for cross validation.
 * 
 * @author Yin Lou
 * 
 */
public class InstancesSplitter {
	
	static class Options {

		@Argument(name = "-r", description = "attribute file path")
		String attPath = null;

		@Argument(name = "-i", description = "input dataset path", required = true)
		String inputPath = null;

		@Argument(name = "-o", description = "output directory path", required = true)
		String outputDirPath = null;

		@Argument(name = "-m", description = "splitting mode:parameter. Splitting mode can be split (s) and cross validation (c) (default: c:5)")
		String crossValidationMode = "c:5";
		
		@Argument(name = "-a", description = "attribute name to perform stratified sampling (default: null)")
		String attToStrafity = null;
		
		@Argument(name = "-s", description = "seed of the random number generator (default: 0)")
		long seed = 0L;
	}
	
	/**
	 * Splits a dataset.
	 * 
	 * <pre>
	 * Usage: mltk.core.processor.InstancesSplitter
	 * -i	input dataset path
	 * -o	output directory path
	 * [-r]	attribute file path
	 * [-m]	splitting mode:parameter. Splitting mode can be split (s) and cross validation (c) (default: c:5)
	 * [-a]	attribute name to perform stratified sampling (default: null)
	 * [-s]	seed of the random number generator (default: 0)
	 * </pre>
	 * 
	 * @param args the command line arguments.
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
					Instances[][] folds = InstancesSplitter.createCrossValidationFolds(instances, opts.attToStrafity, k);
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
					Instances[][] folds = InstancesSplitter.createCrossValidationFolds(instances, opts.attToStrafity, k, ratio);
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
				if (data.length == 2) {
					double ratio = Double.parseDouble(data[1]);
					Instances[] datasets = InstancesSplitter.split(instances, opts.attToStrafity, ratio);
					InstancesWriter.write(datasets[0], opts.outputDirPath + File.separator + prefix + ".attr",
							opts.outputDirPath + File.separator + prefix + ".train");
					InstancesWriter.write(datasets[1], opts.outputDirPath + File.separator + prefix + ".valid");
				} else if (data.length == 3) {
					double[] ratios = new double[data.length - 1];
					for (int i = 0; i < ratios.length; i++) {
						ratios[i] = Double.parseDouble(data[i + 1]);
					}
					Instances[] datasets = InstancesSplitter.split(instances, opts.attToStrafity, ratios);
					InstancesWriter.write(datasets[0], opts.outputDirPath + File.separator + prefix + ".attr",
							opts.outputDirPath + File.separator + prefix + ".train");
					InstancesWriter.write(datasets[1], opts.outputDirPath + File.separator + prefix + ".valid");
					InstancesWriter.write(datasets[2], opts.outputDirPath + File.separator + prefix + ".test");
				} else {
					double[] ratios = new double[data.length - 1];
					for (int i = 0; i < ratios.length; i++) {
						ratios[i] = Double.parseDouble(data[i + 1]);
					}
					Instances[] datasets = InstancesSplitter.split(instances, opts.attToStrafity, ratios);
					for (int i = 0; i < datasets.length; i++) {
						InstancesWriter.write(datasets[i], opts.outputDirPath + File.separator + prefix + ".data." + i);
					}
				}
				break;
			default:
				break;
		}
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
	
	/**
	 * Creates cross validation folds from a dataset. For each cross validation fold contains a training set and a test
	 * set.
	 * 
	 * @param instances the dataset.
	 * @param attToStratify the attribute to perform stratified sampling.
	 * @param k the number of cross validation folds.
	 * @return the cross validation datasets.
	 */
	public static Instances[][] createCrossValidationFolds(Instances instances, String attToStratify, int k) {
		Instances[] datasets = split(instances, attToStratify, k);
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
	 * @param attToStratify the attribute to perform stratified sampling.
	 * @param k the number of cross validation folds.
	 * @param ratio the ratio that controls how many points in the training set for each fold.
	 * @return the cross validation datasets.
	 */
	public static Instances[][] createCrossValidationFolds(Instances instances, String attToStratify, int k, double ratio) {
		Instances[] datasets = split(instances, attToStratify, k);
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
			Instances[] tmp = split(trainSet, attToStratify, ratio);
			folds[i][0] = tmp[0];
			folds[i][1] = tmp[1];
		}
		return folds;
	}

	/**
	 * Splits the dataset according to the ratios. This method returns multiple instances objects, the size of each
	 * partition is determined by the ratios array. The sum of ratios can be smaller than 1.
	 * 
	 * @param instances the dataset.
	 * @param ratios the ratios.
	 * @return partitions of the dataset.
	 */
	public static Instances[] split(Instances instances, double... ratios) {
		if (StatUtils.sum(ratios) > 1) {
			throw new IllegalArgumentException("Sum of ratios is larger than 1");
		}
		Instances dataset = new Instances(instances);
		dataset.shuffle(Random.getInstance().getRandom());
		Instances[] datasets = new Instances[ratios.length];
		for (int i = 0; i < datasets.length; i++) {
			datasets[i] = new Instances(dataset.getAttributes(), dataset.getTargetAttribute());
		}
		double sumRatios = StatUtils.sum(ratios);
		int n = 0;
		for (int k = 0; k < datasets.length; k++) {
			int m = (int) (dataset.size() * ratios[k]);
			if (k == datasets.length - 1 && MathUtils.equals(sumRatios, 1.0)) {
				m = dataset.size() - n;
			}
			Instances partition = datasets[k];
			for (int i = n; i < n + m; i++) {
				partition.add(dataset.get(i));
			}
			n += m;
		}
		return datasets;
	}
	
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
		return split(instances, new double[] { ratio, 1 - ratio });
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
		dataset.shuffle(Random.getInstance().getRandom());
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
	 * Splits the dataset according to the ratio. This method returns two instances objects, the size of the first one
	 * is 100% * ratio of the orignal dataset while the size of the second one is 100% * (1 - ratio) of the orignal
	 * dataset.
	 * 
	 * @param instances the dataset.
	 * @param attToStratify the attribute to perform stratified sampling.
	 * @param ratio the ratio.
	 * @return two smaller datasets.
	 */
	public static Instances[] split(Instances instances, String attToStratify, double ratio) {
		return split(instances, attToStratify, new double[] { ratio, 1 - ratio });
	}

	/**
	 * Splits the dataset according to the ratios. This method returns multiple instances objects, the size of each
	 * partition is determined by the ratios array. The sum of ratios can be smaller than 1.
	 * 
	 * @param instances the dataset.
	 * @param attToStratify the attribute to perform stratified sampling.
	 * @param ratios the ratios.
	 * @return partitions of the dataset.
	 */
	public static Instances[] split(Instances instances, String attToStratify, double... ratios) {
		if (attToStratify == null) {
			return split(instances, ratios);
		}
		List<List<Instance>> strata = getStrata(instances, attToStratify);
		if (strata == null) {
			return split(instances, ratios);
		}
		Instances[] datasets = new Instances[ratios.length];
		for (int i = 0; i < datasets.length; i++) {
			datasets[i] = new Instances(instances.getAttributes(), instances.getTargetAttribute());
		}
		double sumRatios = StatUtils.sum(ratios);
		for (List<Instance> list : strata) {
			int n = 0;
			for (int k = 0; k < datasets.length; k++) {
				int m = (int) (list.size() * ratios[k]);
				if (k == datasets.length -1 && MathUtils.equals(sumRatios, 1.0)) {
					m = list.size() - n;
				}
				Instances partition = datasets[k];
				for (int i = n; i < n + m; i++) {
					partition.add(list.get(i));
				}
				n += m;
			}
		}
		return datasets;
	}

	/**
	 * Splits the dataset into k equi-sized datasets.
	 * 
	 * @param instances the dataset.
	 * @param attToStratify the attribute to perform stratified sampling.
	 * @param k the number of datasets to return.
	 * @return k equi-sized datasets.
	 */
	public static Instances[] split(Instances instances, String attToStratify, int k) {
		if (attToStratify == null) {
			return split(instances, k);
		}
		List<List<Instance>> strata = getStrata(instances, attToStratify);
		if (strata == null) {
			return split(instances, k);
		}
		Instances[] datasets = new Instances[k];
		for (int i = 0; i < datasets.length; i++) {
			datasets[i] = new Instances(instances.getAttributes(), instances.getTargetAttribute());
		}
		for (List<Instance> stratum : strata) {
			for (int i = 0; i < stratum.size(); i++) {
				datasets[i % datasets.length].add(stratum.get(i));
			}
		}
		return datasets;
	}
	
	private static List<List<Instance>> getStrata(Instances instances, String attToStratify) {
		List<List<Instance>> lists = new ArrayList<>();
		Instances dataset = new Instances(instances);
		dataset.shuffle(Random.getInstance().getRandom());
		if (instances.getTargetAttribute().getName().equals(attToStratify)) {
			Attribute targetAtt = instances.getTargetAttribute();
			if (targetAtt == null || targetAtt.getType() == Type.NUMERIC) {
				return null;
			}
			int cardinality = 0;
			if (targetAtt.getType() == Type.BINNED) {
				cardinality = ((BinnedAttribute) targetAtt).getNumBins();
			} else {
				cardinality = ((NominalAttribute) targetAtt).getCardinality();
			}
			for (int i = 0; i < cardinality; i++) {
				lists.add(new ArrayList<Instance>());
			}
			for (Instance instance : instances) {
				int idx = (int) instance.getTarget();
				lists.get(idx).add(instance);
			}
		} else {
			List<Attribute> attributes = instances.getAttributes();
			Attribute attr = null;
			for (Attribute att : attributes) {
				if (att.getName().equals(attToStratify)) {
					attr = att;
					break;
				}
			}
			if (attr == null || attr.getType() == Type.NUMERIC) {
				return null;
			}
			int cardinality = 0;
			if (attr.getType() == Type.BINNED) {
				cardinality = ((BinnedAttribute) attr).getNumBins();
			} else {
				cardinality = ((NominalAttribute) attr).getCardinality();
			}
			for (int i = 0; i < cardinality; i++) {
				lists.add(new ArrayList<Instance>());
			}
			for (Instance instance : dataset) {
				int idx = (int) instance.getValue(attr);
				lists.get(idx).add(instance);
			}
		}
		
		return lists;
	}

}
