package mltk.core.processor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.core.Attribute;
import mltk.core.BinnedAttribute;
import mltk.core.Bins;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.Attribute.Type;
import mltk.core.io.AttributesReader;
import mltk.core.io.InstancesReader;
import mltk.core.io.InstancesWriter;
import mltk.util.Element;
import mltk.util.tuple.DoublePair;

/**
 * Class for discretizers.
 * 
 * @author Yin Lou
 * 
 */
public class Discretizer {

	/**
	 * Constructor.
	 */
	public Discretizer() {

	}

	/**
	 * Discretizes an attribute using bins.
	 * 
	 * @param instances the dataset to discretize.
	 * @param attIndex the attribute index.
	 * @param bins the bins.
	 */
	public static void discretize(Instances instances, int attIndex, Bins bins) {
		Attribute attribute = instances.getAttributes().get(attIndex);
		BinnedAttribute binnedAttribute = new BinnedAttribute(attribute.getName(), bins);
		binnedAttribute.setIndex(attribute.getIndex());
		instances.getAttributes().set(attIndex, binnedAttribute);
		for (Instance instance : instances) {
			int v = bins.getIndex(instance.getValue(attribute.getIndex()));
			instance.setValue(attribute.getIndex(), v);
		}
	}

	/**
	 * Discretized an attribute with specified number of bins.
	 * 
	 * @param instances the dataset to discretize.
	 * @param attIndex the attribute index.
	 * @param maxNumBins the number of bins.
	 */
	public static void discretize(Instances instances, int attIndex, int maxNumBins) {
		Bins bins = computeBins(instances, attIndex, maxNumBins);
		discretize(instances, attIndex, bins);
	}

	/**
	 * Compute bins for a specified attribute.
	 * 
	 * @param instances the dataset to discretize.
	 * @param attIndex the attribute index.
	 * @param maxNumBins the number of bins.
	 */
	public static Bins computeBins(Instances instances, int attIndex, int maxNumBins) {
		Attribute attribute = instances.getAttributes().get(attIndex);
		List<Element<Double>> list = new ArrayList<>();
		for (Instance instance : instances) {
			list.add(new Element<Double>(instance.getWeight(), instance.getValue(attribute)));
		}
		Collections.sort(list);
		List<DoublePair> stats = new ArrayList<>();
		getStats(list, stats);
		if (stats.size() <= maxNumBins) {
			double[] a = new double[stats.size()];
			for (int i = 0; i < a.length; i++) {
				a[i] = stats.get(i).v1;
			}
			return new Bins(a, a);
		} else {
			double totalWeight = 0;
			for (DoublePair stat : stats) {
				totalWeight += stat.v2;
			}
			double binSize = totalWeight / maxNumBins;
			List<Double> boundaryList = new ArrayList<>();
			List<Double> medianList = new ArrayList<>();
			int start = 0;
			double weight = 0;
			for (int i = 0; i < stats.size(); i++) {
				weight += stats.get(i).v2;
				totalWeight -= stats.get(i).v2;
				if (weight >= binSize) {
					if (i == start) {
						boundaryList.add(stats.get(start).v1);
						medianList.add(stats.get(start).v1);
						weight = 0;
						start = i + 1;
					} else {
						double d1 = weight - binSize;
						double d2 = stats.get(i).v2 - d1;
						if (d1 < d2) {
							boundaryList.add(stats.get(i).v1);
							medianList.add(getMedian(stats, start, weight / 2));
							start = i + 1;
							weight = 0;
						} else {
							weight -= stats.get(i).v2;
							boundaryList.add(stats.get(i - 1).v1);
							medianList.add(getMedian(stats, start, weight / 2));
							start = i;
							weight = stats.get(i).v2;
						}
					}
					binSize = (totalWeight + weight) / (maxNumBins - boundaryList.size());
				} else if (i == stats.size() - 1) {
					boundaryList.add(stats.get(i).v1);
					medianList.add(getMedian(stats, start, weight / 2));
				}
			}
			double[] boundaries = new double[boundaryList.size()];
			double[] medians = new double[medianList.size()];
			for (int i = 0; i < boundaries.length; i++) {
				boundaries[i] = boundaryList.get(i);
				medians[i] = medianList.get(i);
			}
			return new Bins(boundaries, medians);
		}
	}

	/**
	 * Compute bins for a list of values.
	 * 
	 * @param x the vector of input data.
	 * @param maxNumBins the number of bins.
	 */
	public static Bins computeBins(double[] x, int maxNumBins) {
		List<Element<Double>> list = new ArrayList<>();
		for (double v : x) {
			list.add(new Element<Double>(1.0, v));
		}
		Collections.sort(list);
		List<DoublePair> stats = new ArrayList<>();
		getStats(list, stats);
		if (stats.size() <= maxNumBins) {
			double[] a = new double[stats.size()];
			for (int i = 0; i < a.length; i++) {
				a[i] = stats.get(i).v1;
			}
			return new Bins(a, a);
		} else {
			double totalWeight = 0;
			for (DoublePair stat : stats) {
				totalWeight += stat.v2;
			}
			double binSize = totalWeight / maxNumBins;
			List<Double> boundaryList = new ArrayList<>();
			List<Double> medianList = new ArrayList<>();
			int start = 0;
			double weight = 0;
			for (int i = 0; i < stats.size(); i++) {
				weight += stats.get(i).v2;
				totalWeight -= stats.get(i).v2;
				if (weight >= binSize) {
					if (i == start) {
						boundaryList.add(stats.get(start).v1);
						medianList.add(stats.get(start).v1);
						weight = 0;
						start = i + 1;
					} else {
						double d1 = weight - binSize;
						double d2 = stats.get(i).v2 - d1;
						if (d1 < d2) {
							boundaryList.add(stats.get(i).v1);
							medianList.add(getMedian(stats, start, weight / 2));
							start = i + 1;
							weight = 0;
						} else {
							weight -= stats.get(i).v2;
							boundaryList.add(stats.get(i - 1).v1);
							medianList.add(getMedian(stats, start, weight / 2));
							start = i;
							weight = stats.get(i).v2;
						}
					}
					binSize = (totalWeight + weight) / (maxNumBins - boundaryList.size());
				} else if (i == stats.size() - 1) {
					boundaryList.add(stats.get(i).v1);
					medianList.add(getMedian(stats, start, weight / 2));
				}
			}
			double[] boundaries = new double[boundaryList.size()];
			double[] medians = new double[medianList.size()];
			for (int i = 0; i < boundaries.length; i++) {
				boundaries[i] = boundaryList.get(i);
				medians[i] = medianList.get(i);
			}
			return new Bins(boundaries, medians);
		}
	}

	/**
	 * Compute bins for a list of values.
	 * 
	 * @param list the histogram.
	 * @param maxNumBins the number of bins.
	 */
	public static Bins computeBins(List<Element<Double>> list, int maxNumBins) {
		Collections.sort(list);
		List<DoublePair> stats = new ArrayList<>();
		getStats(list, stats);
		if (stats.size() <= maxNumBins) {
			double[] a = new double[stats.size()];
			for (int i = 0; i < a.length; i++) {
				a[i] = stats.get(i).v1;
			}
			return new Bins(a, a);
		} else {
			double totalWeight = 0;
			for (DoublePair stat : stats) {
				totalWeight += stat.v2;
			}
			double binSize = totalWeight / maxNumBins;
			List<Double> boundaryList = new ArrayList<>();
			List<Double> medianList = new ArrayList<>();
			int start = 0;
			double weight = 0;
			for (int i = 0; i < stats.size(); i++) {
				weight += stats.get(i).v2;
				totalWeight -= stats.get(i).v2;
				if (weight >= binSize) {
					if (i == start) {
						boundaryList.add(stats.get(start).v1);
						medianList.add(stats.get(start).v1);
						weight = 0;
						start = i + 1;
					} else {
						double d1 = weight - binSize;
						double d2 = stats.get(i).v2 - d1;
						if (d1 < d2) {
							boundaryList.add(stats.get(i).v1);
							medianList.add(getMedian(stats, start, weight / 2));
							start = i + 1;
							weight = 0;
						} else {
							weight -= stats.get(i).v2;
							boundaryList.add(stats.get(i - 1).v1);
							medianList.add(getMedian(stats, start, weight / 2));
							start = i;
							weight = stats.get(i).v2;
						}
					}
					binSize = (totalWeight + weight) / (maxNumBins - boundaryList.size());
				} else if (i == stats.size() - 1) {
					boundaryList.add(stats.get(i).v1);
					medianList.add(getMedian(stats, start, weight / 2));
				}
			}
			double[] boundaries = new double[boundaryList.size()];
			double[] medians = new double[medianList.size()];
			for (int i = 0; i < boundaries.length; i++) {
				boundaries[i] = boundaryList.get(i);
				medians[i] = medianList.get(i);
			}
			return new Bins(boundaries, medians);
		}
	}

	static double getMedian(List<DoublePair> stats, int start, double midPoint) {
		double weight = 0;
		for (int i = start; i < stats.size(); i++) {
			weight += stats.get(i).v2;
			if (weight >= midPoint) {
				return stats.get(i).v1;
			}
		}
		return stats.get((start + stats.size()) / 2).v1;
	}

	static void getStats(List<Element<Double>> list, List<DoublePair> stats) {
		if (list.size() == 0) {
			return;
		}
		double totalWeight = list.get(0).element;
		double lastValue = list.get(0).weight;
		for (int i = 1; i < list.size(); i++) {
			Element<Double> element = list.get(i);
			double value = element.weight;
			double weight = element.element;
			if (value != lastValue) {
				stats.add(new DoublePair(lastValue, totalWeight));
				lastValue = value;
				totalWeight = weight;
			} else {
				totalWeight += weight;
			}
		}
		stats.add(new DoublePair(lastValue, totalWeight));
	}

	static class Options {

		@Argument(name = "-r", description = "attribute file path", required = true)
		String attPath = null;
		
		@Argument(name = "-t", description = "training file path")
		String trainPath = null;

		@Argument(name = "-i", description = "input dataset path", required = true)
		String inputPath = null;

		@Argument(name = "-d", description = "discretized attribute file path")
		String disAttPath = null;

		@Argument(name = "-m", description = "output attribute file path")
		String outputAttPath = null;

		@Argument(name = "-o", description = "output dataset path", required = true)
		String outputPath = null;

		@Argument(name = "-n", description = "maximum num of bins (default: 256)")
		int maxNumBins = 256;

	}

	/**
	 * <p>
	 * 
	 * <pre>
	 * Usage: Discretizer
	 * -r	attribute file path
	 * -i	input dataset path
	 * -o	output dataset path
	 * [-d]	discretized attribute file path
	 * [-m]	output attribute file path
	 * [-n]	maximum num of bins (default: 256)
	 * [-t]	training file path
	 * </pre>
	 * 
	 * </p>
	 * 
	 * @param args the command line arguments.
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		Options app = new Options();
		CmdLineParser parser = new CmdLineParser(Discretizer.class, app);
		try {
			parser.parse(args);
			if (app.maxNumBins < 0) {
				throw new IllegalArgumentException();
			}
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}
		List<Attribute> attributes = null;
		if (app.trainPath != null) {
			Instances trainSet = InstancesReader.read(app.attPath, app.trainPath);
			attributes = trainSet.getAttributes();
			for (int i = 0; i < attributes.size(); i++) {
				Attribute attribute = attributes.get(i);
				if (attribute.getType() == Type.NUMERIC) {
					// Only discretize numeric attributes
					Discretizer.discretize(trainSet, i, app.maxNumBins);
				}
			}
		} else if (app.disAttPath != null) {
			attributes = AttributesReader.read(app.disAttPath).v1;
		} else {
			parser.printUsage();
			System.exit(1);
		}

		Instances instances = InstancesReader.read(app.attPath, app.inputPath);
		List<Attribute> attrs = instances.getAttributes();
		for (int i = 0; i < attrs.size(); i++) {
			Attribute attr = attrs.get(i);
			if (attr.getType() == Type.NUMERIC) {
				BinnedAttribute binnedAttr = (BinnedAttribute) attributes.get(i);
				// Only discretize numeric attributes
				Discretizer.discretize(instances, i, binnedAttr.getBins());
			}
		}

		if (app.outputAttPath != null) {
			InstancesWriter.write(instances, app.outputAttPath, app.outputPath);
		} else {
			InstancesWriter.write(instances, app.outputPath);
		}
	}

}
