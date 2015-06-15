package mltk.core.io;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

import mltk.core.Attribute;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.NominalAttribute;
import mltk.core.NumericalAttribute;
import mltk.util.MathUtils;
import mltk.util.tuple.Pair;

/**
 * Class for reading instances.
 * 
 * @author Yin Lou
 * 
 */
public class InstancesReader {

	/**
	 * Reads a set of instances from attribute file and data file. Attribute file can be null. Default delimiter is
	 * whitespace.
	 * 
	 * @param attFile the attribute file.
	 * @param dataFile the data file.
	 * @return a set of instances.
	 * @throws IOException
	 */
	public static Instances read(String attFile, String dataFile) throws IOException {
		return read(attFile, dataFile, "\\s+");
	}

	/**
	 * Reads a set of instances from attribute file and data file. Attribute file can be null.
	 * 
	 * @param attFile the attribute file.
	 * @param dataFile the data file.
	 * @param delimiter the delimiter.
	 * @return a set of instances.
	 * @throws IOException
	 */
	public static Instances read(String attFile, String dataFile, String delimiter) throws IOException {
		if (attFile != null) {
			Pair<List<Attribute>, Attribute> pair = AttributesReader.read(attFile);
			int classIndex = -1;
			if (pair.v2 != null) {
				classIndex = pair.v2.getIndex();
				pair.v2.setIndex(-1);
			}
			Instances instances = new Instances(pair.v1, pair.v2);
			int totalLength = instances.dimension();
			if (classIndex != -1) {
				totalLength++;
			}
			BufferedReader br = new BufferedReader(new FileReader(dataFile), 65535);
			for (;;) {
				String line = br.readLine();
				if (line == null) {
					break;
				}
				String[] data = line.split(delimiter);
				Instance instance = null;
				if (data.length >= 2 && data[1].indexOf(':') >= 0) {
					// Sparse instance
					instance = parseSparseInstance(data);
				} else if (data.length == totalLength) {
					// Dense instance
					instance = parseDenseInstance(data, classIndex);
				}
				if (instance != null) {
					instances.add(instance);
				}
			}
			br.close();
			return instances;
		} else {
			List<Attribute> attributes = new ArrayList<>();
			Instances instances = new Instances(attributes);
			int totalLength = -1;

			TreeSet<Integer> attrSet = new TreeSet<>();
			BufferedReader br = new BufferedReader(new FileReader(dataFile), 65535);
			for (;;) {
				String line = br.readLine();
				if (line == null) {
					break;
				}
				String[] data = line.split(delimiter);
				Instance instance = null;
				if (data.length >= 2 && data[1].indexOf(':') >= 0) {
					// Sparse instance
					instance = parseSparseInstance(data, attrSet);
				} else {
					// Dense instance
					if (totalLength == -1) {
						totalLength = data.length;
					} else if (data.length == totalLength) {
						instance = parseDenseInstance(data, -1);
					}

				}
				if (instance != null) {
					instances.add(instance);
				}
			}
			br.close();

			if (totalLength == -1) {
				for (Integer attIndex : attrSet) {
					Attribute att = new NumericalAttribute("f" + attIndex);
					att.setIndex(attIndex);
					attributes.add(att);
				}
			} else {
				for (int j = 0; j < totalLength; j++) {
					Attribute att = new NumericalAttribute("f" + j);
					att.setIndex(j);
					attributes.add(att);
				}
			}

			assignTargetAttribute(instances);
			return instances;
		}
	}

	/**
	 * Reads a set of dense instances from data file. Default delimiter is whitespace.
	 * 
	 * @param file the data file.
	 * @param classIndex the index of the class attribute, -1 if no class attribute.
	 * @return a set of dense instances.
	 * @throws IOException
	 */
	public static Instances read(String file, int classIndex) throws IOException {
		return read(file, classIndex, "\\s+");
	}

	/**
	 * Reads a set of dense instances from data file.
	 * 
	 * @param file the data file.
	 * @param classIndex the index of the class attribute, -1 if no class attribute.
	 * @param delimiter the delimiter.
	 * @return a set of dense instances.
	 * @throws IOException
	 */
	public static Instances read(String file, int classIndex, String delimiter) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(file), 65535);
		List<Attribute> attributes = new ArrayList<>();
		Instances instances = new Instances(attributes);
		for (;;) {
			String line = br.readLine();
			if (line == null) {
				break;
			}
			String[] data = line.split(delimiter);
			Instance instance = parseDenseInstance(data, classIndex);
			instances.add(instance);
		}
		br.close();

		int numAttributes = instances.get(0).getValues().length;
		for (int i = 0; i < numAttributes; i++) {
			Attribute att = new NumericalAttribute("f" + i);
			att.setIndex(i);
			attributes.add(att);
		}

		if (classIndex >= 0) {
			assignTargetAttribute(instances);
		}

		return instances;
	}

	/**
	 * Parses a dense instance from strings.
	 * 
	 * @param data the string array.
	 * @param classIndex the class index.
	 * @return a dense instance from strings.
	 */
	private static Instance parseDenseInstance(String[] data, int classIndex) {
		if (classIndex < 0) {
			double[] vector = new double[data.length];
			double classValue = Double.NaN;
			for (int i = 0; i < data.length; i++) {
				vector[i] = Double.parseDouble(data[i]);
			}
			return new Instance(vector, classValue);
		} else {
			double[] vector = new double[data.length - 1];
			double classValue = Double.NaN;
			for (int i = 0; i < data.length; i++) {
				double value = Double.parseDouble(data[i]);
				if (i < classIndex) {
					vector[i] = value;
				} else if (i > classIndex) {
					vector[i - 1] = value;
				} else {
					classValue = value;
				}
			}
			return new Instance(vector, classValue);
		}
	}

	/**
	 * Parses a sparse instance from strings.
	 * 
	 * @param data the string array.
	 * @param attrSet the attributes set.
	 * @return a sparse instance from strings.
	 */
	private static Instance parseSparseInstance(String[] data, TreeSet<Integer> attrSet) {
		double classValue = Double.parseDouble(data[0]);
		int[] indices = new int[data.length - 1];
		double[] values = new double[data.length - 1];
		for (int i = 0; i < indices.length; i++) {
			String[] pair = data[i + 1].split(":");
			indices[i] = Integer.parseInt(pair[0]);
			values[i] = Double.parseDouble(pair[1]);
			attrSet.add(indices[i]);
		}
		return new Instance(indices, values, classValue);
	}

	/**
	 * Parses a sparse instance from strings.
	 * 
	 * @param data the string array.
	 * @return a sparse instance from strings.
	 */
	private static Instance parseSparseInstance(String[] data) {
		double classValue = Double.parseDouble(data[0]);
		int[] indices = new int[data.length - 1];
		double[] values = new double[data.length - 1];
		for (int i = 0; i < indices.length; i++) {
			String[] pair = data[i + 1].split(":");
			indices[i] = Integer.parseInt(pair[0]);
			values[i] = Double.parseDouble(pair[1]);
		}
		return new Instance(indices, values, classValue);
	}

	/**
	 * Assigns target attribute for a dataset.
	 * 
	 * @param instances the data set.
	 */
	private static void assignTargetAttribute(Instances instances) {
		boolean isInteger = true;
		for (Instance instance : instances) {
			if (!MathUtils.isInteger(instance.getTarget())) {
				isInteger = false;
				break;
			}
		}
		if (isInteger) {
			TreeSet<Integer> set = new TreeSet<>();
			for (Instance instance : instances) {
				double target = instance.getTarget();
				set.add((int) target);
			}
			String[] states = new String[set.size()];
			int i = 0;
			for (Integer v : set) {
				states[i++] = v.toString();
			}
			instances.setTargetAttribute(new NominalAttribute("target", states));
		} else {
			instances.setTargetAttribute(new NumericalAttribute("target"));
		}
	}

}
