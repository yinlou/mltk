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
	 * @param allowMV flag set when missing values are allowed
	 * @return a set of instances.
	 * @throws IOException
	 */
	public static Instances read(String attFile, String dataFile, boolean allowMV) throws IOException {
		return read(attFile, dataFile, "\\s+", allowMV);
	}

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
		return read(attFile, dataFile, "\\s+", false);
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
	public static Instances read(String attFile, String dataFile, String delimiter, boolean allowMV) throws IOException {
		if (attFile != null) {
			AttrInfo ainfo = AttributesReader.read(attFile);
			return read(ainfo, dataFile, delimiter, allowMV);
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
						instance = parseDenseInstance(data);
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
	 * Reads a set of instances from attribute information and data file.
	 * 
	 * @param ainfo attribute information
	 * @param dataFile the data file.
	 * @param delimiter the delimiter.
	 * @return a set of instances.
	 * @throws IOException
	 */
	public static Instances read(AttrInfo ainfo, String dataFile, String delimiter, boolean allowMV) throws IOException {
		Instances instances = new Instances(ainfo.attributes, ainfo.clsAttr);

		BufferedReader br = new BufferedReader(new FileReader(dataFile), 65535);
		for (;;) {
			String line = br.readLine();
			if (line == null) {
				break;
			}
			String[] data = line.split(delimiter);
			Instance instance = parseDenseInstance(data, ainfo, allowMV);
			if (instance != null) {
				instances.add(instance);
			}
		}
		br.close();
		return instances;
	}



	/**
	 * Parses a dense instance from strings.
	 * 
	 * @param data the string array.
	 * @param ainfo attribute information containing the class index.
	 * @return a dense instance from strings.
	 */
	private static Instance parseDenseInstance(String[] data, AttrInfo ainfo, boolean allowMV) {
		double[] vector = new double[ainfo.columns.size()];
		for (int i = 0; i < vector.length; i++) {
			int col = ainfo.columns.get(i);
			if(allowMV) {
				vector[i] = data[col].equals("?") ?
							Double.NaN :
							Double.parseDouble(data[col]);
			} else {
				if(data[col].equals("?"))
					System.out.println("Missing values are not allowed.\n");
				vector[i] = Double.parseDouble(data[col]); //crashes on "?"
			}
		}
		double classValue = (ainfo.clsAttr.getIndex() < 0) ? 
							Double.NaN : 
							Double.parseDouble(data[ainfo.clsAttr.getIndex()]);
		return new Instance(vector, classValue);
	}
	
	/**
	 * Parses a dense instance from strings.
	 * 
	 * @param data the string array.
	 * @return a dense instance from strings.
	 */	
	private static Instance parseDenseInstance(String[] data) {
		double[] vector = new double[data.length];
		for (int i = 0; i < vector.length; i++) {
			vector[i] = Double.parseDouble(data[i]);
		}		
		return new Instance(vector, Double.NaN);
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
