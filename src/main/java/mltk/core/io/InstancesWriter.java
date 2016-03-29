package mltk.core.io;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

import mltk.core.Attribute;
import mltk.core.Instance;
import mltk.core.Instances;

/**
 * Class for writing instances.
 * 
 * @author Yin Lou
 * 
 */
public class InstancesWriter {

	/**
	 * Writes a set of dense instances to attribute file and data file.
	 * 
	 * @param instances the dense instances to write.
	 * @param attFile the attribute file path.
	 * @param dataFile the data file path.
	 * @throws IOException
	 */
	public static void write(Instances instances, String attFile, String dataFile) throws IOException {
		List<Attribute> attributes = instances.getAttributes();
		PrintWriter out = new PrintWriter(attFile);
		for (Attribute attribute : attributes) {
			out.println(attribute);
		}
		out.println(instances.getTargetAttribute() + " (class)");
		out.flush();
		out.close();

		write(instances, dataFile);
	}

	/**
	 * Writes a set of dense/sparse instances to data file.
	 * 
	 * @param instances the dense instances to write.
	 * @param file the data file path.
	 * @throws IOException
	 */
	public static void write(Instances instances, String file) throws IOException {
		PrintWriter out = new PrintWriter(file);
		for (Instance instance : instances) {
			out.println(instance);
		}
		out.flush();
		out.close();
	}

}
