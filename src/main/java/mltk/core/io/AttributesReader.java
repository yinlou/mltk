package mltk.core.io;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import mltk.core.Attribute;
import mltk.core.BinnedAttribute;
import mltk.core.NominalAttribute;
import mltk.core.NumericalAttribute;
import mltk.util.tuple.Pair;

/**
 * Class for reading attributes. It only reads in a list of attributes from the attribute file.
 * 
 * @author Yin Lou
 * 
 */
public class AttributesReader {

	/**
	 * Reads attributes and class attribute from attribute file.
	 * 
	 * @param attFile the attribute file.
	 * @return a pair of attributes and class attribute (null if no class attribute).
	 * @throws IOException
	 */
	public static Pair<List<Attribute>, Attribute> read(String attFile) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(attFile), 65535);
		List<Attribute> attributes = new ArrayList<Attribute>();
		Attribute clsAttr = null;
		for (int i = 0;; i++) {
			String line = br.readLine();
			if (line == null) {
				break;
			}
			Attribute att = null;
			if (line.indexOf("binned") != -1) {
				att = BinnedAttribute.parse(line);
			} else if (line.indexOf("{") != -1) {
				att = NominalAttribute.parse(line);
			} else {
				att = NumericalAttribute.parse(line);
			}
			att.setIndex(i);
			if (line.indexOf("(class)") != -1) {
				clsAttr = att;
				i--;
			} else {
				attributes.add(att);
			}
		}
		br.close();

		return new Pair<List<Attribute>, Attribute>(attributes, clsAttr);
	}

}
