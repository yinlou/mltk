package mltk.core.io;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

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
	 * @return a pair of attributes and target attribute (null if no target attribute).
	 * @throws IOException
	 */
	public static Pair<List<Attribute>, Attribute> read(String attFile) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(attFile), 65535);
		Pair<List<Attribute>, Attribute> pair = read(br);
		br.close();

		return pair;
	}
	
	/**
	 * Reads attributes and class attribute from attribute file.
	 * 
	 * @param br the reader.
	 * @return a pair of attributes and target attribute (null if no target attribute).
	 * @throws IOException
	 */
	public static Pair<List<Attribute>, Attribute> read(BufferedReader br) throws IOException {
		List<Attribute> attributes = new ArrayList<Attribute>();
		Attribute targetAtt = null;
		Set<String> usedNames = new HashSet<>();
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
			if (line.indexOf(" (target)") != -1) {
				targetAtt = att;
				i--;
			} else {
				if (usedNames.contains(att.getName())) {
					throw new RuntimeException("Duplicate attribute name: " + att.getName());
				}
				usedNames.add(att.getName());
				attributes.add(att);
			}
			if (line.indexOf(" (x)") != -1) {
				att.setIndex(-1);
			}
		}

		return new Pair<List<Attribute>, Attribute>(attributes, targetAtt);
	}

}
