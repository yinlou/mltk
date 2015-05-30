package mltk.core.io;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import mltk.core.Attribute;
import mltk.core.BinnedAttribute;
import mltk.core.NominalAttribute;
import mltk.core.NumericalAttribute;


/**
 * Class for reading attributes information. 
 * It reads in a list of attributes, list of inactive attributes and a class attribute
 *  from the attribute file.
 * 
 * @author Yin Lou, modified by Daria Sorokina
 * 
 */
public class AttributesReader {

	/**
	 * Reads attributes and class attribute from attribute file.
	 * 
	 * @param attFile the attribute file.
	 * @return attribute information instance.
	 * @throws IOException
	 */
	public static AttrInfo read(String attFile) throws IOException {
		
		Set<String> neverNames = new HashSet<String>();
		AttrInfo ainfo = new AttrInfo();
		
		BufferedReader br1 = new BufferedReader(new FileReader(attFile), 65535);
		String line = br1.readLine();		
		while ((line != null) && (line.indexOf("contexts:") == -1)) {
			line = br1.readLine();
		}
		if (line != null) {
			line = br1.readLine();
		}
		while (line != null) {
			neverNames.add(line.split(" ")[0]);
			line = br1.readLine();
		}
		br1.close();
		
		BufferedReader br2 = new BufferedReader(new FileReader(attFile), 65535);
		ainfo.clsAttr = null;
		for (int i = 0, col = 0;; i++, col++) {
			line = br2.readLine();
			if ((line == null) || (line.indexOf("contexts:") != -1)) {
				break;
			}
			String aname = line.split(": ")[0];
			if(neverNames.contains(aname)) {
				i--;
				continue;
			}
			Attribute att = null;
			if (line.indexOf("binned") != -1) {
				att = BinnedAttribute.parse(line);
			} else if (line.indexOf("{") != -1) {
				att = NominalAttribute.parse(line);
			} else {
				att = NumericalAttribute.parse(line);
			}
			if (line.indexOf("(class)") != -1) {
				att.setIndex(col);
				ainfo.clsAttr = att;
				i--;
			} else {
				att.setIndex(i);
				ainfo.attributes.add(att);
				ainfo.columns.add(col);
			}
		}
		br2.close();

		return ainfo;
	}

}
