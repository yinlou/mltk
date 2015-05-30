package mltk.core.io;

import java.util.ArrayList;
import java.util.List;

import mltk.core.Attribute;

/**
 * Structure for information on attributes in the data: 
 * list of active attributes, class attributes,  
 * columns corresponding to the active attributes in the data file
 * 
 * @author Daria Sorokina
 * 
 */

public class AttrInfo {
	public List<Attribute> attributes;
	public Attribute clsAttr;
	public List<Integer> columns;
	
	/**
	 * Default constructor 
	 */
	public AttrInfo() {
		this.attributes = new ArrayList<Attribute>();
		this.columns = new ArrayList<Integer>();
	}

	/**
	 * Maps names of attributes into their ids
	 * 
	 * @param names names of attributes
	 * @return a list of corresponding attribute ids.
	 */
	public List<Integer> getIds(List<String> names)	{
		List<Integer> ids = new ArrayList<>();
		for(String name: names)
			for(Attribute att: attributes)
				if(att.getName().equals(name)) {
					ids.add(att.getIndex());
					break;
				}
		return ids;
	}
}