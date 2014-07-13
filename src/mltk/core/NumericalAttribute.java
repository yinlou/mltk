package mltk.core;

/**
 * Class for numerical attributes.
 * 
 * @author Yin Lou
 * 
 */
public class NumericalAttribute extends Attribute {

	/**
	 * Constructor.
	 * 
	 * @param name the name of this attribute.
	 */
	public NumericalAttribute(String name) {
		this(name, -1);
	}

	/**
	 * Constructor.
	 * 
	 * @param name the name of this attribute.
	 * @param index the index of this attribute.
	 */
	public NumericalAttribute(String name, int index) {
		this.name = name;
		this.index = index;
		this.type = Type.NUMERIC;
	}

	public NumericalAttribute copy() {
		NumericalAttribute copy = new NumericalAttribute(this.name);
		copy.index = this.index;
		return copy;
	}

	public String toString() {
		return name + ": cont";
	}

	/**
	 * Parses a numerical attribute object from a string.
	 * 
	 * @param str the string.
	 * @return a parsed numerical attribute.
	 */
	public static NumericalAttribute parse(String str) {
		String[] data = str.split(": ");
		return new NumericalAttribute(data[0]);
	}

}
