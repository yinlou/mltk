package mltk.core;

/**
 * Class for nominal attributes.
 * 
 * @author Yin Lou
 * 
 */
public class NominalAttribute extends Attribute {

	protected String[] states;

	/**
	 * Constructor.
	 * 
	 * @param name the name of this attribute.
	 * @param states the states for this attribute.
	 */
	public NominalAttribute(String name, String[] states) {
		this(name, states, -1);
	}

	/**
	 * Constructor.
	 * 
	 * @param name the name of this attribute.
	 * @param states the states for this attribute.
	 * @param index the index of this attribute.
	 */
	public NominalAttribute(String name, String[] states, int index) {
		this.name = name;
		this.states = states;
		this.index = index;
		this.type = Type.NOMINAL;
	}

	public NominalAttribute copy() {
		NominalAttribute copy = new NominalAttribute(name, states);
		copy.index = this.index;
		return copy;
	}

	/**
	 * Returns the cardinality of this attribute.
	 * 
	 * @return the cardinality of this attribute.
	 */
	public int getCardinality() {
		return states.length;
	}

	/**
	 * Returns the state given an index.
	 * 
	 * @param index the index.
	 * @return the state given an index.
	 */
	public String getState(int index) {
		return states[index];
	}

	/**
	 * Returns the states.
	 * 
	 * @return the states.
	 */
	public String[] getStates() {
		return states;
	}

	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(name).append(": {").append(states[0]);
		for (int i = 1; i < states.length; i++) {
			sb.append(", ").append(states[i]);
		}
		sb.append("}");
		return sb.toString();
	}

	/**
	 * Parses a nominal attribute ojbect from a string.
	 * 
	 * @param str the string.
	 * @return a parsed nominal attribute.
	 */
	public static NominalAttribute parse(String str) {
		String[] data = str.split(": ");
		int start = data[1].indexOf('{') + 1;
		int end = data[1].indexOf('}');
		String[] states = data[1].substring(start, end).split(",");
		for (int j = 0; j < states.length; j++) {
			states[j] = states[j].trim();
		}
		return new NominalAttribute(data[0], states);
	}

}
