package mltk.predictor;

/**
 * Class for response distribution family. This class is used for GLMs/GAMs.
 * 
 * @author Yin Lou
 *
 */
public enum Family {

	GAUSSIAN("gaussian", LinkFunction.IDENTITY),
	BINOMIAL("binomial", LinkFunction.LOGIT);

	/**
	 * Parses an enumeration from a string.
	 * 
	 * @param name the family name.
	 * @return a parsed distribution.
	 */
	public static Family get(String name) {
		for (Family family : Family.values()) {
			if (name.startsWith(family.name)) {
				return family;
			}
		}
		throw new IllegalArgumentException("Invalid family name: " + name);
	}
	String name;
	
	LinkFunction link;
	
	Family(String name, LinkFunction link) {
		this.name = name;
		this.link = link;
	}

	/**
	 * Returns the default link function for this family.
	 * 
	 * @return the default link function for this family.
	 */
	public LinkFunction getDefaultLinkFunction() {
		return link;
	}

	/**
	 * Returns the string representation of this family with default link function.
	 */
	public String toString() {
		return name + "(" + link + ")";
	}

}
