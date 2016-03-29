package mltk.predictor;

import mltk.util.MathUtils;

/**
 * Class for link functions.
 * 
 * @author Yin Lou
 *
 */
public enum LinkFunction {
	
	IDENTITY("identity"),
	LOGIT("logit");
	
	/**
	 * Parses a link function from a string.
	 * 
	 * @param name the name of the function.
	 * @return a parsed link function.
	 */
	public static LinkFunction get(String name) {
		for (LinkFunction link : LinkFunction.values()) {
			if (link.name.startsWith(name)) {
				return link;
			}
		}
		throw new IllegalArgumentException("Unknown link function: " + name);
	}
	
	String name;
	
	LinkFunction(String name) {
		this.name = name;
	}
	
	/**
	 * Applies the inverse of this link function.
	 * 
	 * @param x the argument.
	 * @return the inverse of this link function.
	 */
	public double applyInverse(double x) {
		double r = 0;
		switch (this) {
			case IDENTITY:
				r = x;
				break;
			case LOGIT:
				r = MathUtils.sigmoid(x);
				break;
			default:
				break;
		}
		return r;
	}
	
	/**
	 * Returns the string representation of this link function.
	 */
	public String toString() {
		return name;
	}
	
}
