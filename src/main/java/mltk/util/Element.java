package mltk.util;

/**
 * Class for weighted elements.
 * 
 * @author Yin Lou
 * 
 * @param <T>
 */
public class Element<T> implements Comparable<Element<T>> {

	public T element;
	public double weight;

	/**
	 * Constructs a weighted element.
	 * 
	 * @param element the element.
	 * @param weight the weight.
	 */
	public Element(T element, double weight) {
		this.element = element;
		this.weight = weight;
	}

	@Override
	public int compareTo(Element<T> e) {
		if (this.weight < e.weight) {
			return -1;
		} else if (this.weight > e.weight) {
			return 1;
		} else {
			return 0;
		}
	}

}
