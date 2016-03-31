package mltk.util;

/**
 * Class for weighted elements.
 * 
 * @author Yin Lou
 * 
 * @param <T> the type of the element.
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
		return Double.compare(this.weight, e.weight);
	}

}
