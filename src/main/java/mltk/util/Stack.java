package mltk.util;

import java.util.ArrayList;
import java.util.EmptyStackException;
import java.util.List;

/**
 * Class for generic stacks.
 * 
 * @author Yin Lou
 * 
 * @param <T>
 */
public class Stack<T> {

	protected List<T> list;

	/**
	 * Constructor.
	 */
	public Stack() {
		this.list = new ArrayList<T>();
	}

	/**
	 * Inserts an item into the stack.
	 * 
	 * @param item the item.
	 */
	public void push(T item) {
		list.add(item);
	}

	/**
	 * Looks at the object at the top of this stack without removing it from the stack.
	 * 
	 * @return the top element in the stack.
	 */
	public T peek() {
		if (list.size() == 0) {
			throw new EmptyStackException();
		}
		return list.get(list.size() - 1);
	}

	/**
	 * Removes the object at the top of this stack and returns that object as the value of this function.
	 * 
	 * @return the top element in the stack.
	 */
	public T pop() {
		T item = peek();
		list.remove(list.size() - 1);
		return item;
	}

	/**
	 * Returns <code>true</code> if the stack is empty.
	 * 
	 * @return <code>true</code> if the stack is empty.
	 */
	public boolean isEmpty() {
		return list.size() == 0;
	}

}
