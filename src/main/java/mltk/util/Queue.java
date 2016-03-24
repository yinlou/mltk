package mltk.util;

import java.util.LinkedList;

/**
 * Class for generic queues.
 * 
 * @author Yin Lou
 * 
 * @param <T>
 */
public class Queue<T> {

	protected LinkedList<T> list;

	/**
	 * Constructor.
	 */
	public Queue() {
		list = new LinkedList<T>();
	}

	/**
	 * Inserts an item to the queue.
	 * 
	 * @param item the item.
	 */
	public void enqueue(T item) {
		list.addLast(item);
	}

	/**
	 * Removes the first element in the queue.
	 * 
	 * @return the first element in the queue.
	 */
	public T dequeue() {
		T item = list.getFirst();
		list.removeFirst();
		return item;
	}

	/**
	 * Returns <code>true</code> if the queue is empty.
	 * 
	 * @return <code>true</code> if the queue is empty.
	 */
	public boolean isEmpty() {
		return list.size() == 0;
	}

}
