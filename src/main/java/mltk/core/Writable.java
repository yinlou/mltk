package mltk.core;

import java.io.BufferedReader;
import java.io.PrintWriter;

/**
 * Writable interface.
 * 
 * @author Yin Lou
 * 
 */
public interface Writable {

	/**
	 * Reads in this object.
	 * 
	 * @param in the reader.
	 * @throws Exception
	 */
	void read(BufferedReader in) throws Exception;

	/**
	 * Writes this object.
	 * 
	 * @param out the writer.
	 * @throws Exception
	 */
	void write(PrintWriter out) throws Exception;

}
