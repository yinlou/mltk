package mltk.predictor;

import java.io.BufferedReader;
import java.io.PrintWriter;

import mltk.core.Copyable;
import mltk.core.Writable;

/**
 * Interface for predictors.
 * 
 * @author Yin Lou
 * 
 */
public interface Predictor extends Writable, Copyable<Predictor> {
    
    /**
     * Reads in this predictor. This method is used in {@link mltk.predictor.io.PredictorReader}.
     * 
     * @param in the reader.
     * @throws Exception
     */
    public void read(BufferedReader in) throws Exception;

    /**
     * Writes this predictor. This method is used in {@link mltk.predictor.io.PredictorWriter}.
     * 
     * @param out the writer.
     * @throws Exception
     */
    public void write(PrintWriter out) throws Exception;

}
