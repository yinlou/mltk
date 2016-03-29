package mltk.cmdline.options;

import mltk.cmdline.Argument;

public class LearnerOptions {

	@Argument(name = "-r", description = "attribute file path")
	public String attPath = null;

	@Argument(name = "-t", description = "train set path", required = true)
	public String trainPath = null;

	@Argument(name = "-o", description = "output model path")
	public String outputModelPath = null;
	
	@Argument(name = "-V", description = "verbose (default: true)")
	public boolean verbose = true;
	
}
