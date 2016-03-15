package mltk.cmdline.options;

import mltk.cmdline.Argument;

public class LearnerOptions {

	@Argument(name = "-r", description = "attribute file path")
	public String attPath = "binned_attr.txt";

	@Argument(name = "-t", description = "train set path")
	public String trainPath = "binned_train.txt";

	@Argument(name = "-o", description = "output model path")
	public String outputModelPath = null;

	@Argument(name = "-V", description = "verbose (default: true)")
	public boolean verbose = true;

}
