package mltk.predictor.gam.tool;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.core.Attribute;
import mltk.core.BinnedAttribute;
import mltk.core.Bins;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.NominalAttribute;
import mltk.core.io.InstancesReader;
import mltk.predictor.Regressor;
import mltk.predictor.function.CubicSpline;
import mltk.predictor.gam.GAM;
import mltk.predictor.io.PredictorReader;

/**
 * Class for visualizing 1D and 2D components in a GAM.
 * 
 * @author Yin Lou
 * 
 */
public class Visualizer {

	/**
	 * Enumeration of output terminals.
	 * 
	 * @author Yin Lou
	 * 
	 */
	public enum Terminal {

		/**
		 * PNG terminal.
		 */
		PNG("png"),
		/**
		 * PDF terminal.
		 */
		PDF("pdf");

		String term;

		Terminal(String term) {
			this.term = term;
		}

		public String toString() {
			return term;
		}

		/**
		 * Parses an enumeration from a string.
		 * 
		 * @param term the string.
		 * @return a parsed terminal.
		 */
		public static Terminal getEnum(String term) {
			for (Terminal re : Terminal.values()) {
				if (re.term.compareTo(term) == 0) {
					return re;
				}
			}
			throw new IllegalArgumentException("Invalid Terminal value: " + term);
		}

	}

	/**
	 * Generates a set of Gnuplot scripts for visualizing low dimensional components in a GAM.
	 * 
	 * @param gam the GAM model.
	 * @param instances the training set.
	 * @param dirPath the directory path to write to.
	 * @param outputTerminal output plot format (png or pdf).
	 * @throws IOException
	 */
	public static void generateGnuplotScripts(GAM gam, Instances instances, String dirPath, Terminal outputTerminal)
			throws IOException {
		List<Attribute> attributes = instances.getAttributes();
		List<int[]> terms = gam.getTerms();
		List<Regressor> regressors = gam.getRegressors();

		File dir = new File(dirPath);
		if (!dir.exists()) {
			dir.mkdirs();
		}

		double[] value = new double[attributes.size()];
		Instance point = new Instance(value);

		String terminal = outputTerminal.toString();
		for (int i = 0; i < terms.size(); i++) {
			int[] term = terms.get(i);
			Regressor regressor = regressors.get(i);
			if (term.length == 1) {
				Attribute f = attributes.get(term[0]);
				switch (f.getType()) {
					case BINNED:
						int numBins = ((BinnedAttribute) f).getNumBins();
						if (numBins == 1) {
							continue;
						}
						break;
					case NOMINAL:
						int numStates = ((NominalAttribute) f).getStates().length;
						if (numStates == 1) {
							continue;
						}
						break;
					default:
						break;
				}
				PrintWriter out = new PrintWriter(dir.getAbsolutePath() + File.separator + f.getName() + ".plt");
				out.printf("set term %s\n", terminal);
				out.printf("set output \"%s.%s\"\n", f.getName(), terminal);
				out.println("set datafile separator \"\t\"");
				out.println("set grid");
				switch (f.getType()) {
					case BINNED:
						int numBins = ((BinnedAttribute) f).getNumBins();
						Bins bins = ((BinnedAttribute) f).getBins();
						double[] boundaries = bins.getBoundaries();
						double start = boundaries[0] - 1;
						if (boundaries.length >= 2) {
							start = boundaries[0] - (boundaries[1] - boundaries[0]);
						}
						out.printf("set xrange[%f:%f]\n", start, boundaries[boundaries.length - 1]);
						out.println("plot \"-\" u 1:2 w l t \"\"");
						List<Double> predList = new ArrayList<>();
						for (int j = 0; j < numBins; j++) {
							point.setValue(term[0], j);
							predList.add(regressor.regress(point));
						}
						point.setValue(term[0], 0);
						out.printf("%f\t%f\n", start, predList.get(0));
						for (int j = 0; j < numBins; j++) {
							point.setValue(term[0], j);
							out.printf("%f\t%f\n", boundaries[j], predList.get(j));
							if (j < numBins - 1) {
								out.printf("%f\t%f\n", boundaries[j], predList.get(j + 1));
							}
						}
						out.println("e");
						break;
					case NOMINAL:
						out.println("set style data histogram");
						out.println("set style histogram cluster gap 1");
						out.println("set style fill solid border -1");
						out.println("set boxwidth 0.9");
						out.println("plot \"-\" u 2:xtic(1) t \"\"");
						out.println("set xtic rotate by -90");
						String[] states = ((NominalAttribute) f).getStates();
						for (int j = 0; j < states.length; j++) {
							point.setValue(term[0], j);
							out.printf("%s\t%f\n", states[j], regressor.regress(point));
						}
						out.println("e");
						break;
					default:
						Set<Double> values = new HashSet<>();
						for (Instance instance : instances) {
							values.add(instance.getValue(term[0]));
						}
						List<Double> list = new ArrayList<>(values);
						Collections.sort(list);
						out.printf("set xrange[%f:%f]\n", list.get(0), list.get(list.size() - 1));
						if (regressor instanceof CubicSpline) {
							CubicSpline spline = (CubicSpline) regressor;
							out.println("z(x) = x < 0 ? 0 : x ** 3");
							out.println("h(x, k) = z(x - k)");
							double[] knots = spline.getKnots();
							double[] w = spline.getCoefficients();
							StringBuilder sb = new StringBuilder();
							sb.append("plot ").append(spline.getIntercept());
							sb.append(" + ").append(w[0]).append(" * x");
							sb.append(" + ").append(w[1]).append(" * (x ** 2)");
							sb.append(" + ").append(w[2]).append(" * (x ** 3)");
							for (int j = 0; j < knots.length; j++) {
								sb.append(" + ").append(w[j + 3]).append(" * ");
								sb.append("h(x, ").append(knots[j]).append(")");
							}
							sb.append(" t \"\"");
							out.println(sb.toString());
						} else {
							out.println("plot \"-\" u 1:2 w lp t \"\"");
							for (double v : list) {
								point.setValue(term[0], v);
								out.printf("%f\t%f\n", v, regressor.regress(point));
							}
						}
						break;
				}
				out.flush();
				out.close();
			} else if (term.length == 2) {
				Attribute f1 = attributes.get(term[0]);
				Attribute f2 = attributes.get(term[1]);
				PrintWriter out = new PrintWriter(dir.getAbsolutePath() 
						+ File.separator + f1.getName() + "_" + f2.getName() 
						+ ".plt");
				out.printf("set term %s\n", terminal);
				out.printf("set output \"%s_%s.%s\"\n", f1.getName(), 
						f2.getName(), terminal);
				out.println("set datafile separator \"\t\"");
				int size1 = 0;
				if (f1.getType() == Attribute.Type.BINNED) {
					size1 = ((BinnedAttribute) f1).getNumBins();
				} else if (f1.getType() == Attribute.Type.NOMINAL) {
					size1 = ((NominalAttribute) f1).getCardinality();
				}
				int size2 = 0;
				if (f2.getType() == Attribute.Type.BINNED) {
					size2 = ((BinnedAttribute) f2).getNumBins();
				} else if (f2.getType() == Attribute.Type.NOMINAL) {
					size2 = ((NominalAttribute) f2).getCardinality();
				}
				if (f1.getType() == Attribute.Type.NOMINAL) {
					out.print("set ytics(");
					String[] states = ((NominalAttribute) f1).getStates();
					for (int j = 0; j < states.length; j++) {
						out.printf("%s %d", states[j], j);
					}
					out.println(")");
				}
				if (f2.getType() == Attribute.Type.NOMINAL) {
					out.print("set xtics(");
					String[] states = ((NominalAttribute) f2).getStates();
					for (int j = 0; j < states.length; j++) {
						out.printf("%s %d", states[j], j);
					}
					out.println(")");
				}
				out.println("set view map");
				out.println("set style data pm3d");
				out.println("set style function pm3d");
				out.println("set pm3d corners2color c4");
				Bins bins1 = ((BinnedAttribute) f1).getBins();
				double[] boundaries1 = bins1.getBoundaries();
				double start1 = boundaries1[0] - 1;
				if (boundaries1.length >= 2) {
					start1 = boundaries1[0] - (boundaries1[1] - boundaries1[0]);
				}
				Bins bins2 = ((BinnedAttribute) f2).getBins();
				double[] boundaries2 = bins2.getBoundaries();
				double start2 = boundaries2[0] - 1;
				if (boundaries2.length >= 2) {
					start2 = boundaries2[0] - (boundaries2[1] - boundaries2[0]);
				}
				out.printf("set yrange[%f:%f]\n", start1, boundaries1[boundaries1.length - 1]);
				out.printf("set xrange[%f:%f]\n", start2, boundaries2[boundaries2.length - 1]);
				out.println("splot \"-\"");
				for (int r = -1; r < size1; r++) {
					if (r == -1) {
						point.setValue(term[0], 0);
					} else {
						point.setValue(term[0], r);
					}
					for (int c = -1; c < size2; c++) {
						if (c == -1) {
							point.setValue(term[1], 0);
						} else {
							point.setValue(term[1], c);
						}
						if (c == -1) {
							out.print(start2 + "\t");
						} else {
							out.print(boundaries2[c] + "\t");
						}
						if (r == -1) {
							out.print(start1 + "\t");
						} else {
							out.print(boundaries1[r] + "\t");
						}
						out.println(gam.regress(point));
					}
					out.println();
				}
				out.println("e");
				out.flush();
				out.close();
			}
		}
	}

	static class Options {

		@Argument(name = "-r", description = "attribute file path", required = true)
		String attPath = null;

		@Argument(name = "-d", description = "dataset path", required = true)
		String datasetPath = null;

		@Argument(name = "-i", description = "input model path", required = true)
		String inputModelPath = null;

		@Argument(name = "-o", description = "output directory path", required = true)
		String dirPath = null;

		@Argument(name = "-t", description = "output terminal (default: png)")
		String terminal = "png";

	}

	/**
	 * <p>
	 * 
	 * <pre>
	 * Usage: mltk.predictor.gam.tool.Visualizer
	 * -r	attribute file path
	 * -d	dataset path
	 * -i	input model path
	 * -o	output directory path
	 * [-t]	output terminal (default: png)
	 * </pre>
	 * 
	 * </p>
	 * 
	 * @param args the command line arguments.
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(Visualizer.class, opts);
		try {
			parser.parse(args);
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}
		Instances dataset = InstancesReader.read(opts.attPath, opts.datasetPath);
		GAM gam = PredictorReader.read(opts.inputModelPath, GAM.class);

		Visualizer.generateGnuplotScripts(gam, dataset, opts.dirPath, Terminal.getEnum(opts.terminal));
	}

}
