package mltk.predictor.gam.tool;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
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
import mltk.predictor.function.Array1D;
import mltk.predictor.function.Array2D;
import mltk.predictor.function.CubicSpline;
import mltk.predictor.function.Function1D;
import mltk.predictor.gam.GAM;
import mltk.predictor.io.PredictorReader;
import mltk.util.MathUtils;

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
		int p = -1;
		Map<Integer, Attribute> attMap = new HashMap<>(attributes.size());
		for (Attribute attribute : attributes) {
			int attIndex = attribute.getIndex();
			attMap.put(attIndex, attribute);
			if (attIndex > p) {
				p = attIndex;
			}
		}
		p++;
		
		List<int[]> terms = gam.getTerms();
		List<Regressor> regressors = gam.getRegressors();

		File dir = new File(dirPath);
		if (!dir.exists()) {
			dir.mkdirs();
		}

		double[] value = new double[p];
		Instance point = new Instance(value);

		String terminal = outputTerminal.toString();
		for (int i = 0; i < terms.size(); i++) {
			int[] term = terms.get(i);
			Regressor regressor = regressors.get(i);
			if (term.length == 1) {
				Attribute f = attMap.get(term[0]);
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
				Double predictionOnMV = null;
				if (regressor instanceof Function1D) {
					double predOnMV = ((Function1D) regressor).getPredictionOnMV();
					if (!MathUtils.isZero(predOnMV)) {
						predictionOnMV = predOnMV;
					}
				} else if (regressor instanceof Array1D) {
					double predOnMV = ((Array1D) regressor).getPredictionOnMV();
					if (!MathUtils.isZero(predOnMV)) {
						predictionOnMV = predOnMV;
					}
				}
				PrintWriter out = new PrintWriter(dir.getAbsolutePath() + File.separator + f.getName() + ".plt");
				out.printf("set term %s\n", terminal);
				out.printf("set output \"%s.%s\"\n", f.getName(), terminal);
				out.println("set datafile separator \"\t\"");
				out.println("set grid");
				if (predictionOnMV != null) {
					out.println("set multiplot layout 1,2 rowsfirst");
				}
				// Plot main function
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
						List<Double> predList = new ArrayList<>();
						for (int j = 0; j < numBins; j++) {
							point.setValue(term[0], j);
							predList.add(regressor.regress(point));
						}
						point.setValue(term[0], 0);
						{// Writing plot data to file
							String fileName = f.getName() + ".dat";
							out.println("plot \"" + fileName + "\" u 1:2 w l t \"\"");
							PrintWriter writer = new PrintWriter(dir.getAbsolutePath() + File.separator + fileName);
							writer.printf("%f\t%f\n", start, predList.get(0));
							for (int j = 0; j < numBins; j++) {
								point.setValue(term[0], j);
								writer.printf("%f\t%f\n", boundaries[j], predList.get(j));
								if (j < numBins - 1) {
									writer.printf("%f\t%f\n", boundaries[j], predList.get(j + 1));
								}
							}
							writer.flush();
							writer.close();
						}
						break;
					case NOMINAL:
						out.println("set style data histogram");
						out.println("set style histogram cluster gap 1");
						out.println("set style fill solid border -1");
						out.println("set boxwidth 0.9");
						out.println("set xtic rotate by -90");
						String[] states = ((NominalAttribute) f).getStates();
						{// Writing plot data to file
							String fileName = f.getName() + ".dat";
							out.println("plot \"" + fileName + "\" u 2:xtic(1) t \"\"");
							PrintWriter writer = new PrintWriter(dir.getAbsolutePath() + File.separator + fileName);
							for (int j = 0; j < states.length; j++) {
								point.setValue(term[0], j);
								writer.printf("%s\t%f\n", states[j], regressor.regress(point));
							}
							writer.flush();
							writer.close();
						}
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
							out.println("e");
						}
						break;
				}
				// Plot prediction on missing value
				if (predictionOnMV != null) {
					out.println("set style fill solid border -1");
					out.println("set xtic rotate by 0");
					out.println("plot \"-\" using 2:xtic(1) with histogram t \"\"");
					out.println("missing value\t" + predictionOnMV);
					out.println("e");
				}
				out.flush();
				out.close();
			} else if (term.length == 2) {
				Attribute f1 = attMap.get(term[0]);
				Attribute f2 = attMap.get(term[1]);
				String fileName = f1.getName() + "_" + f2.getName();
				PrintWriter out = new PrintWriter(dir.getAbsolutePath() 
						+ File.separator + fileName + ".plt");
				out.printf("set term %s\n", terminal);
				out.printf("set output \"%s_%s.%s\"\n", f1.getName(), 
						f2.getName(), terminal);
				out.println("set datafile separator \"\t\"");
				int numRow = 1;
				int numCol = 1;
				double[] predictionsOnMV1 = null;
				double[] predictionsOnMV2 = null;
				double predictionsOnMV12 = 0.0;
				if (regressor instanceof Array2D) {
					Array2D ary2d = (Array2D) regressor;
					for (double v : ary2d.getPredictionsOnMV1()) {
						if (!MathUtils.isZero(v)) {
							numRow = 2;
							predictionsOnMV1 = ary2d.getPredictionsOnMV1();
							break;
						}
					}
					for (double v : ary2d.getPredictionsOnMV2()) {
						if (!MathUtils.isZero(v)) {
							numCol = 2;
							predictionsOnMV2 = ary2d.getPredictionsOnMV2();
							break;
						}
					}
					if (!MathUtils.isZero(ary2d.getPredictionOnMV12())) {
						numRow = 2;
						numCol = 2;
						predictionsOnMV12 = ary2d.getPredictionOnMV12();
					}
				}
				if (numRow > 1 || numCol > 1) {
					out.printf("set multiplot layout %d,%d rowsfirst\n", numRow, numCol);
				}
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
					for (int j = 0; j < states.length - 1; j++) {
						out.printf("\"%s\" %d, ", states[j], j);
					}
					out.printf("\"%s\" %d)\n", states[states.length - 1], states.length - 1);
				}
				if (f2.getType() == Attribute.Type.NOMINAL) {
					out.print("set xtics(");
					String[] states = ((NominalAttribute) f2).getStates();
					for (int j = 0; j < states.length - 1; j++) {
						out.printf("\"%s\" %d, ", states[j], j);
					}
					out.printf("\"%s\" %d) rotate\n", states[states.length - 1], states.length - 1);
				}
				out.println("unset border");
				out.println("set view map");
				out.println("set style data pm3d");
				out.println("set style function pm3d");
				out.println("set pm3d corners2color c4");
				double[] rangeY = null;
				double[] valueY = null;
				double[] rangeX = null;
				double[] valueX = null;
				if (f1 instanceof BinnedAttribute) {
					rangeY = new double[size1 + 1];
					valueY = new double[size1 + 1];
					Bins bins = ((BinnedAttribute) f1).getBins();
					double[] boundaries = bins.getBoundaries();
					double start = boundaries[0] - 1;
					if (boundaries.length >= 2) {
						start = boundaries[0] - (boundaries[1] - boundaries[0]);
					}
					
					rangeY[0] = start;
					valueY[0] = 0;
					for (int k = 0; k < boundaries.length; k++) {
						rangeY[k + 1] = boundaries[k];
						valueY[k + 1] = k;
					}
				} else if (f1 instanceof NominalAttribute) {
					rangeY = new double[size1];
					valueY = new double[size1];
					for (int k = 0; k < size1; k++) {
						rangeY[k] = k;
						valueY[k] = k;
					}
				}
				out.printf("set yrange[%f:%f]\n", rangeY[0] - 1, rangeY[rangeY.length - 1] + 1);
				if (f2 instanceof BinnedAttribute) {
					rangeX = new double[size2 + 1];
					valueX = new double[size2 + 1];
					Bins bins = ((BinnedAttribute) f2).getBins();
					double[] boundaries = bins.getBoundaries();
					double start = boundaries[0] - 1;
					if (boundaries.length >= 2) {
						start = boundaries[0] - (boundaries[1] - boundaries[0]);
					}
					
					rangeX[0] = start;
					valueX[0] = 0;
					for (int k = 0; k < boundaries.length; k++) {
						rangeX[k + 1] = boundaries[k];
						valueX[k + 1] = k;
					}
				} else if (f2 instanceof NominalAttribute) {
					rangeX = new double[size2];
					valueX = new double[size2];
					for (int k = 0; k < size2; k++) {
						rangeX[k] = k;
						valueX[k] = k;
					}
				}
				out.printf("set xrange[%f:%f]\n", rangeX[0] - 1, rangeX[rangeX.length - 1] + 1);
				
				out.println("splot \"" + fileName + ".dat\" with image t \"\"");
				PrintWriter writer = new PrintWriter(dir.getAbsolutePath() + File.separator + fileName + ".dat");
				for (int r = 0; r < rangeY.length; r++) {
					point.setValue(term[0], valueY[r]);
					for (int c = 0; c < rangeX.length; c++) {
						point.setValue(term[1], valueX[c]);
						writer.println(rangeX[c] + "\t" + rangeY[r] + "\t" + gam.regress(point));
					}
					writer.println();
				}
				writer.flush();
				writer.close();
				
				if (numRow > 1 || numCol > 1) {
					// multiplot is on
					if (predictionsOnMV2 != null) {
						out.println("reset");
						writer = new PrintWriter(dir.getAbsolutePath() + File.separator + fileName + "_mv2.dat");
						if (f1.getType() == Attribute.Type.NOMINAL) {
							String[] states = ((NominalAttribute) f1).getStates();
							out.println("set style data histogram");
							out.println("set style histogram cluster gap 1");
							out.println("set style fill solid border -1");
							out.println("set boxwidth 0.9");
							out.println("set xtic rotate by -90");
							out.println("plot \"" + fileName + "_mv2.dat\" u 2:xtic(1) t \"\"");
							
							for (int k = 0; k < states.length; k++) {
								writer.println(states[k] + "\t" + predictionsOnMV2[k]);
							}
						} else if (f1.getType() == Attribute.Type.BINNED) {
							out.println("plot \"" + fileName + "_mv2.dat\" u 1:2 w l t \"\"");
							writer.println(rangeY[0] + "\t" + predictionsOnMV2[0]);
							for (int k = 0; k < predictionsOnMV2.length; k++) {
								writer.println(rangeY[k + 1] + "\t" + predictionsOnMV2[k]);
							}
						}
						writer.flush();
						writer.close();
					} else if (numCol > 1) {
						out.println("set multiplot next");
					}
					if (predictionsOnMV1 != null) {
						out.println("reset");
						writer = new PrintWriter(dir.getAbsolutePath() + File.separator + fileName + "_mv1.dat");
						if (f2.getType() == Attribute.Type.NOMINAL) {
							String[] states = ((NominalAttribute) f2).getStates();
							out.println("set style data histogram");
							out.println("set style histogram cluster gap 1");
							out.println("set style fill solid border -1");
							out.println("set boxwidth 0.9");
							out.println("set xtic rotate by -90");
							out.println("plot \"" + fileName + "_mv1.dat\" u 2:xtic(1) t \"\"");
							
							for (int k = 0; k < states.length; k++) {
								writer.println(states[k] + "\t" + predictionsOnMV1[k]);
							}
						} else if (f2.getType() == Attribute.Type.BINNED) {
							out.println("plot \"" + fileName + "_mv1.dat\" u 1:2 w l t \"\"");
							writer.println(rangeX[0] + "\t" + predictionsOnMV1[0]);
							for (int k = 0; k < predictionsOnMV1.length; k++) {
								writer.println(rangeX[k + 1] + "\t" + predictionsOnMV1[k]);
							}
						}
						writer.flush();
						writer.close();
					} else if (numRow > 1) {
						out.println("set multiplot next");
					}
					if (!MathUtils.isZero(predictionsOnMV12)) {
						out.println("reset");
						out.println("set datafile separator \"\t\"");
						out.println("set style fill solid border -1");
						out.println("set xtic rotate by 0");
						out.println("plot \"-\" using 2:xtic(1) with histogram t \"\"");
						out.println("missing value\t" + predictionsOnMV12);
						out.println("e");
					} else if (numRow > 1 && numCol > 1) {
						out.println("set multiplot next");
					}
				}
				
				
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
	 * Generates scripts for visualizing GAMs.
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
