package mltk.predictor.gam.tool;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
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
import mltk.predictor.gam.GAM;
import mltk.predictor.io.PredictorReader;

/**
 * Class for visualizing 1D and 2D components in a GAM.
 *
 * @author Yin Lou, updated by Sebastien Dubois
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

		@Override
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
	 * Generates data and plots for visualizing low dimensional components in a GAM.
	 *
	 * @param gam the GAM model.
	 * @param instances the training set.
	 * @param dirPath the directory path to write to.
	 * @param feats Set of variables used for the GAM
	 * @param scriptDir Directory with the R script files to plot
	 * @throws IOException
	 */
	public static void generateGnuplotScripts(GAM gam, Instances instances, String dirPath, Set<Integer> feats, String scriptDir)
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

		String cmd = "";

		for (int i = 0; i < terms.size(); i++) {
			int[] term = terms.get(i);
			Regressor regressor = regressors.get(i);
			if (term.length == 1 && (feats == null || feats.contains(term[0]))) {
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
				String filename = dir.getAbsolutePath() + File.separator + f.getName();
				PrintWriter dataout = new PrintWriter(filename + ".txt");
				dataout.println(f.getName() + "\ty");

				switch (f.getType()) {
					case BINNED:
						int numBins = ((BinnedAttribute) f).getNumBins();
						Bins bins = ((BinnedAttribute) f).getBins();
						double[] boundaries = bins.getBoundaries();
						double start = boundaries[0] - 1;
						if (boundaries.length >= 2) {
							start = boundaries[0] - (boundaries[1] - boundaries[0]);
						}
						List<Double> predList = new ArrayList<>();
						for (int j = 0; j < numBins; j++) {
							point.setValue(term[0], j);
							predList.add(regressor.regress(point));
						}
						point.setValue(term[0], 0);
						dataout.printf("%f\t%f\n", start, predList.get(0));
						for (int j = 0; j < numBins; j++) {
							point.setValue(term[0], j);
							dataout.printf("%f\t%f\n", boundaries[j], predList.get(j));
							if (j < numBins - 1) {
								dataout.printf("%f\t%f\n", boundaries[j], predList.get(j + 1));
							}
						}
						cmd = "Rscript " + scriptDir + File.separator + "plot_binned.R " + filename ;
						break;
					case NOMINAL:
						String[] states = ((NominalAttribute) f).getStates();
						for (int j = 0; j < states.length; j++) {
							point.setValue(term[0], j);
							dataout.printf("%s\t%f\n", states[j], regressor.regress(point));
						}
						cmd = "Rscript " + scriptDir + File.separator + "plot_cat.R " + filename ;
						break;
					default:
						System.out.println("Cannot plot attribute, neither BINNED nor NOMINAL");
						break;
//						Set<Double> values = new HashSet<>();
//						for (Instance instance : instances) {
//							values.add(instance.getValue(term[0]));
//						}
//						List<Double> list = new ArrayList<>(values);
//						Collections.sort(list);
//						out.printf("set xrange[%f:%f]\n", list.get(0), list.get(list.size() - 1));
//						if (regressor instanceof CubicSpline) {
//							CubicSpline spline = (CubicSpline) regressor;
//							out.println("z(x) = x < 0 ? 0 : x ** 3");
//							out.println("h(x, k) = z(x - k)");
//							double[] knots = spline.getKnots();
//							double[] w = spline.getCoefficients();
//							StringBuilder sb = new StringBuilder();
//							sb.append("plot ").append(spline.getIntercept());
//							sb.append(" + ").append(w[0]).append(" * x");
//							sb.append(" + ").append(w[1]).append(" * (x ** 2)");
//							sb.append(" + ").append(w[2]).append(" * (x ** 3)");
//							for (int j = 0; j < knots.length; j++) {
//								sb.append(" + ").append(w[j + 3]).append(" * ");
//								sb.append("h(x, ").append(knots[j]).append(")");
//							}
//							sb.append(" t \"\"");
//							out.println(sb.toString());
//						} else {
//							out.println("plot \"-\" u 1:2 w lp t \"\"");
//							for (double v : list) {
//								point.setValue(term[0], v);
//								out.printf("%f\t%f\n", v, regressor.regress(point));
//							}
//						}
//						break;
				}
				dataout.flush();
				dataout.close();

				Process process = Runtime.getRuntime().exec(cmd);
			    Scanner scanner = new Scanner(process.getInputStream());
			    while (scanner.hasNext()) {
			        System.out.println(scanner.nextLine());
			    }
			    scanner.close();

			} else if (term.length == 2) {
				Attribute f1 = attributes.get(term[0]);
				Attribute f2 = attributes.get(term[1]);

				String filename = dir.getAbsolutePath() + File.separator + f1.getName() + "__" + f2.getName();
				PrintWriter dataout = new PrintWriter(filename + ".txt");

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
				if(f1.getType() == Attribute.Type.BINNED && f2.getType() == Attribute.Type.BINNED) {
					dataout.println(f2.getName() + "\t" + f1.getName() + "\tz");
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
								dataout.print(start2 + "\t");
							} else {
								dataout.print(boundaries2[c] + "\t");
							}
							if (r == -1) {
								dataout.print(start1 + "\t");
							} else {
								dataout.print(boundaries1[r] + "\t");
							}
							dataout.println(gam.regress(point));
						}
					}
					cmd = "Rscript " + scriptDir + File.separator + "plot_binned_terms.R " + filename ;
				}
				else if(f1.getType() == Attribute.Type.NOMINAL && f2.getType() == Attribute.Type.NOMINAL) {
					dataout.println(f2.getName() + "\t" + f1.getName() + "\tz");
					String[] states1 = ((NominalAttribute) f1).getStates();
					String[] states2 = ((NominalAttribute) f2).getStates();

					for (int j = 0; j < states1.length; j++) {
						for(int k = 0; k < states2.length; k++) {
							point.setValue(term[0], j);
							point.setValue(term[1], k);
							dataout.print(states2[k] + "\t");
							dataout.print(states1[j] + "\t");
							dataout.printf("%f\n", regressor.regress(point));
						}
					}
					cmd = "Rscript " + scriptDir + File.separator + "plot_cat_terms.R " + filename ;
				}
				else {
					BinnedAttribute f;
					NominalAttribute nf;
					int idx_f;
					int idx_nf;
					if(f1.getType() == Attribute.Type.BINNED) {
						f = (BinnedAttribute) f1;
						nf = (NominalAttribute) f2;
						idx_f = 0;
						idx_nf = 1;
					}
					else {
						f = (BinnedAttribute) f2;
						nf = (NominalAttribute) f1;
						idx_f = 1;
						idx_nf = 0;
					}
					String[] states = nf.getStates();

					dataout.print("x");
					for(int j = 0; j < states.length; j++) {
						dataout.print("\t" + states[j]);
					}
					dataout.print("\n");

					int numBins = f.getNumBins();
					Bins bins = f.getBins();
					double[] boundaries = bins.getBoundaries();
					double start = boundaries[0] - 1;
					if (boundaries.length >= 2) {
						start = boundaries[0] - (boundaries[1] - boundaries[0]);
					}
					// start
					point.setValue(term[idx_f], 0);
					dataout.printf("%f", start);
					for(int k=0; k < states.length; k++) {
						point.setValue(term[idx_nf], k);
						dataout.printf("\t%f", regressor.regress(point));
					}
					dataout.print("\n");

					// cont
					for (int j = 0; j < numBins; j++) {
						point.setValue(term[idx_f], j);
						dataout.printf("%f", boundaries[j]);
						for(int k=0; k < states.length; k++) {
							point.setValue(term[idx_nf], k);
							dataout.printf("\t%f", regressor.regress(point));
						}
						dataout.print("\n");
						if (j < numBins - 1) {
							point.setValue(term[idx_f], j+1);
							dataout.printf("%f", boundaries[j]);
							for(int k=0; k < states.length; k++) {
								point.setValue(term[idx_nf], k);
								dataout.printf("\t%f", regressor.regress(point));
							}
							dataout.print("\n");
						}
					}
					cmd = "Rscript " +
						  scriptDir + File.separator + "plot_mixed_terms.R " +
						  filename + " " +
						  nf.getName() + " " + f.getName() ;
				}
				dataout.flush();
				dataout.close();

				Process process = Runtime.getRuntime().exec(cmd);
			    Scanner scanner = new Scanner(process.getInputStream());
			    while (scanner.hasNext()) {
			        System.out.println(scanner.nextLine());
			    }
			    scanner.close();
			}
		}
	}

	static class Options {

		@Argument(name = "-r", description = "attribute file path (default : binned_attr.txt)")
		String attPath = "binned_attr.txt";

		@Argument(name = "-d", description = "dataset path (default : binned_train.txt")
		String datasetPath = "binned_train.txt";

		@Argument(name = "-i", description = "input model path", required = true)
		String inputModelPath = null;

		@Argument(name = "-o", description = "output directory path", required = true)
		String dirPath = null;

		@Argument(name = "-s", description = "Rscripts directory (default : ../../utils)")
		String scriptDir = "../../utils";

		@Argument(name = "-f", description = "selected features path")
		String featPath = null;
	}

	/**
	 * <p>
	 *
	 * <pre>
	 * Usage: mltk.predictor.gam.tool.Visualizer
	 * -i	input model path
	 * -o	output directory path
	 * [-r]	attribute file path
	 * [-d]	dataset path
	 * [-f] features path
	 * [-s] Rscript directory (default : ../../utils)
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

		Set<Integer> feats = null;
		if(opts.featPath != null) {
			feats = new HashSet<Integer>();
			BufferedReader br = new BufferedReader(new FileReader(opts.featPath));
			String line = br.readLine();
			String[] data = line.split(", ");
			br.close();

			for(int i=0; i < data.length; i++) {
				feats.add(Integer.parseInt(data[i]));
			}
		}

		Visualizer.generateGnuplotScripts(gam, dataset, opts.dirPath, feats, opts.scriptDir);
	}

}
