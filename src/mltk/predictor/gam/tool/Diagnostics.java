package mltk.predictor.gam.tool;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.io.InstancesReader;
import mltk.predictor.Regressor;
import mltk.predictor.gam.GAM;
import mltk.predictor.io.PredictorReader;
import mltk.util.StatUtils;
import mltk.util.Element;

/**
 * Class for GAM diagnostics.
 *
 * @author Yin Lou
 *
 */
public class Diagnostics {

	/**
	 * Computes the weights for each term in a GAM.
	 *
	 * @param gam the GAM model.
	 * @param instances the training set.
	 * @return the list of weights for each term in a GAM.
	 */
	public static List<Element<int[]>> diagnose(GAM gam, Instances instances) {
		List<Element<int[]>> list = new ArrayList<>();
		Map<int[], List<Regressor>> map = new HashMap<>();
		List<int[]> terms = gam.getTerms();
		List<Regressor> regressors = gam.getRegressors();
		for (int i = 0; i < terms.size(); i++) {
			int[] term = terms.get(i);
			if (!map.containsKey(term)) {
				map.put(term, new ArrayList<Regressor>());
			}
			Regressor regressor = regressors.get(i);
			map.get(term).add(regressor);
		}

		double[] predictions = new double[instances.size()];
		for (int[] term : map.keySet()) {
			List<Regressor> regressorList = map.get(term);
			for (int i = 0; i < instances.size(); i++) {
				predictions[i] = 0;
				Instance instance = instances.get(i);
				for (Regressor regressor : regressorList) {
					predictions[i] += regressor.regress(instance);
				}
			}
			double weight = StatUtils.variance(predictions);
			list.add(new Element<int[]>(term, weight));
		}

		return list;
	}

	static class Options {

		@Argument(name = "-r", description = "attribute file path")
		String attPath = "binned_attr.txt";

		@Argument(name = "-d", description = "dataset path")
		String datasetPath = "binned_train.txt";

		@Argument(name = "-i", description = "input model path", required = true)
		String inputModelPath = null;

		@Argument(name = "-o", description = "output path", required = true)
		String outputPath = null;

	}

	/**
	 * <p>
	 *
	 * <pre>
	 * Usage: mltk.predictor.gam.tool.Diagnostics
	 * -d	dataset path
	 * -i	input model path
	 * -o	output path
	 * [-r]	attribute file path
	 * </pre>
	 *
	 * </p>
	 *
	 * @param args the command line arguments.
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(Diagnostics.class, opts);
		try {
			parser.parse(args);
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}
		Instances dataset = InstancesReader.read(opts.attPath, opts.datasetPath);
		GAM gam = PredictorReader.read(opts.inputModelPath, GAM.class);

		List<Element<int[]>> list = Diagnostics.diagnose(gam, dataset);
		Collections.sort(list);
		Collections.reverse(list);

		PrintWriter out = new PrintWriter(opts.outputPath);
		StringBuilder sbuilder1 = new StringBuilder();
		StringBuilder sbuilder2 = new StringBuilder();
		for (Element<int[]> element : list) {
			int[] term = element.element;
			double weight = element.weight;

			if(term.length == 1) {
				sbuilder1.append(term[0] + ",");
				sbuilder2.append(weight + ",");
			}
			else if(term.length == 2) {
				sbuilder1.append(term[0] + ":" + term[1] + ",");
				sbuilder2.append(weight + ",");
			}
			else {
				System.out.println("ERROR contain terms with more than two variables");
			}


		}
		// remove last ','
		sbuilder1.deleteCharAt(sbuilder1.length()-1);
		sbuilder2.deleteCharAt(sbuilder2.length()-1);

		// print to file
		out.println(sbuilder1.toString());
		out.println(sbuilder2.toString());

		out.flush();
		out.close();
	}

}
