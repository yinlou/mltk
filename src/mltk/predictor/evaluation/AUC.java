package mltk.predictor.evaluation;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Comparator;

import mltk.core.Instances;
import mltk.util.tuple.DoublePair;

/**
 * Class for evaluating area under ROC curve.
 *
 * @author Yin Lou
 *
 */
public class AUC extends SimpleMetric {

	private class DoublePairComparator implements Comparator<DoublePair> {

		@Override
		public int compare(DoublePair o1, DoublePair o2) {
			int cmp = Double.compare(o1.v1, o2.v1);
			if (cmp == 0) {
				cmp = Double.compare(o1.v2, o2.v2);
			}
			return cmp;
		}

	}

	public String outPath = null;

	/**
	 * Constructor.
	 */
	public AUC() {
		super(true);
	}

	@Override
	public double eval(double[] preds, double[] targets) {
		DoublePair[] a = new DoublePair[preds.length];
		for (int i = 0; i < preds.length; i++) {
			a[i] = new DoublePair(preds[i], targets[i]);
		}
		return eval(a);
	}

	@Override
	public double eval(double[] preds, Instances instances) {
		DoublePair[] a = new DoublePair[preds.length];
		for (int i = 0; i < preds.length; i++) {
			a[i] = new DoublePair(preds[i], instances.get(i).getTarget());
		}
		return eval(a);
	}

	protected double eval(DoublePair[] a) {
		Arrays.sort(a, new DoublePairComparator());
		double[] fraction = new double[a.length];
		for (int idx = 0; idx < fraction.length;) {
			int begin = idx;
			double pos = 0;
			for (; idx < fraction.length && a[idx].v1 == a[begin].v1; idx++) {
				pos += a[idx].v2;
			}
			double frac = pos / (idx - begin);
			for (int i = begin; i < idx; i++) {
				fraction[i] = frac;
			}
		}

		double tt = 0;
		double tf = 0;
		double ft = 0;
		double ff = 0;

		for (int i = 0; i < a.length; i++) {
			tf += a[i].v2;
			ff += 1 - a[i].v2;
		}

		double area = 0;
		double tpfPrev = 0;
		double fpfPrev = 0;

		double[] all_tpf = new double[a.length];
		double[] all_fpf = new double[a.length];
		double[] precisions = new double[a.length];

		// tt : True Positive, tf : False Negative, ff : True Negative, ft : False Positive
		for (int i = a.length - 1; i >= 0; i--) {
			tt += fraction[i];
			tf -= fraction[i];
			ft += 1 - fraction[i];
			ff -= 1 - fraction[i];
			double tpf = tt / (tt + tf);
			double fpf = 1.0 - ff / (ft + ff);
			if(outPath != null) {
				all_tpf[i] = tpf;
				all_fpf[i] = fpf;
				precisions[i] = tt / (tt + ft);
			}

			area += 0.5 * (tpf + tpfPrev) * (fpf - fpfPrev);
			tpfPrev = tpf;
			fpfPrev = fpf;
		}

		if(outPath != null) {
			try {
				PrintWriter out = new PrintWriter(outPath);

				StringBuilder sbuilder1 = new StringBuilder(Arrays.toString(all_tpf));
				// delete first and last bracket
				sbuilder1.deleteCharAt(0);
				sbuilder1.deleteCharAt(sbuilder1.length()-1);
				out.println(sbuilder1.toString());

				StringBuilder sbuilder2 = new StringBuilder(Arrays.toString(all_fpf));
				// delete first and last bracket
				sbuilder2.deleteCharAt(0);
				sbuilder2.deleteCharAt(sbuilder2.length()-1);
				out.println(sbuilder2.toString());

				StringBuilder sbuilder3 = new StringBuilder(Arrays.toString(precisions));
				// delete first and last bracket
				sbuilder3.deleteCharAt(0);
				sbuilder3.deleteCharAt(sbuilder3.length()-1);
				out.println(sbuilder3.toString());

				out.flush();
				out.close();

			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		return area;
	}

}
