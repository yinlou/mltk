package mltk.predictor.evaluation;

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
public class AUC extends Metric {

	private class DoublePairComparator implements Comparator<DoublePair> {

		@Override
		public int compare(DoublePair o1, DoublePair o2) {
			if (o1.v1 < o2.v1) {
				return -1;
			} else if (o1.v1 > o2.v1) {
				return 1;
			} else {
				if (o1.v2 < o2.v2) {
					return -1;
				} else if (o1.v2 > o2.v2) {
					return 1;
				} else {
					return 0;
				}
			}
		}

	}

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

		for (int i = a.length - 1; i >= 0; i--) {
			tt += fraction[i];
			tf -= fraction[i];
			ft += 1 - fraction[i];
			ff -= 1 - fraction[i];
			double tpf = tt / (tt + tf);
			double fpf = 1.0 - ff / (ft + ff);
			area += 0.5 * (tpf + tpfPrev) * (fpf - fpfPrev);
			tpfPrev = tpf;
			fpfPrev = fpf;
		}

		return area;
	}

}
