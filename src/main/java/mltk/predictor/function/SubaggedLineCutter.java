package mltk.predictor.function;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import mltk.core.Attribute;
import mltk.core.BinnedAttribute;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.NominalAttribute;
import mltk.predictor.BaggedEnsemble;
import mltk.predictor.function.SubagSequence.SampleDelta;
import mltk.util.MathUtils;

/**
 * Class for cutting lines with subagging.
 * 
 * @author Yin Lou
 *
 */
public class SubaggedLineCutter extends EnsembledLineCutter {
	
	private int subsampleSize;
	
	private SubagSequence ss;
	
	/**
	 * Constructor.
	 */
	public SubaggedLineCutter() {
		this(false);
	}
	
	/**
	 * Constructor.
	 * 
	 * @param isClassification {@code true} if it is a classification problem.
	 */
	public SubaggedLineCutter(boolean isClassification) {
		attIndex = -1;
		this.isClassification = isClassification;
	}
	
	/**
	 * Creates internal subsamples.
	 * 
	 * @param n the size of the data set.
	 * @param subsampleRatio the subsample ratio.
	 * @param baggingIters the number of bagging iterations.
	 */
	public void createSubags(int n, double subsampleRatio, int baggingIters) {
		this.subsampleSize = (int) (n * subsampleRatio);
		ss = new SubagSequence(n, subsampleSize, baggingIters);
	}

	@Override
	public BaggedEnsemble build(Instances instances) {
		return build(instances, attIndex, numIntervals);
	}
	
	class Histogram {
		double[][] histogram;
		
		Histogram(double[][] histogram) {
			this.histogram = histogram;
		}
		
		Histogram copy() {
			double[][] newHistogram = new double[histogram.length][histogram[0].length];
			for (int i = 0; i < histogram.length; i++) {
				double[] hist = histogram[i];
				double[] newHist = newHistogram[i];
				for (int j = 0; j < hist.length; j++) {
					newHist[j] = hist[j];
				}
			}
			return new Histogram(newHistogram);
		}
	}
	
	/**
	 * Builds an 1D function ensemble.
	 * 
	 * @param instances the training set.
	 * @param attIndex the attribute index.
	 * @param numIntervals the number of intervals.
	 * @return an 1D function ensemble.
	 */
	public BaggedEnsemble build(Instances instances, int attIndex, int numIntervals) {
		Attribute attribute = instances.getAttributes().get(attIndex);
		return build(instances, attribute, numIntervals);
	}
	
	
	/**
	 * Builds an 1D function ensemble.
	 * 
	 * @param instances the training set.
	 * @param attribute the attribute.
	 * @param numIntervals the number of intervals.
	 * @return an 1D function ensemble.
	 */
	public BaggedEnsemble build(Instances instances, Attribute attribute, int numIntervals) {
		if (ss == null) {
			ss = new SubagSequence(instances.size(), subsampleSize, baggingIters);
		}
		
		SubagSequence.Sample[] samples = ss.samples;
		int[] start = ss.start;
		int[] end = ss.end;
		SampleDelta[] deltas = ss.deltas;
		int[] count = Arrays.copyOf(ss.count, ss.count.length);
		
		Function1D[] funcs = new Function1D[ss.samples.length];
		
		int attIndex = attribute.getIndex();
		
		double[] targets = new double[instances.size()];
		double[] fvalues = new double[instances.size()];
		double[] weights = new double[instances.size()];
		for (int i = 0; i < instances.size(); i++) {
			Instance instance = instances.get(i);
			targets[i] = instance.getTarget();
			fvalues[i] = instance.getValue(attribute);
			weights[i] = instance.getWeight();
		}

		if (attribute.getType() == Attribute.Type.NUMERIC) {
			throw new RuntimeException("Not implemented yet!");
		} else {
			int size = 0;
			if (attribute.getType() == Attribute.Type.BINNED) {
				size = ((BinnedAttribute) attribute).getNumBins();
			} else {
				size = ((NominalAttribute) attribute).getCardinality();
			}
			
			Histogram[] histograms = new Histogram[ss.samples.length];
			{// Initialization
				double[][] histogram = new double[size + 1][2];
				for (int index : samples[start[0]].indices) {
					double value = fvalues[index];
					double weight = weights[index];
					double target = targets[index];
					int idx = histogram.length - 1;
					if (!Double.isNaN(value)) {
						idx = (int) value;
					}
					if (isClassification) {
						histogram[idx][0] += target;
					} else {
						histogram[idx][0] += target * weight;
					}
					histogram[idx][1] += weight;
				}
				
				histograms[0] = new Histogram(histogram);
				funcs[0] = buildFromHistogram(attIndex, histograms[0]);
			}
			
			for (int k = 0; k < start.length; k++) {
				int from = start[k];
				Histogram histStart = histograms[from];
				count[from]--;
				
				// Generate histogram for its child
				int to = end[k];
				Histogram histEnd = null;
				if (count[from] == 0) {
					histEnd = histStart;
				} else {
					histEnd = histStart.copy();
				}
				SampleDelta delta = deltas[k];
				for (int index : delta.toAdd) {
					double value = fvalues[index];
					double weight = weights[index];
					double target = targets[index];
					int idx = histEnd.histogram.length - 1;
					if (!Double.isNaN(value)) {
						idx = (int) value;
					}
					if (isClassification) {
						histEnd.histogram[idx][0] += target;
					} else {
						histEnd.histogram[idx][0] += target * weight;
					}
					histEnd.histogram[idx][1] += weight;
				}
				for (int index : delta.toDel) {
					double value = fvalues[index];
					double weight = weights[index];
					double target = targets[index];
					int idx = histEnd.histogram.length - 1;
					if (!Double.isNaN(value)) {
						idx = (int) value;
					}
					if (isClassification) {
						histEnd.histogram[idx][0] -= target;
					} else {
						histEnd.histogram[idx][0] -= target * weight;
					}
					histEnd.histogram[idx][1] -= weight;
				}
				
				histograms[to] = histEnd;
				funcs[to] = buildFromHistogram(attIndex, histEnd);
			}
		}
		
		BaggedEnsemble ensemble = new BaggedEnsemble(ss.samples.length);
		for (Function1D func : funcs) {
			ensemble.add(func);
		}
		
		return ensemble;
	}
	
	protected Function1D buildFromHistogram(int attIndex, Histogram histogram) {
		final int size = histogram.histogram.length;
		List<double[]> histograms = new ArrayList<>(size);
		for (int i = 0; i < size - 1; i++) {
			if (!MathUtils.isZero(histogram.histogram[i][1])) {
				double[] hist = histogram.histogram[i];
				histograms.add(new double[] {i, hist[0], hist[1]});
			}
		}
		histograms.add(new double[] { Double.NaN, histogram.histogram[size - 1][0], histogram.histogram[size - 1][1] });
		
		Function1D func = LineCutter.build(attIndex, histograms, numIntervals);
		return func;
	}

}
