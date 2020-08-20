package mltk.predictor.function;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import mltk.core.Attribute;
import mltk.core.BinnedAttribute;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.NominalAttribute;
import mltk.core.Sampling;
import mltk.predictor.BaggedEnsemble;
import mltk.util.tuple.IntPair;
import mltk.util.Element;
import mltk.util.MathUtils;

/**
 * Class for cutting lines with bagging.
 * 
 * @author Yin Lou
 *
 */
public class BaggedLineCutter extends EnsembledLineCutter {
	
	private List<IntPair[]> samples;
	
	/**
	 * Constructor.
	 */
	public BaggedLineCutter() {
		this(false);
	}
	
	/**
	 * Constructor.
	 * 
	 * @param isClassification {@code true} if it is a classification problem.
	 */
	public BaggedLineCutter(boolean isClassification) {
		attIndex = -1;
		this.isClassification = isClassification;
	}
	
	/**
	 * Creates internal bootstrap samples.
	 * 
	 * @param n the size of the dataset to sample.
	 * @param baggingIters the number of bagging iterations.
	 */
	public void createBootstrapSamples(int n, int baggingIters) {
		samples = new ArrayList<>(baggingIters);
		if (baggingIters <= 0) {
			// No bagging
			IntPair[] indices = new IntPair[n];
			for (int i = 0; i < n; i++) {
				indices[i] = new IntPair(i, 1);
			}
			samples.add(indices);
		} else {
			for (int b = 0; b < baggingIters; b++) {
				samples.add(Sampling.createBootstrapSampleIndices(n));
			}
		}
	}

	@Override
	public BaggedEnsemble build(Instances instances) {
		return build(instances, attIndex, numIntervals);
	}

	@Override
	public BaggedEnsemble build(Instances instances, Attribute attribute, int numIntervals) {
		int attIndex = attribute.getIndex();
		
		if (samples == null) {
			createBootstrapSamples(instances.size(), baggingIters);
		}
		
		BaggedEnsemble ensemble = new BaggedEnsemble(samples.size());
		
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
			for (IntPair[] indices : samples) {
				double sumRespOnMV = 0.0;
				double sumWeightOnMV = 0.0;
				
				List<Element<double[]>> pairs = new ArrayList<>(instances.size());
				for (IntPair entry : indices) {
					int index = entry.v1;
					int w = entry.v2;
					double weight = weights[index];
					double value = fvalues[index];
					double target = targets[index];
					if (!Double.isNaN(value)) {
						if (isClassification) {
							pairs.add(new Element<double[]>(new double[] { target * w, weight * w }, value));
						} else {
							pairs.add(new Element<double[]>(new double[] { target * weight * w, weight * w }, value));
						}
					} else {
						if (isClassification) {
							sumRespOnMV += target * w;
						} else {
							sumRespOnMV += target * weight * w;
						}
						sumWeightOnMV += weight * w;
					}
				}
				
				Collections.sort(pairs);
				List<double[]> histograms = new ArrayList<>();
				LineCutter.getHistograms(pairs, histograms);
				histograms.add(new double[] { Double.NaN, sumRespOnMV, sumWeightOnMV });
				
				Function1D func = LineCutter.build(attIndex, histograms, numIntervals);
				ensemble.add(func);
			}
		} else {
			int size = 0;
			if (attribute.getType() == Attribute.Type.BINNED) {
				size = ((BinnedAttribute) attribute).getNumBins();
			} else {
				size = ((NominalAttribute) attribute).getCardinality();
			}
			for (IntPair[] indices : samples) {
				List<double[]> histograms;
				double sumRespOnMV = 0.0;
				double sumWeightOnMV = 0.0;
				
				double[][] histogram = new double[size][3];
				for (IntPair entry : indices) {
					int index = entry.v1;
					int w = entry.v2;
					double weight = weights[index];
					double value = fvalues[index];
					double target = targets[index];
					
					if (!Double.isNaN(value)) {
						int idx = (int) value;
						if (isClassification) {
							histogram[idx][0] += target * w;
						} else {
							histogram[idx][0] += target * weight * w;
						}
						histogram[idx][1] += weight * w;
					} else {
						if (isClassification) {
							sumRespOnMV += target * w;
						} else {
							sumRespOnMV += target * weight * w;
						}
						sumWeightOnMV += weight * w;
					}
				}

				histograms = new ArrayList<>(histogram.length + 1);
				for (int i = 0; i < histogram.length; i++) {
					if (!MathUtils.isZero(histogram[i][1])) {
						double[] hist = histogram[i];
						histograms.add(new double[] {i, hist[0], hist[1]});
					}
				}
				histograms.add(new double[] { Double.NaN, sumRespOnMV, sumWeightOnMV });
				
				Function1D func = LineCutter.build(attIndex, histograms, numIntervals);
				ensemble.add(func);
			}
		}
		
		return ensemble;
	}

}
