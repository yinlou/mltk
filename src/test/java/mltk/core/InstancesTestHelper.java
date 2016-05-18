package mltk.core;

import java.util.ArrayList;
import java.util.List;

public class InstancesTestHelper {
	
	private static InstancesTestHelper instance = null;
	
	private Instances denseClaDataset;
	private Instances denseRegDataset;
	
	public static InstancesTestHelper getInstance() {
		if (instance == null) {
			instance = new InstancesTestHelper();
		}
		return instance;
	}
	
	public Instances getDenseClassificationDataset() {
		return denseClaDataset;
	}
	
	public Instances getDenseRegressionDataset() {
		return denseRegDataset;
	}
	
	private InstancesTestHelper() {
		List<Attribute> attributes = new ArrayList<>();
		
		NumericalAttribute f1 = new NumericalAttribute("f1", 0);
		NominalAttribute f2 = new NominalAttribute("f2", new String[] {"a", "b", "c"}, 1);
		BinnedAttribute f3 = new BinnedAttribute("f3", 256, 2);
		Bins bins = new Bins(new double[] {1, 5, 6}, new double[] {0.5, 2.5, 3});
		BinnedAttribute f4 = new BinnedAttribute("f4", bins, 3);
		
		attributes.add(f1);
		attributes.add(f2);
		attributes.add(f3);
		attributes.add(f4);
		
		Attribute claTarget = new NominalAttribute("target", new String[] {"0", "1"});
		Attribute regTarget = new NumericalAttribute("target");
		
		denseClaDataset = new Instances(attributes, claTarget);
		for (int i = 0; i < 1000; i++) {
			double[] v = new double[4];
			v[0] = i * 0.1;
			v[1] = i % f2.getCardinality();
			v[2] = (i + 1000) % f3.getNumBins();
			v[3] = i % f3.getNumBins();
			double target = (i % 10) < 8 ? 0 : 1;
			Instance instance = new Instance(v, target);
			denseClaDataset.add(instance);
		}
		
		denseRegDataset = denseClaDataset.copy();
		denseRegDataset.setTargetAttribute(regTarget);
	}
	
}
