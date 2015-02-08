package mltk.core.processor;

import java.util.ArrayList;
import java.util.List;

import mltk.core.Attribute;
import mltk.core.BinnedAttribute;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.NominalAttribute;
import mltk.core.NumericalAttribute;

/**
 * Class for one-hot encoders. Binned attributes and nominal attributes are
 * transformed into a set of binary attributes using one-hot encoding. 
 * 
 * @author Yin Lou
 *
 */
public class OneHotEncoder {

	/**
	 * Transforms all binned and nominal attributes to binary attributes using
	 * one-hot encoding.
	 * 
	 * @param instances the input instances.
	 * @return the transformed instances.
	 */
	public Instances process(Instances instances) {
		List<Attribute> attrListOld = instances.getAttributes();
		List<Attribute> attrListNew = new ArrayList<>();
		int[] offset = new int[instances.dimension()];
		boolean[] isNumerical = new boolean[instances.dimension()];
		int attIndex = 0;
		for (int j = 0; j < attrListOld.size(); j++) {
			Attribute attribute = attrListOld.get(j);
			offset[j] = attIndex;
			String name = attribute.getName();
			if (attribute instanceof BinnedAttribute) {
				BinnedAttribute binnedAttribute = (BinnedAttribute) attribute;
				int size = binnedAttribute.getNumBins();
				for (int k = 0; k < size; k++) {
					NumericalAttribute attr = new NumericalAttribute(name + "_" + k);
					attr.setIndex(attIndex++);
					attrListNew.add(attr);
				}
			} else if (attribute instanceof NominalAttribute) {
				NominalAttribute nominalAttribute = (NominalAttribute) attribute;
				String[] states = nominalAttribute.getStates();
				for (String state : states) {
					NumericalAttribute attr = new NumericalAttribute(name + "_" + state);
					attr.setIndex(attIndex++);
					attrListNew.add(attr);
				}
			} else {
				NumericalAttribute attr = new NumericalAttribute(name);
				attr.setIndex(attIndex++);
				attrListNew.add(attr);
				isNumerical[j] = true;
			}
		}
		
		Instances instancesNew = new Instances(attrListNew, instances.getTargetAttribute(), 
				instances.size());
		for (Instance instance : instances) {
			int[] indices = new int[instances.dimension()];
			double[] values = new double[instances.dimension()];
			for (int j = 0; j < attrListOld.size(); j++) {
				if (isNumerical[j]) {
					indices[j] = offset[j];
					values[j] = instance.getValue(attrListOld.get(j));
				} else {
					int v = (int) instance.getValue(attrListOld.get(j));
					indices[j] = offset[j] + v;
					values[j] = 1.0;
				}
			}
			Instance instanceNew = new Instance(indices, values, instance.getTarget(), 
					instance.getWeight());
			instancesNew.add(instanceNew);	
		}
		
		return instancesNew;
	}
	
}
