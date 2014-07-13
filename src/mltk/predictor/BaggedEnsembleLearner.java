package mltk.predictor;

import mltk.core.Instances;

/**
 * Class for learning bagged ensembles.
 * 
 * @author Yin Lou
 * 
 */
public class BaggedEnsembleLearner extends Learner {

	protected int baggingIters;
	protected Learner learner;
	protected Instances[] bags;

	/**
	 * Constructor.
	 * 
	 * @param baggingIters the number of bagging iterations.
	 * @param learner the learner.
	 */
	public BaggedEnsembleLearner(int baggingIters, Learner learner) {
		this.baggingIters = baggingIters;
		this.learner = learner;
	}

	/**
	 * Returns the number of bagging iterations.
	 * 
	 * @return the number of bagging iterations.
	 */
	public int getBaggingIterations() {
		return baggingIters;
	}

	/**
	 * Sets the number of bagging iterations.
	 * 
	 * @param baggingIters the number of bagging iterations.
	 */
	public void setBaggingIterations(int baggingIters) {
		this.baggingIters = baggingIters;
	}

	/**
	 * Returns the learner.
	 * 
	 * @return the learner.
	 */
	public Learner getLearner() {
		return learner;
	}

	/**
	 * Sets the learner.
	 * 
	 * @param learner the learner.
	 */
	public void setLearner(Learner learner) {
		this.learner = learner;
	}

	/**
	 * Returns the bootstrap samples.
	 * 
	 * @return the bootstrap samples.
	 */
	public Instances[] getBags() {
		return bags;
	}

	/**
	 * Sets the bootstrap samples.
	 * 
	 * @param bags the bootstrap samples.
	 */
	public void setBags(Instances[] bags) {
		this.bags = bags;
	}

	@Override
	public BaggedEnsemble build(Instances instances) {
		// Create bags
		bags = Bagging.createBags(instances, baggingIters);

		BaggedEnsemble baggedEnsemble = new BaggedEnsemble(bags.length);
		for (Instances bag : bags) {
			baggedEnsemble.add(learner.build(bag));
		}
		return baggedEnsemble;
	}

	/**
	 * Builds a bagged ensemble.
	 * 
	 * @param bags the bootstrap samples.
	 * @return a bagged ensemble.
	 */
	public BaggedEnsemble build(Instances[] bags) {
		BaggedEnsemble baggedEnsemble = new BaggedEnsemble(bags.length);
		for (Instances bag : bags) {
			baggedEnsemble.add(learner.build(bag));
		}
		return baggedEnsemble;
	}

}
