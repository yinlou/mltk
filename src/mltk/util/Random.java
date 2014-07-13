package mltk.util;

/**
 * Class for global random object.
 * 
 * @author Yin Lou
 * 
 */
public class Random {

	protected static Random instance = null;
	protected java.util.Random rand;

	protected Random() {
		rand = new java.util.Random();
	}

	/**
	 * Returns the random object.
	 * 
	 * @return the singleton random object.
	 */
	public static Random getInstance() {
		if (instance == null) {
			instance = new Random();
		}
		return instance;
	}

	/**
	 * Sets the random seed.
	 * 
	 * @param seed the random seed.
	 */
	public void setSeed(long seed) {
		rand.setSeed(seed);
	}

	/**
	 * Returns the next pseudorandom, uniformly distributed <code>int</code> value from this random number generator's
	 * sequence.
	 * 
	 * @return a random integer.
	 */
	public int nextInt() {
		return rand.nextInt();
	}

	/**
	 * Returns the next pseudorandom, uniformly distributed <code>int</code> value between 0 (inclusive) and n
	 * (exclusive) from this random number generator's sequence.
	 * 
	 * @param n the range.
	 * @return a random integer in [0, n- 1].
	 */
	public int nextInt(int n) {
		return rand.nextInt(n);
	}

	/**
	 * Returns the next pseudorandom, uniformly distributed <code>double</code> value between 0.0 and 1.0 from this
	 * random number generator's sequence.
	 * 
	 * @return a random <code>double</code> value.
	 */
	public double nextDouble() {
		return rand.nextDouble();
	}

	/**
	 * Returns the next pseudorandom, uniformly distributed <code>float</code> value between 0.0 and 1.0 from this
	 * random number generator's sequence.
	 * 
	 * @return a random <code>float</code> value.
	 */
	public float nextFloat() {
		return rand.nextFloat();
	}

	/**
	 * Returns the next pseudorandom, Gaussian ("normally") distributed <code>
	 * double</code> value with mean 0.0 and standard deviation 1.0 from this random number generator's sequence.
	 * 
	 * @return a random <code>double</code> value.
	 */
	public double nextGaussian() {
		return rand.nextGaussian();
	}

	/**
	 * Returns the next pseudorandom, uniformly distributed <code>long</code> value from this random number generator's
	 * sequence.
	 * 
	 * @return a random <code>long</code> value.
	 */
	public long nextLong() {
		return rand.nextLong();
	}

	/**
	 * Returns the next pseudorandom, uniformly distributed <code>boolean</code> value from this random number
	 * generator's sequence.
	 * 
	 * @return a random <code>boolean</code> value.
	 */
	public boolean nextBoolean() {
		return rand.nextBoolean();
	}

	/**
	 * Generates random bytes and places them into a user-supplied byte array.
	 */
	public void nextBytes(byte[] bytes) {
		rand.nextBytes(bytes);
	}

	/**
	 * Returns the backend Java random object.
	 * 
	 * @return the backend Java random object.
	 */
	public java.util.Random getRandom() {
		return rand;
	}

}
