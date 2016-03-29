package mltk.util.tuple;

import java.util.Comparator;

/**
 * Class for comparing &lt;int, double&gt; pairs. By default int is used as key, and in ascending order.
 * 
 * @author Yin Lou
 * 
 */
public class IntDoublePairComparator implements Comparator<IntDoublePair> {

	protected boolean ascending;
	protected boolean firstIsKey;

	public IntDoublePairComparator() {
		this(true, true);
	}

	public IntDoublePairComparator(boolean firstIsKey) {
		this(firstIsKey, true);
	}

	public IntDoublePairComparator(boolean firstIsKey, boolean ascending) {
		this.firstIsKey = firstIsKey;
		this.ascending = ascending;
	}

	@Override
	public int compare(IntDoublePair o1, IntDoublePair o2) {
		int cmp = 0;
		if (firstIsKey) {
			cmp = Integer.compare(o1.v1, o2.v1);
		} else {
			cmp = Double.compare(o1.v2, o2.v2);
		}
		if (!ascending) {
			cmp = -cmp;
		}
		return cmp;
	}

}
