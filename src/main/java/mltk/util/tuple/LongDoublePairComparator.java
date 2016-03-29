package mltk.util.tuple;

import java.util.Comparator;

/**
 * Class for comparing &lt;long, double&gt; pairs. By default long is used as key, and in ascending order.
 * 
 * @author Yin Lou
 * 
 */
public class LongDoublePairComparator implements Comparator<LongDoublePair> {

	protected boolean ascending;
	protected boolean firstIsKey;

	public LongDoublePairComparator() {
		this(true, true);
	}

	public LongDoublePairComparator(boolean firstIsKey) {
		this(firstIsKey, true);
	}

	public LongDoublePairComparator(boolean firstIsKey, boolean ascending) {
		this.firstIsKey = firstIsKey;
		this.ascending = ascending;
	}

	@Override
	public int compare(LongDoublePair o1, LongDoublePair o2) {
		int cmp = 0;
		if (firstIsKey) {
			cmp = Long.compare(o1.v1, o2.v1);
		} else {
			cmp = Double.compare(o1.v2, o2.v2);
		}
		if (!ascending) {
			cmp = -cmp;
		}
		return cmp;
	}

}
