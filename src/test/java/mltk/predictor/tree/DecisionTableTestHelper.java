package mltk.predictor.tree;

public class DecisionTableTestHelper {
	
	private static DecisionTableTestHelper instance = null;
	
	private DecisionTable dt1;
	private DecisionTable dt2;
	
	public static DecisionTableTestHelper getInstance() {
		if (instance == null) {
			instance = new DecisionTableTestHelper();
		}
		return instance;
	}
	
	public DecisionTable getTable1() {
		return dt1;
	}
	
	public DecisionTable getTable2() {
		return dt2;
	}
	
	private DecisionTableTestHelper() {
		buildDecisionTable1();
		buildDecisionTable2();
	}
	
	private void buildDecisionTable1() {
		int[] attIndices = new int[] {0, 1, 2};
		double[] splits = new double[] {50, 1.5, 23.5};
		long[] predIndices = new long[] {0, 1, 2, 3, 4, 5, 6, 7};
		double[] predValues = new double[] {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
		
		dt1 = new DecisionTable(attIndices, splits, predIndices, predValues);
	}
	
	private void buildDecisionTable2() {
		int[] attIndices = new int[] {3, 2, 0};
		double[] splits = new double[] {1, 56.5, 20};
		long[] predIndices = new long[] {0, 1, 2, 3, 4, 5, 6, 7};
		double[] predValues = new double[] {1.0, -0.9, 0.8, -0.7, 0.6, -0.5, 0.4, -0.3, 0.2};
		
		dt2 = new DecisionTable(attIndices, splits, predIndices, predValues);
	}

}
