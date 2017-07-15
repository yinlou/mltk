package mltk.predictor.tree;

public class RegressionTreeTestHelper {
	
	private static RegressionTreeTestHelper instance = null;
	
	private RegressionTree tree1;
	private RegressionTree tree2;
	
	public static RegressionTreeTestHelper getInstance() {
		if (instance == null) {
			instance = new RegressionTreeTestHelper();
		}
		return instance;
	}
	
	public RegressionTree getTree1() {
		return tree1;
	}
	
	public RegressionTree getTree2() {
		return tree2;
	}
	
	private RegressionTreeTestHelper() {
		buildTree1();
		buildTree2();
	}
	
	private void buildTree1() {
		TreeInteriorNode root = new TreeInteriorNode(1, 0.5);
		TreeNode leaf1 = new RegressionTreeLeaf(0.4);
		TreeInteriorNode right = new TreeInteriorNode(2, -1.5);
		TreeNode leaf2 = new RegressionTreeLeaf(0.5);
		TreeNode leaf3 = new RegressionTreeLeaf(0.6);
		right.left = leaf2;
		right.right = leaf3;
		root.left = leaf1;
		root.right = right;
		
		tree1 = new RegressionTree(root);
	}
	
	private void buildTree2() {
		TreeInteriorNode root = new TreeInteriorNode(5, 0);
		TreeNode leaf1 = new RegressionTreeLeaf(-0.4);
		TreeInteriorNode left = new TreeInteriorNode(0, -3.5);
		TreeNode leaf2 = new RegressionTreeLeaf(-0.5);
		TreeNode leaf3 = new RegressionTreeLeaf(-0.6);
		left.left = leaf1;
		left.right = leaf2;
		root.left = left;
		root.right = leaf3;
		
		tree2 = new RegressionTree(root);
	}

}
