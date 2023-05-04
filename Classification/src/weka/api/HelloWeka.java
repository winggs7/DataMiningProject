package weka.api;

public class HelloWeka {
	public static void main(String[] args) throws Exception {		
		String filename = "4_Cate";
		String className = "Category";
		Classification c = new Classification(filename, className);
		
//		c.j48();
//		c.logisticRegression();
//		c.naiveBayes();
//		c.svm();
		c.randomForest();
	}

}