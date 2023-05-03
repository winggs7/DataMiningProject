package weka.api;

public class HelloWeka {
	public static void main(String[] args) throws Exception {		
		String filename = "segment";
		Classification c = new Classification(filename);
		
		c.setTestSet(true);
		
//		c.j48();
//		c.logisticRegression();
//		c.naiveBayes();
//		c.svm();
		c.randomForest();
	}

}