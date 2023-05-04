package weka.api;

public class HelloWeka {
	public static void main(String[] args) throws Exception {		
		String filename = "HepatitisC_Final_Ver";
		String className = "Category";
		Classification c = new Classification(filename, className);

		c.isCrossValidation(true);
		
		c.logisticRegression();
//		c.randomForest();
	}

}