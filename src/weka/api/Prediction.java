package weka.api;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.trees.M5P;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.LibSVMLoader;

public class Prediction {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		DataSource source = new DataSource("cpu.arff");
		Instances dataset = source.getDataSet();
		
		dataset.setClassIndex(dataset.numAttributes()-1);
		
		LinearRegression lr = new LinearRegression();
		lr.buildClassifier(dataset);
		
//		M5P m5p = new M5P();
//		m5p.buildClassifier(dataset);
		
		System.out.println(lr);
		
		Evaluation eval = new Evaluation(dataset);
		eval.evaluateModel(lr, dataset);
		
		System.out.println(eval.toSummaryString("Evaluation results:\n", false));
		
		System.out.println("Correct % = " + eval.pctCorrect());
		System.out.println("Incorrect % = " + eval.pctIncorrect());
		System.out.println(eval.errorRate());

	}

}
