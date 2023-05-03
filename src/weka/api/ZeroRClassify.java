package weka.api;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.rules.ZeroR;

public class ZeroRClassify {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		DataSource source = new DataSource("windata1.arff");
		Instances dataset = source.getDataSet();
		
		dataset.setClassIndex(dataset.numAttributes()-1);
		
		ZeroR zeroR = new ZeroR();
		zeroR.buildClassifier(dataset);

		System.out.println(zeroR);
		
		Evaluation eval = new Evaluation(dataset);
		eval.evaluateModel(zeroR, dataset);
		
		System.out.println(eval.toSummaryString("Evaluation results:\n", false));
		
		System.out.println("Correct % = " + eval.pctCorrect());
		System.out.println("Incorrect % = " + eval.pctIncorrect());
	}

}


