package weka.api;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.Logistic;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Classification {
	
	private String filename;
	private String filenameTest;
	private DataSource source;
	private DataSource sourceTest;
	private Instances dataset;
	private Instances datasetTest;
	private Evaluation eval;
	private int folds = 10;
	private Random rand;
	
	private boolean isTestSet = false;
	
	public Classification (String filename) throws Exception {
		this.filename = filename + ".arff";
		this.filenameTest = filename + "-test.arff";
		this.source = new DataSource(this.filename);
		this.dataset = this.source.getDataSet();
		this.dataset.setClassIndex(this.dataset.numAttributes()-1);
		this.eval = new Evaluation(this.dataset);
		this.rand = new Random(1);
	}
	
	public boolean isTestSet() {
		return isTestSet;
	}

	public void setTestSet(boolean isTestSet) throws Exception {
		this.isTestSet = isTestSet;
		if (isTestSet == true) {
			sourceTest = new DataSource(filenameTest);
			datasetTest = sourceTest.getDataSet();
			datasetTest.setClassIndex(datasetTest.numAttributes()-1);
		}
	}

	public void j48 () throws Exception {
		J48 tree = new J48();
		
		if (isTestSet) {
			tree.buildClassifier(dataset);
			eval.evaluateModel(tree, datasetTest);
		} else {
			tree.buildClassifier(dataset);
			eval.crossValidateModel(tree, dataset, folds, rand);			
		}
		
		printEval();
	}
	
	public void logisticRegression () throws Exception {		
		Logistic logistic = new Logistic();
		
		if (isTestSet) {
			logistic.buildClassifier(dataset);
			eval.evaluateModel(logistic, datasetTest);
		} else {
			logistic.buildClassifier(dataset);
			eval.crossValidateModel(logistic, dataset, folds, rand);			
		}

		printEval();
	}
	
	public void naiveBayes () throws Exception {
		NaiveBayes nb = new NaiveBayes();
		
		if (isTestSet) {
			nb.buildClassifier(dataset);
			eval.evaluateModel(nb, datasetTest);
		} else {
			nb.buildClassifier(dataset);
			eval.crossValidateModel(nb, dataset, folds, rand);			
		}
		
		printEval();
	}
	
	public void svm () throws Exception {
		SMO svm = new SMO();
		
		if (isTestSet) {
			svm.buildClassifier(dataset);
			eval.evaluateModel(svm, datasetTest);
		} else {
			svm.buildClassifier(dataset);
			eval.crossValidateModel(svm, dataset, folds, rand);			
		}
		
		printEval();
	}
	
	public void randomForest () throws Exception {
		RandomForest rf = new RandomForest();
		
		if (isTestSet) {
			rf.buildClassifier(dataset);
			eval.evaluateModel(rf, datasetTest);
		} else {
			rf.buildClassifier(dataset);
			eval.crossValidateModel(rf, dataset, folds, rand);			
		}
		
		printEval();
	}
	
	public void printEval () throws Exception {
		System.out.println(eval.toSummaryString("Evaluation results:\n", false));
		System.out.println(eval.toClassDetailsString());
	}

}
