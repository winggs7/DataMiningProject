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
	private String className;
	private DataSource source;
	private Instances dataset;
	private Evaluation eval;
	private int folds = 10;
	private Random rand;

	public Classification(String filename, String className) throws Exception {
		this.filename = filename + ".arff";
		this.className = className;
		this.source = new DataSource(this.filename);
		this.dataset = this.source.getDataSet();
		this.dataset.setClass(this.dataset.attribute(this.className));
//		this.dataset.setClassIndex(dataset.numAttributes()-1);
		this.eval = new Evaluation(this.dataset);
		this.rand = new Random(1);
	}

	public void printEval() throws Exception {
		System.out.println(eval.toSummaryString("Evaluation results:\n", false));
		System.out.println(eval.toClassDetailsString());
	}
	
	public void randomForest() throws Exception {
		RandomForest rf = new RandomForest();

		rf.buildClassifier(dataset);
		eval.crossValidateModel(rf, dataset, folds, rand);

		printEval();
	}

	public void logisticRegression() throws Exception {
		Logistic logistic = new Logistic();

		logistic.buildClassifier(dataset);
		eval.crossValidateModel(logistic, dataset, folds, rand);

		printEval();
	}
	
	public void j48() throws Exception {
		J48 tree = new J48();

		tree.buildClassifier(dataset);
		eval.crossValidateModel(tree, dataset, folds, rand);

		printEval();
	}

	public void naiveBayes() throws Exception {
		NaiveBayes nb = new NaiveBayes();

		nb.buildClassifier(dataset);
		eval.crossValidateModel(nb, dataset, folds, rand);

		printEval();
	}

	public void svm() throws Exception {
		SMO svm = new SMO();

		svm.buildClassifier(dataset);
		eval.crossValidateModel(svm, dataset, folds, rand);

		printEval();
	}

}
