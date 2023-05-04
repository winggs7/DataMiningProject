package weka.api;

import java.io.File;
import java.text.DecimalFormat;
import java.time.Duration;
import java.time.LocalTime;
import java.util.Random;

import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.supervised.instance.SpreadSubsample;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.Logistic;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

public class Classification {

	private String filename;
	private DataSource source;
	private Instances dataset;
	private Evaluation eval;
	private int folds = 10;
	private int seeds = 5;
	private Random rand;
	
	private double sumAcc = 0;
	private double recallSum1 = 0;
	private double recallSum2 = 0;
	private double recallSum3 = 0;
	private double recallSum4 = 0;
	private long timeSum = 0;
	
	private boolean flag = false;
	private boolean flagCross = false;

	public Classification(String filename, String className) throws Exception {
		this.filename = filename + ".arff";
		this.source = new DataSource(this.filename);
		this.dataset = this.source.getDataSet();
		this.dataset.setClass(dataset.attribute(className));
//		this.dataset.setClassIndex(dataset.numAttributes()-1);
		this.eval = new Evaluation(this.dataset);
		this.rand = new Random(seeds);
	}

	public void printEval() throws Exception {
		System.out.println(eval.toSummaryString("Evaluation results:\n", false));
		System.out.println(eval.toClassDetailsString());
	}

	public void randomForest() throws Exception {
		this.flag = false;
		if (flagCross) {
			crossValuation();
		} else {
			classificationData();
		}
	}

	public void logisticRegression() throws Exception {
		this.flag = true;
		if (flagCross) {
			crossValuation();
		} else {
			classificationData();
		}
	}

	public void isCrossValidation(boolean flagCross) {
		this.flagCross = flagCross;
	}

	public void classificationData() throws Exception {
		Logistic logistic = new Logistic();
		RandomForest rf = new RandomForest();

		Instances randData = new Instances(dataset);
		randData.randomize(rand);

		randData.stratify(folds);

		SMOTE smote = new SMOTE();
		SpreadSubsample sss = new SpreadSubsample();

		for (int i = 0; i < folds; i++) {

			Evaluation e = new Evaluation(randData);

			Instances train = randData.trainCV(folds, i);
			Instances test = randData.testCV(folds, i);

			smote.setPercentage(400);
			smote.setInputFormat(train);

			Instances trainReSample = new Instances(train);

			for (int j = 2; j <= 4; j++) {
				smote.setClassValue(String.valueOf(j));
				trainReSample = new Instances(Filter.useFilter(trainReSample, smote));
				smote.setInputFormat(trainReSample);
			}

			sss.setInputFormat(trainReSample);
			sss.setDistributionSpread(3);
			trainReSample = new Instances(Filter.useFilter(trainReSample, sss));

			ArffSaver saver = new ArffSaver();
			saver.setInstances(trainReSample);

			saver.setFile(new File(i + ".arff"));
			saver.writeBatch();

			long start = System.currentTimeMillis();
			if (flag) {
				logistic.buildClassifier(trainReSample);
				e.evaluateModel(logistic, test);
			} else {
				rf.buildClassifier(trainReSample);
				e.evaluateModel(rf, test);
			}
			long end = System.currentTimeMillis();

			sumAcc += e.pctCorrect();
			recallSum1 += e.recall(0);
			recallSum2 += e.recall(1);
			recallSum3 += e.recall(2);
			recallSum4 += e.recall(3);
			timeSum += (end - start);

			System.out.println(e.toSummaryString("Evaluation results:\n", false));
			System.out.println(e.toClassDetailsString());
			System.out.println(e.toMatrixString());
		}

		System.out.println(
				"___________________________________________________________________________________________________");
		System.out.println(
				"--------------------------------------Final evaluation --------------------------------------------");

		DecimalFormat f = new DecimalFormat("##.00");

		System.out.println("\nCorrect avg: " + f.format(sumAcc / folds) + "%");
		System.out.println("\nRecall:");
		System.out.println("______ Class 0: " + f.format((recallSum1 / folds) * 100) + "%");
		System.out.println("______ Class 1: " + f.format((recallSum2 / folds) * 100) + "%");
		System.out.println("______ Class 2: " + f.format((recallSum3 / folds) * 100) + "%");
		System.out.println("______ Class 3: " + f.format((recallSum4 / folds) * 100) + "%");
		System.out.println("\nTime avg: " + timeSum + " milliseconds");

	}

	public void crossValuation() throws Exception {
		Logistic logistic = new Logistic();
		RandomForest rf = new RandomForest();

		long start = System.currentTimeMillis();
		if (flag) {
			logistic.buildClassifier(dataset);
			eval.crossValidateModel(logistic, dataset, folds, rand);
		} else {
			rf.buildClassifier(dataset);
			eval.crossValidateModel(rf, dataset, folds, rand);
		}
		long end = System.currentTimeMillis();
		timeSum += (end - start);
		
		System.out.println(eval.toSummaryString("Evaluation results:\n", false));
		System.out.println(eval.toClassDetailsString());
		System.out.println(eval.toMatrixString());

		System.out.println(
				"___________________________________________________________________________________________________");
		System.out.println(
				"--------------------------------------Final evaluation --------------------------------------------");

		DecimalFormat f = new DecimalFormat("##.00");

		System.out.println("\nCorrect avg: " + f.format(eval.pctCorrect()) + "%");
		System.out.println("\nRecall:");
		System.out.println("______ Class 0: " + f.format(eval.recall(0) * 100) + "%");
		System.out.println("______ Class 1: " + f.format(eval.recall(1) * 100) + "%");
		System.out.println("______ Class 2: " + f.format(eval.recall(2) * 100) + "%");
		System.out.println("______ Class 3: " + f.format(eval.recall(3) * 100) + "%");
		System.out.println("\nTime avg: " + timeSum + " milliseconds");
	}

}
