package weka.api;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import weka.core.Instances;
import weka.classifiers.Evaluation;

import weka.classifiers.functions.LinearRegression;
public class validationRegression {
    public static void main(String[] args) throws Exception {
        System.out.println("Model Evaluation:");

        BufferedReader breader = null;
        breader = new BufferedReader(new FileReader("windata1.arff"));
        Instances dataset = new Instances(breader);

        dataset.setClassIndex(dataset.numAttributes()-1);
        
        LinearRegression lr= new LinearRegression();
        
        int seed=1;
        int folds=10;
        
        Random rand = new Random(seed);
        
        Instances randData = new Instances(dataset);
        randData.randomize(rand);
        
        if(randData.classAttribute().isNominal())
            randData.stratify(folds);
        
        for(int n=0;n<folds;n++){
            Evaluation eval = new Evaluation(randData);
            Instances train = randData.trainCV(folds,n);
            Instances test = randData.testCV(folds, n);
            lr.buildClassifier(train);
            eval.evaluateModel(lr, test);
            
            //System.out.println(test);
            System.out.println("Folds "+(n+1));
           // System.out.println(test);
            System.out.println(lr);
            System.out.println(eval);
            System.out.println("Correct:    "+eval.meanAbsoluteError());
            System.out.println("Incorrect:  "+eval.pctIncorrect());
        }
    }
}
