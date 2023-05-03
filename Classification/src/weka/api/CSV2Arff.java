package weka.api;

import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class CSV2Arff {

	public static void main(String[] args) throws Exception{
		// TODO Auto-generated method stub
		String filename = "Test_set";
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File(filename + ".csv"));
		String[] nominalVals = {"Sex,Category:nominal,nominal"};
		loader.setNominalLabelSpecs(nominalVals);
		
		Instances data = loader.getDataSet();
		data.setClass(data.attribute("Category"));
		
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		
		saver.setFile(new File(filename + ".arff"));
		saver.writeBatch();
	}

}
