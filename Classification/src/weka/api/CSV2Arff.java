package weka.api;

import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class CSV2Arff {

	public static void main(String[] args) throws Exception{
		// TODO Auto-generated method stub
		String filename = "4_Cate";
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File(filename + ".csv"));
		String[] nominalVals = {"Sex:0,1", "Category:0,1,2,3"};
		loader.setNominalLabelSpecs(nominalVals);
		Instances data = loader.getDataSet();
		
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		
		saver.setFile(new File(filename + ".arff"));
		saver.writeBatch();
	}

}
