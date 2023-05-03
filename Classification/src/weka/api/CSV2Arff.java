package weka.api;

import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class CSV2Arff {

	public static void main(String[] args) throws Exception{
		// TODO Auto-generated method stub
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File("4_Cate_SMOTE.csv"));
		Instances data = loader.getDataSet();
		
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		
		saver.setFile(new File("4_Cate_SMOTE.arff"));
		saver.writeBatch();
	}

}
