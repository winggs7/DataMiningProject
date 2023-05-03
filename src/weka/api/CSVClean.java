package weka.api;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.List;

import au.com.bytecode.opencsv.CSVReader;
import au.com.bytecode.opencsv.CSVWriter;

public class CSVClean {

	public static void main(String[] args) throws Exception {
		System.out.println("Hi");
		File inputFile = new File("wind_dataset.csv");

        // Read existing file
        CSVReader reader = new CSVReader(new FileReader(inputFile), ',');
        List<String[]> csvBody = reader.readAll();
        // get CSV row column and replace with by using row and column
        for(int i=0; i<csvBody.size(); i++){
            String[] strArray = csvBody.get(i);
            for(int j=0; j<strArray.length; j++){
                if(strArray[j].equalsIgnoreCase("NA")){ //String to be replaced
                    csvBody.get(i)[j] = ""; //Target replacement
                }
            }
        }
        reader.close();

        // Write to CSV file which is open
        CSVWriter writer = new CSVWriter(new FileWriter(inputFile), ',');
        writer.writeAll(csvBody);
        writer.flush();
        writer.close();
        System.out.println("Hello");
	}
	
}
