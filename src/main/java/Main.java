import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.*;
import weka.core.converters.ConverterUtils;

import java.util.*;

public class Main {
public static void main(String[] args) throws Exception, InterruptedException {
    int maxThreads = 16;
    for (int dividedProcess = 1; dividedProcess <= maxThreads; dividedProcess *= 2) {
        List<Thread> threads = new ArrayList<>();
        List<Integer> bestsKList = new ArrayList<>();
        List<Double> percentageTrueList = new ArrayList<>();

        var startDate = new Date();
        for (int threadNumber = 1; threadNumber <= dividedProcess; threadNumber++) {
            int finalDividedProcess = dividedProcess;
            int finalThreadNumber = threadNumber;
            Thread thread = new Thread(() -> {
                int bestKInThread = 0;
                double bestPercentageInThread = 0;
                try {
                    for (int kValue = ((maxThreads / finalDividedProcess) * (finalThreadNumber - 1)) + 1;
                         kValue <= ((maxThreads / finalDividedProcess) * finalThreadNumber);
                         kValue++) {
                        double positiveRateForEvolution = evolution(kValue);
                        if(positiveRateForEvolution >= bestPercentageInThread) {
                            bestKInThread = kValue;
                            bestPercentageInThread = positiveRateForEvolution;
                        }
                    }
                } catch (Exception error) {
                    System.err.println(error.getMessage());
                }
                bestsKList.add(bestKInThread);
                percentageTrueList.add(bestPercentageInThread);
            });
            threads.add(thread);
            thread.start();
        }
        for (Thread thread : threads) {
            thread.join();
        }
        int theBestKForTheBestK = 0;
        double theBestPercentageForTheBestPercentage = 0.0;
        for (int i = 0; i < percentageTrueList.size(); i++) {
            if(theBestPercentageForTheBestPercentage < percentageTrueList.get(i)) {
                theBestKForTheBestK = bestsKList.get(i);
                theBestPercentageForTheBestPercentage = percentageTrueList.get(i);
            }
        }
        var endDate = new Date();
        var milliSecond = Math.abs(endDate.getTime() - startDate.getTime());
        System.out.println("How many thread: " + dividedProcess);
        System.out.println("The best K = " + theBestKForTheBestK);
        System.out.println("Time: " + milliSecond/1000 +"s");
    }
}
    public static double evolution(int kValue) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("data/Dry_Bean_Dataset.arff");
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        IBk classifier = new IBk();
        String[] options = Utils.splitOptions("-K " + kValue + " -W 0");
        classifier.setOptions(options);
        classifier.buildClassifier(data);
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(classifier, data, 10, new Random(1));
        return eval.weightedTruePositiveRate();
    }
}
