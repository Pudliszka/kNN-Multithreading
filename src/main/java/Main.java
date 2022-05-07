import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.*;
import weka.core.converters.ConverterUtils;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;
import java.util.concurrent.TimeUnit;

public class Main {

    public static void main(String[] args) throws Exception {
        ManhattanDistance manDistance = new ManhattanDistance();
        EuclideanDistance euDistance = new EuclideanDistance();
        ChebyshevDistance cheDistance = new ChebyshevDistance();
        int basicValue = 16;
        for (int threadDevide = 1; threadDevide <= basicValue; threadDevide *= 2) {
            List<Thread> threads = new ArrayList<>();
            var startDate = new Date();
            for (int threadNumber = 1; threadNumber <= threadDevide; threadNumber++) {
                int finalThreadDevide = threadDevide;
                int finalThreadNumber = threadNumber;
                Thread thread = new Thread(() -> {
                    try {
                        for (int kValue = ((basicValue / finalThreadDevide) * (finalThreadNumber - 1)) + 1; kValue <= ((basicValue / finalThreadDevide) * finalThreadNumber); kValue++) {
                            ConverterUtils.DataSource source = new ConverterUtils.DataSource("data/Dry_Bean_Dataset.arff");
                            Instances data = source.getDataSet();
                            if (data.classIndex() == -1) {
                                data.setClassIndex(data.numAttributes() - 1);
                            }
                            IBk classifier = new IBk();
                            classifier.buildClassifier(data);
                            String[] options = Utils.splitOptions("-K " + kValue + " -W 0");
                            classifier.setOptions(options);
                            classifier.setCrossValidate(false);
                            classifier.setBatchSize("100");
                            classifier.setDebug(false);
                            classifier.setDoNotCheckCapabilities(false);
                            classifier.setMeanSquared(false);
                            classifier.getNearestNeighbourSearchAlgorithm().setDistanceFunction(manDistance);
                            classifier.setNumDecimalPlaces(2);
                            classifier.setWindowSize(0);
                            classifier.buildClassifier(data);
                            Evaluation eval = new Evaluation(data);
                            eval.crossValidateModel(classifier, data, 10, new Random(1));
                        }
                    } catch (Exception e) {
                        System.err.println(e);
                    }
                });
                threads.add(thread);
                thread.start();
            }
            for (Thread thread : threads) {
                try {
                    thread.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            var endDate = new Date();
            var miliSecond = Math.abs(endDate.getTime() - startDate.getTime());
            System.out.println("Liczba threadów: " + threadDevide);
            System.out.println("Czas: " + miliSecond/1000 +"s");
        }
    }
}
