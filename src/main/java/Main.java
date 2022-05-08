import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.*;
import weka.core.converters.ConverterUtils;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public class Main {

    //zapuścić 5 razy zapisać w tabeli w wordzie wyliczyc średnia i na wykresie pokazać średnią
    public static void main(String[] args) throws Exception {
        int basicValue = 16;
        for (int threadDevide = 1; threadDevide <= basicValue; threadDevide *= 2) {
            List<Thread> threads = new ArrayList<>();
            List<Integer> bestsKList = new ArrayList<>();
            List<Double> procentTrueList = new ArrayList<>();

            var startDate = new Date();
            for (int threadNumber = 1; threadNumber <= threadDevide; threadNumber++) {
                int finalThreadDevide = threadDevide;
                int finalThreadNumber = threadNumber;
                Thread thread = new Thread(() -> {
                    int bestK = 0;
                    double bestProcent = 0;
                    try {
                        for (int kValue = ((basicValue / finalThreadDevide) * (finalThreadNumber - 1)) + 1; kValue <= ((basicValue / finalThreadDevide) * finalThreadNumber); kValue++) {
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
                            if(eval.weightedTruePositiveRate() >= bestProcent) {
                                bestK = kValue;
                                bestProcent = eval.weightedTruePositiveRate();
                                System.out.println("New Best K = " + bestK);
                            }
                        }
                    } catch (Exception e) {
                        System.err.println(e);
                    }
                    bestsKList.add(bestK);
                    procentTrueList.add(bestProcent);
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
            int theBestKForTheBestK = 0;
            double theBestProcentForTheBestProcent = 0.0;
            for (int i = 0; i < procentTrueList.size(); i++) {
                if(theBestProcentForTheBestProcent < procentTrueList.get(i)) {
                    theBestKForTheBestK = bestsKList.get(i);
                    theBestProcentForTheBestProcent = procentTrueList.get(i);
                }
            }
            var endDate = new Date();
            var miliSecond = Math.abs(endDate.getTime() - startDate.getTime());
            System.out.println("How many thread: " + threadDevide);
            System.out.println("The best K = " + theBestKForTheBestK);
            System.out.println("Time: " + miliSecond/1000 +"s");
        }

    }
}
