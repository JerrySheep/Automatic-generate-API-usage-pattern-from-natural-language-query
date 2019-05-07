import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by yanhao on 2019/4/3.
 */
public class main {
    public static void main(String[] args) throws SQLException, ClassNotFoundException {
//        System.out.println("ok");

        //String path = args[0]; //要遍历的路径
        System.out.println("The java file is stored in : " + args[0]);
        System.out.println("The sequence data is stored in : " + args[1]);
        System.out.println("The usage data is stored in : " + args[2]);
        System.out.println("The data info is stored in : " + args[3]);

        String path = args[0];

        //String path = "/Users/yanhao/Desktop/Github_Projects";
        File file = new File(path);		//获取其file对象
        final List<String> api_sequence = new ArrayList();
        final List<String> api_usage = new ArrayList();

        String sequencePath = args[1];
        String usagePath = args[2];
        String data_path = args[3];

        File[] fs = file.listFiles();

        func(fs, api_sequence, api_usage, sequencePath, usagePath, data_path);

//        int dataSize = api_sequence.size();
//
//        for(int i = 0; i < dataSize; ++i){
//            System.out.println(api_sequence.get(i));
//            System.out.println(api_usage.get(i));
//            System.out.println("\n");
//        }


//        try {
//            BufferedWriter sequence = new BufferedWriter(new FileWriter(args[1]));
//            //BufferedWriter sequence = new BufferedWriter(new FileWriter("/Users/yanhao/Desktop/api_sequence.txt"));
//            for (String s : api_sequence) {
//                sequence.write(s);
//                sequence.newLine();
//                sequence.flush();
//            }
//            sequence.close();
//
//            BufferedWriter usage = new BufferedWriter(new FileWriter(args[2]));
//            //BufferedWriter usage = new BufferedWriter(new FileWriter("/Users/yanhao/Desktop/api_usage.txt"));
//
//
//            for (String s : api_usage) {
//                usage.write(s);
//                usage.newLine();
//                usage.flush();
//            }
//            usage.close();
//
//        } catch (IOException e) {
//            // TODO Auto-generated catch block
//            e.printStackTrace();
//        }
        try {
            BufferedWriter sequence = new BufferedWriter(new FileWriter(sequencePath + "_final.csv", true));
            //BufferedWriter sequence = new BufferedWriter(new FileWriter("/Users/yanhao/Desktop/api_sequence.txt", true));
            for (String s : api_sequence) {
                sequence.write(s);
                sequence.newLine();
                sequence.flush();
            }
            sequence.close();

            BufferedWriter usage = new BufferedWriter(new FileWriter(usagePath + "_final.csv", true));
            //BufferedWriter usage = new BufferedWriter(new FileWriter("/Users/yanhao/Desktop/api_usage.txt", true));

            for (String s : api_usage) {
                usage.write(s);
                usage.newLine();
                usage.flush();
            }
            usage.close();

            System.out.println("fine, finish");

        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        System.out.println(DemoVisitorTest.javaFileNumber + " java files!");

    }

    private static void func(File[] file, final List<String> api_sequence, final List<String> api_usage, String sequencePath, String usagePath, String data_path) throws SQLException, ClassNotFoundException {
//        File[] fs = file.listFiles();
        final List<String> data_info = new ArrayList();

        System.out.println(file.length);
        int i;
        int j;
        File[] f;
        for(i = 0; i < file.length; ++i){
            if(file[i].isDirectory()){
                f = file[i].listFiles();
                for(j = 0; j < f.length; ++j){
                    if(f[j].isFile()){
                        String filename = f[j].getName();
                        if(filename.endsWith(".java")){
//                    System.out.println("parser");
                            ++DemoVisitorTest.javaFileNumber;
                            String filepath = f[j].toString();

                            DemoVisitorTest test = new DemoVisitorTest(filepath, api_sequence, api_usage);
                            test = null;

                            System.out.println("The " + DemoVisitorTest.javaFileNumber + " download(s)");
                            System.out.println(filepath);
                            if(DemoVisitorTest.javaFileNumber % 10000 == 0 && api_sequence.size() == api_usage.size()){
                                try {
                                    BufferedWriter sequence = new BufferedWriter(new FileWriter(sequencePath + "_" + DemoVisitorTest.javaFileNumber + ".csv", true));
                                    //BufferedWriter sequence = new BufferedWriter(new FileWriter("/Users/yanhao/Desktop/api_sequence.txt", true));
                                    for (String s : api_sequence) {
                                        sequence.write(s);
                                        sequence.newLine();
                                        sequence.flush();
                                    }
                                    sequence.close();

                                    BufferedWriter usage = new BufferedWriter(new FileWriter(usagePath + "_" + DemoVisitorTest.javaFileNumber + ".csv", true));
                                    //BufferedWriter usage = new BufferedWriter(new FileWriter("/Users/yanhao/Desktop/api_usage.txt", true));

                                    for (String s : api_usage) {
                                        usage.write(s);
                                        usage.newLine();
                                        usage.flush();
                                    }
                                    usage.close();

                                    data_info.add("The " + DemoVisitorTest.javaFileNumber + " api_sequence size is " + api_sequence.size());
                                    data_info.add("The " + DemoVisitorTest.javaFileNumber + " api_usage size is " + api_usage.size());

                                    api_sequence.clear();
                                    api_usage.clear();

                                    data_info.add("After clean, The size is " + api_sequence.size() + " and " + api_usage.size());

                                    data_info.add("Fine, store the " + DemoVisitorTest.javaFileNumber + " data\n");

                                    BufferedWriter data_info_show = new BufferedWriter(new FileWriter(data_path, true));
                                    for (String s : data_info) {
                                        data_info_show.write(s);
                                        data_info_show.newLine();
                                        data_info_show.flush();
                                    }
                                    data_info_show.close();

                                    System.gc();

                                } catch (IOException e) {
                                    // TODO Auto-generated catch block
                                    e.printStackTrace();
                                }
                            }

                        }
                    }
                }
            }
        }
//        for(File fileInfo : file){
////            if(f.isDirectory())	//若是目录，则递归打印该目录下的文件
////                func(f, api_sequence, api_usage, sequencePath, usagePath);
//            if(fileInfo.isDirectory()){
//                File[] fs = fileInfo.listFiles();
//                for(File f : fs){
//                    if(f.isFile())	{
//                        String filename = f.getName();
//                        if(filename.endsWith(".java")){
////                    System.out.println("parser");
//                            ++DemoVisitorTest.javaFileNumber;
//                            String filepath = f.toString();
//                            DemoVisitorTest test = new DemoVisitorTest(filepath, api_sequence, api_usage);
//                            test = null;
//
//                            System.out.println("The " + DemoVisitorTest.javaFileNumber + " download(s)");
//                            if(DemoVisitorTest.javaFileNumber % 10000 == 0){
//                                try {
//                                    BufferedWriter sequence = new BufferedWriter(new FileWriter(sequencePath, true));
//                                    //BufferedWriter sequence = new BufferedWriter(new FileWriter("/Users/yanhao/Desktop/api_sequence.txt", true));
//                                    for (String s : api_sequence) {
//                                        sequence.write(s);
//                                        sequence.newLine();
//                                        sequence.flush();
//                                    }
//                                    sequence.close();
//
//                                    BufferedWriter usage = new BufferedWriter(new FileWriter(usagePath, true));
//                                    //BufferedWriter usage = new BufferedWriter(new FileWriter("/Users/yanhao/Desktop/api_usage.txt", true));
//
//                                    for (String s : api_usage) {
//                                        usage.write(s);
//                                        usage.newLine();
//                                        usage.flush();
//                                    }
//                                    usage.close();
//
//                                    System.out.println(api_sequence.size());
//                                    api_sequence.clear();
//                                    api_usage.clear();
//
//                                    System.out.println("fine, store some data");
//                                    System.out.println(api_sequence.size());
//
//                                    System.gc();
//
//                                } catch (IOException e) {
//                                    // TODO Auto-generated catch block
//                                    e.printStackTrace();
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//
//        }
    }
}
