import sun.awt.windows.ThemeReader;

import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;

/**
 * Created by yanhao on 2019/3/29.
 */
public class main {
    public static void main(String[] args){
        String github_csv;
        System.out.println("Please enter the csv file which you would like to operate:");
        //Scanner scanner = new Scanner(System.in);
        //github_csv = scanner.nextLine();
        github_csv = args[0];
        System.out.println("Your github info path is : " + github_csv);
        //"/Users/yanhao/Desktop/github_info.csv"
        File csv = new File(github_csv);
        if(!csv.exists()){
            return;
        }
        csv.setReadable(true);//设置可读
        csv.setWritable(true);//设置可写

        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(csv));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        String line = "";
        String everyLine = "";
        ArrayList<String> allString = new ArrayList<>();
        try {
            while ((line = br.readLine()) != null) // 读取到的内容给line变量
            {
                everyLine = line;
                //System.out.println(everyLine);
                allString.add(everyLine);
            }
            System.out.println("csv表格中所有行数：" + allString.size());
        } catch (IOException e) {
            e.printStackTrace();
        }

        String localPath;
        System.out.println("Please enter the path of the file which you would like to store the java file of the github project:");
        //localPath = scanner.nextLine();
        localPath = args[1];
        System.out.println("Your github info store path is : " + localPath);
        //localPath = "/data/phantom/github_info/Github_Projects";
        //url = "git@github.com:"+user+"/"+name;

//        for(int i = 0; i < allString.size(); i += 10){
//            try{
//                String[] urls = new String[10];
//                for(int j = 0; j < 10; ++j){
//                    urls[j] = allString.get(i + j);
//                }
//
//                Thread downloaderThread = null;
//                for (String url : urls) {
//                    // 创建文件下载器线程
//                    downloaderThread = new Thread(new GitClone(localPath, url, i));
//                    // 启动文件下载器线程
//                    downloaderThread.start();
//                }
//            }
//            catch (Exception e){
//                e.printStackTrace();
//            }
//        }

        int projectNumber = 31107;

        int coreNumber = Runtime.getRuntime().availableProcessors();
        System.out.println("core number: " + coreNumber);

        int threadsNumber = coreNumber * 2 + 2;

        while(true){
            if(GitClone.count.get() < threadsNumber){
            //if(GitClone.GetCount() < 10){
                Thread downloadThread = new Thread(new GitClone(localPath, allString.get(projectNumber++), projectNumber));
                downloadThread.start();
            }

            if(projectNumber == allString.size()){
                System.out.println("finish");
                break;
            }

        }
    }
}
