import org.eclipse.jgit.api.CloneCommand;
import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.api.errors.InvalidRemoteException;
import org.eclipse.jgit.api.errors.JGitInternalException;
import org.eclipse.jgit.api.errors.TransportException;
import org.eclipse.jgit.transport.UsernamePasswordCredentialsProvider;

import java.io.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by yanhao on 2019/3/24.
 */
public class GitClone implements Runnable{
    private String localPath;
    private String url;
    private int number;
    public static AtomicInteger count = new AtomicInteger(0);
    //public static int count = 0;

    public  GitClone(String filepath, String fileUrl, int fileNumber){
        localPath = filepath;
        url = fileUrl;
        number = fileNumber;
    }

    public void func(File oldFile, File newFile) throws IOException{
        File[] fs = oldFile.listFiles();
        for(File f: fs){
            if(f.isDirectory())	//若是目录，则递归打印该目录下的文件
                func(f, newFile);
            if(f.isFile())	{
                String filename = f.getName();
                if(filename.endsWith(".java")){
                    FileInputStream in = new FileInputStream(f);
                    FileOutputStream out = new FileOutputStream(new File(newFile, filename));

                    byte[] buffer = new byte[2097152];
                    int readByte = 0;
                    while((readByte = in.read(buffer)) != -1){
                        out.write(buffer, 0, readByte);
                    }

                    in.close();
                    out.close();
                }
            }
        }
    }

    public int cloneRepository(String url, String localPath) {
        try {
            //System.out.println("开始下载");

            String[] urlInfo = url.split("/");
            url = "https://github.com/" + urlInfo[urlInfo.length - 2] + "/" + urlInfo[urlInfo.length - 1] + ".git";
            //System.out.println(url);

            CloneCommand cc = Git.cloneRepository().setURI(url);

            UsernamePasswordCredentialsProvider user = new UsernamePasswordCredentialsProvider("your github account", "your github password");
            cc.setCredentialsProvider(user);

            cc.setDirectory(new File(localPath)).call();

            //System.out.println("下载完成");

            deleteFile(new File(localPath + url));

            return 1;
        } catch (InvalidRemoteException e) {//这GitHub项目不存在
	  		//e.printStackTrace();
            System.out.println("$$$GitHub not exists:" + localPath);
            File f = new File(localPath);
            deleteFile(f);
            return 0;
        } catch (TransportException e) {//网络问题
            System.out.println("###network destroy:");
            e.printStackTrace();
            File f = new File(localPath);
            deleteFile(f);

            return 3;
        } catch (JGitInternalException e) {//该存储的目录已经存在
//	  		e.printStackTrace();
            System.out.println("&&&project exists");
            return 2;
        } catch (Exception e) {
            e.printStackTrace();
            return 4;
        }
    }

    public void deleteFile(File file) {
        if (file.exists()) {//判断文件是否存在
            if (file.isFile()) {//判断是否是文件
                file.delete();//删除文件
            } else if (file.isDirectory()) {//否则如果它是一个目录
                File[] files = file.listFiles();//声明目录下所有的文件 files[];
                for (int i = 0; i < files.length; i++) {//遍历目录下所有的文件
                    this.deleteFile(files[i]);//把每个文件用这个方法进行迭代
                }
                file.delete();//删除文件夹
            }
        }
    }
    public void storeAndDelete(){
        try{
            File newFile = new File(localPath + "_info/");
            if(!newFile.exists()){
                newFile.mkdirs();
            }

            File oldFile = new File(localPath);		//获取其file对象
            func(oldFile, newFile);

            deleteFile(oldFile);
        }
        catch (Exception e){
            e.printStackTrace();
        }

    }

//    public static synchronized int GetCount() {
//        return count;
//    }
//
//    public synchronized void IncrCount() {
//        count++;
//        return;
//    }
//
//    public synchronized void decrCount() {
//        count--;
//        return;
//    }


    @Override
    public void run(){
        count.incrementAndGet();

        //IncrCount();

        int isExist;

        System.out.println("The " + number + " download(s)!");

        localPath += "/" + number;
        //System.out.println(localPath);

        isExist = cloneRepository(url,localPath);

        if (isExist == 1){
            storeAndDelete();
        }

        //System.out.println("The number and the count: " + number + " " + count);
        count.decrementAndGet();
        //decrCount();
        //System.out.println("This is " + count);
        //System.out.println("other is " + GitClone.count);
    }
}
