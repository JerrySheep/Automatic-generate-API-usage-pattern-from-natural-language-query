import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.actionSystem.PlatformDataKeys;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.ui.Messages;
import org.eclipse.jdt.internal.compiler.ast.MessageSend;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.security.PublicKey;

import static com.yourkit.util.Util.sleep;

/**
 * Created by yanhao on 2019/4/26.
 */
public class Main extends AnAction {
    public static String ss;

    @Override
    public void actionPerformed(AnActionEvent event) {
//        Project project = event.getData(PlatformDataKeys.PROJECT);
        ss = Messages.showInputDialog(
//                project,
                "Input your natural language query:",
                "Natural Language Query",
                Messages.getQuestionIcon());

//        testDemo testDemo = new testDemo();
//        testDemo.start();
//
//        try{
//            testDemo.join();
//        }catch (InterruptedException e){
//            e.printStackTrace();
//        }
        ss = transform();

        Messages.showInfoMessage(
                "The API usage pattern result is:\n" + ss,
                "API Usage Pattern");


    }

    public static String transform(){
        Runtime rt = Runtime.getRuntime(); //Runtime.getRuntime()返回当前应用程序的Runtime对象
        Process ps = null;  //Process可以控制该子进程的执行或获取该子进程的信息。
        try {
            String[] args1=new String[]{"/usr/local/bin/python3 (Your python3 location)","the pythonTool.py file location", ss};
            ps = rt.exec(args1);   //该对象的exec()方法指示Java虚拟机创建一个子进程执行指定的可执行程序，并返回与该子进程对应的Process对象实例。
            BufferedReader in = new BufferedReader(new InputStreamReader(
                    ps.getInputStream()));
            String line;
            boolean judge = true;
            while ((line = in.readLine()) != null) {
                if(judge){
                    ss = line;
                    judge = false;
                }
            }
            in.close();
            ps.waitFor();
            ps.destroy();
//            System.out.println(Main.ss);

            return ss;
        } catch (IOException e1) {
            e1.printStackTrace();
        } catch (InterruptedException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return ss;
    }
}
