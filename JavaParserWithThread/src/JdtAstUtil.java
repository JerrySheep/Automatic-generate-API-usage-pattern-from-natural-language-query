/**
 * Created by yanhao on 2019/3/26.
 */
import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

import org.eclipse.jdt.core.dom.AST;
import org.eclipse.jdt.core.dom.ASTParser;
import org.eclipse.jdt.core.dom.CompilationUnit;

public class JdtAstUtil {
    /**
     * get compilation unit of source code
     * @param javaFilePath
     * @return CompilationUnit
     */
    public static CompilationUnit getCompilationUnit(String javaFilePath){
        byte[] input = null;
        try {
            BufferedInputStream bufferedInputStream = new BufferedInputStream(new FileInputStream(javaFilePath));
            input = new byte[bufferedInputStream.available()];
            bufferedInputStream.read(input);
            bufferedInputStream.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }


        ASTParser astParser = ASTParser.newParser(AST.JLS4);
        astParser.setSource(new String(input).toCharArray());
        astParser.setKind(ASTParser.K_COMPILATION_UNIT);
        astParser.setEnvironment( // apply classpath
//                new String[] { "/usr/lib/jvm/java-8-oracle/jre/bin" }, //
                null,
                null, null, true);
        astParser.setResolveBindings(true);
        astParser.setBindingsRecovery(true);
        astParser.setUnitName("any_name");
        CompilationUnit result = (CompilationUnit) (astParser.createAST(null));

        return result;
    }
}
