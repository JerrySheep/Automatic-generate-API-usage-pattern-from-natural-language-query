/**
 * Created by yanhao on 2019/3/26.
 */
import org.eclipse.jdt.core.dom.*;

import java.io.*;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class DemoVisitorTest {
    public static int javaFileNumber = 0;
    public static AtomicInteger count = new AtomicInteger(0);

    static String jdkPrefix[] = {"java.applet", "java.awt", "java.awt.color", "java.awt.datatransfer", "java.awt.dnd",
            "java.awt.event", "java.awt.font", "java.awt.geom", "java.awt.im", "java.awt.image", "java.awt.image.renderable", "java.awt.im.spi", "java.awt.print", "java.beans", "java.beans.beancontext", "java.io", "java.lang", "java.lang.annotation", "java.lang.instrument", "java.lang.invoke", "java.lang.management", "java.lang.ref", "java.lang.reflect", "java.math", "java.net", "java.nio", "java.nio.channels", "java.nio.channels.spi", "java.nio.charset", "java.nio.charset.spi", "java.nio.file", "java.nio.file.attribute", "java.nio.file.spi", "java.rmi", "java.rmi.activation", "java.rmi.dgc", "java.rmi.registry", "java.rmi.server", "java.security", "java.security.acl", "java.security.cert", "java.security.interfaces", "java.security.spec", "java.sql", "java.text", "java.text.spi", "java.util", "java.util.concurrent", "java.util.concurrent.atomic", "java.util.concurrent.locks", "java.util.jar", "java.util.logging", "java.util.prefs", "java.util.regex", "java.util.spi", "java.util.zip", "javax.accessibility", "javax.activation", "javax.activity", "javax.annotation", "javax.annotation.processing", "javax.crypto", "javax.crypto.interfaces", "javax.crypto.spec", "javax.imageio", "javax.imageio.event", "javax.imageio.metadata", "javax.imageio.plugins.bmp", "javax.imageio.plugins.jpeg", "javax.imageio.spi", "javax.imageio.stream", "javax.jws", "javax.jws.soap", "javax.lang.model", "javax.lang.model.element", "javax.lang.model.type", "javax.lang.model.util", "javax.management", "javax.management.loading", "javax.management.modelmbean", "javax.management.monitor", "javax.management.openmbean", "javax.management.relation", "javax.management.remote", "javax.management.remote.rmi", "javax.management.timer", "javax.naming", "javax.naming.directory", "javax.naming.event", "javax.naming.ldap", "javax.naming.spi", "javax.net", "javax.net.ssl", "javax.print", "javax.print.attribute", "javax.print.attribute.standard", "javax.print.event", "javax.rmi", "javax.rmi.CORBA", "javax.rmi.ssl", "javax.script", "javax.security.auth", "javax.security.auth.callback", "javax.security.auth.kerberos", "javax.security.auth.login", "javax.security.auth.spi", "javax.security.auth.x500", "javax.security.cert", "javax.security.sasl", "javax.sound.midi", "javax.sound.midi.spi", "javax.sound.sampled", "javax.sound.sampled.spi", "javax.sql", "javax.sql.rowset", "javax.sql.rowset.serial", "javax.sql.rowset.spi", "javax.swing", "javax.swing.border", "javax.swing.colorchooser", "javax.swing.event", "javax.swing.filechooser", "javax.swing.plaf", "javax.swing.plaf.basic", "javax.swing.plaf.metal", "javax.swing.plaf.multi", "javax.swing.plaf.nimbus", "javax.swing.plaf.synth", "javax.swing.table", "javax.swing.text", "javax.swing.text.html", "javax.swing.text.html.parser", "javax.swing.text.rtf", "javax.swing.tree", "javax.swing.undo", "javax.tools", "javax.transaction", "javax.transaction.xa", "javax.xml", "javax.xml.bind", "javax.xml.bind.annotation", "javax.xml.bind.annotation.adapters", "javax.xml.bind.attachment", "javax.xml.bind.helpers", "javax.xml.bind.util", "javax.xml.crypto", "javax.xml.crypto.dom", "javax.xml.crypto.dsig", "javax.xml.crypto.dsig.dom", "javax.xml.crypto.dsig.keyinfo", "javax.xml.crypto.dsig.spec", "javax.xml.datatype", "javax.xml.namespace", "javax.xml.parsers", "javax.xml.soap", "javax.xml.stream", "javax.xml.stream.events", "javax.xml.stream.util", "javax.xml.transform", "javax.xml.transform.dom", "javax.xml.transform.sax", "javax.xml.transform.stax", "javax.xml.transform.stream", "javax.xml.validation", "javax.xml.ws", "javax.xml.ws.handler", "javax.xml.ws.handler.soap", "javax.xml.ws.http", "javax.xml.ws.soap", "javax.xml.ws.spi", "javax.xml.ws.spi.http", "javax.xml.ws.wsaddressing", "javax.xml.xpath", "org.ietf.jgss", "org.omg.CORBA", "org.omg.CORBA_2_3", "org.omg.CORBA_2_3.portable", "org.omg.CORBA.DynAnyPackage", "org.omg.CORBA.ORBPackage", "org.omg.CORBA.portable", "org.omg.CORBA.TypeCodePackage", "org.omg.CosNaming", "org.omg.CosNaming.NamingContextExtPackage", "org.omg.CosNaming.NamingContextPackage", "org.omg.Dynamic", "org.omg.DynamicAny", "org.omg.DynamicAny.DynAnyFactoryPackage", "org.omg.DynamicAny.DynAnyPackage", "org.omg.IOP", "org.omg.IOP.CodecFactoryPackage", "org.omg.IOP.CodecPackage", "org.omg.Messaging", "org.omg.PortableInterceptor", "org.omg.PortableInterceptor.ORBInitInfoPackage", "org.omg.PortableServer", "org.omg.PortableServer.CurrentPackage", "org.omg.PortableServer.POAManagerPackage", "org.omg.PortableServer.POAPackage", "org.omg.PortableServer.portable", "org.omg.PortableServer.ServantLocatorPackage", "org.omg.SendingContext", "org.omg.stub.java.rmi", "org.w3c.dom", "org.w3c.dom.bootstrap", "org.w3c.dom.events", "org.w3c.dom.ls", "org.xml.sax", "org.xml.sax.ext", "org.xml.sax.helpers"};

    public static boolean isJdkApi(String s) {
        for(String si:jdkPrefix){
            if(s.startsWith(si))
                return true;
        }
        return false;
    }

    public DemoVisitorTest(final String path, final List api_sequence, final List api_usage) throws SQLException, StackOverflowError{
        CompilationUnit comp = null;
        try {
            comp = JdtAstUtil.getCompilationUnit(path);
        }
        catch (Exception e) {
            System.out.println("ERROR : Parse file >> " + path);
            //insertErrorStatus(con, path, projectId, disk, 4);
            e.printStackTrace();
            return;
        }

        comp.accept(new ASTVisitor() {
                        //查看抽象语法树AST下MethodDeclaration节点,用于方法提取
                        @Override
                        public boolean visit(MethodDeclaration node) {
                            //block为方法具体实现节点名
                            Block block = node.getBody();
                            final List ret = new ArrayList();
                            final List retFull = new ArrayList();
                            final String usage = "";
//                            final int[] cnt = {0};
                            final boolean debug = false;
//                String body_code = block.toString();
                            if(debug)
                                System.out.println(block.toString());
                            //getName()：return method name
                            ret.add(node.getName());
                            //javadoc是Sun公司提供的一个技术，它从程序源代码中抽取类、方法、成员等注释形成一个和源代码配套的API帮助文档。
                            //也就是说，只要在编写程序时以一套特定的标签作注释，在程序编写完成后，通过Javadoc就可以同时形成程序的开发文档了。
                            Javadoc docs = node.getJavadoc();
                            if(docs == null) {
                                return false;
                            }
                            List tags = docs.tags();
                            boolean haveJavadoc = false;
                            if(tags.size() > 0) {
                                String fullDocs = tags.get(0).toString().trim();
                                boolean isDocs = true;
                                for(int i = 0; i < fullDocs.length(); i++) {
                                    char ch = fullDocs.charAt(i);
                                    if(ch == '*' || ch == ' ') continue;
                                    if(ch == '@') {
                                        isDocs = false;
                                    }
                                    break;
                                }
                                if(isDocs) {
                                    retFull.add(fullDocs);
                                    if(debug) System.out.println(fullDocs);
                                    String[] sentences = fullDocs.split("\\. ");
                                    for(String sentence: sentences) {
                                        if(sentence.trim().length() > 0) {
                                            ret.add(sentence.trim());
                                            haveJavadoc = true;
                                            break;
                                        }
                                    }
                                    //                     类似于获取如下JavaDoc：
                                    //                    /**
                                    //                     * get compilation unit of source code
                                    //                     * @param javaFilePath
                                    //                     * @return CompilationUnit
                                    //                     */
                                }

                            }
                            if(!haveJavadoc) return true;
                            String temp_s = "";
                            ret.add(temp_s);
                            if(block != null){
                                block.accept(new ASTVisitor() {
                                    // IF statement
                                    @Override
                                    public boolean visit(IfStatement node) {
                                        if(debug)
                                            System.out.println("if(");
                                        ret.set(2, ret.get(2) +"if ( ");
                                        Expression expression = node.getExpression();
                                        if(expression != null) {
                                            expression.accept(this);
                                        }
                                        if(debug)
                                            System.out.println(")");
                                        ret.set(2, ret.get(2) +") ");
                                        Statement then_statement = node.getThenStatement();
                                        if(debug)
                                            System.out.println("{");
                                        ret.set(2, ret.get(2) +"{ ");
                                        if(then_statement != null) {
                                            then_statement.accept(this);
                                        }
                                        if(debug)
                                            System.out.println("}");
                                        ret.set(2, ret.get(2) +"} ");
                                        Statement else_statement = node.getElseStatement();
                                        if(else_statement != null) {
                                            if(debug)
                                                System.out.println("else {");
                                            ret.set(2, ret.get(2) +"else { ");
                                            else_statement.accept(this);
                                            if(debug)
                                                System.out.println("}");
                                            ret.set(2, ret.get(2) +"} ");
                                        }
                                        return false;
                                    }

                                    // While Statement
                                    @Override
                                    public boolean visit(WhileStatement node) {
                                        if(debug) System.out.println("while ( ");
                                        ret.set(2, ret.get(2) +"while ( ");
                                        Expression expression = node.getExpression();
                                        if(expression != null) {
                                            expression.accept(this);
                                        }
                                        if(debug) System.out.println(")");
                                        ret.set(2, ret.get(2) +") ");
                                        Statement body = node.getBody();
                                        if(debug) System.out.println("{");
                                        ret.set(2, ret.get(2) +"{ ");
                                        if(body != null) {
                                            body.accept(this);
                                        }
                                        if(debug) System.out.println("}");
                                        ret.set(2, ret.get(2) +"} ");
                                        return false;
                                    }

                                    // Do while Statement
                                    @Override
                                    public boolean visit(DoStatement node) {
                                        if(debug) System.out.println("do { ");
                                        ret.set(2, ret.get(2) + "do { ");
                                        Statement body = node.getBody();
                                        if(body != null) {
                                            body.accept(this);
                                        }
                                        if(debug) System.out.println("} while ( ");
                                        ret.set(2, ret.get(2) + "} while ( ");
                                        Expression expression = node.getExpression();
                                        if(expression != null) {
                                            expression.accept(this);
                                        }
                                        if(debug) System.out.println(") ;");
                                        ret.set(2, ret.get(2) + ") ; ");
                                        return false;
                                    }

                                    // For Statement
                                    @Override
                                    public boolean visit(ForStatement node) {
                                        if(debug) System.out.println("for(");
                                        ret.set(2, ret.get(2) +"for ( ");
                                        List initializers = node.initializers();
                                        if(initializers != null) {
                                            for(int i = 0; i < initializers.size(); i++) {
                                                ((ASTNode) (initializers.get(i))).accept(this);
                                            }
                                        }
                                        if(debug) System.out.println(";");
                                        ret.set(2, ret.get(2) +"; ");

                                        Expression expression = node.getExpression();
                                        if(expression != null) {
                                            expression.accept(this);
                                        }
                                        if(debug) System.out.println(";");
                                        ret.set(2, ret.get(2) +"; ");

                                        List updaters = node.updaters();
                                        if(updaters != null) {
                                            for(int i = 0; i < updaters.size(); i++) {
                                                ((ASTNode) (updaters.get(i))).accept(this);
                                            }
                                        }
                                        if(debug) System.out.println(")");
                                        ret.set(2, ret.get(2) +") ");
                                        Statement body = node.getBody();
                                        if(debug) System.out.println("{");
                                        ret.set(2, ret.get(2) +"{ ");
                                        if(body != null) {
                                            body.accept(this);
                                        }
                                        if(debug) System.out.println("}");
                                        ret.set(2, ret.get(2) +"} ");
                                        return false;
                                    }

                                    //Enhanced For Statement
                                    @Override
                                    public boolean visit(EnhancedForStatement node) {
                                        if(debug) System.out.println("for(");
                                        ret.set(2, ret.get(2) + "for ( ");
                                        Expression expression = node.getExpression();
                                        if(expression != null) {
                                            expression.accept(this);
                                        }
                                        if(debug) System.out.println(")");
                                        ret.set(2, ret.get(2) + ") ");
                                        Statement body = node.getBody();
                                        if(debug) System.out.println("{");
                                        ret.set(2, ret.get(2) + "{ ");
                                        if(body != null) {
                                            body.accept(this);
                                        }
                                        if(debug) System.out.println("}");
                                        ret.set(2, ret.get(2) + "} ");
                                        return false;
                                    }

                                    // Switch
                                    @Override
                                    public boolean visit(SwitchStatement node) {
                                        Expression expression = node.getExpression();
                                        if(debug) System.out.println("Switch ( ");
                                        ret.set(2, ret.get(2) + "Switch ( ");
                                        if(expression != null) {
                                            expression.accept(this);
                                        }
                                        if(debug) System.out.println(") ");
                                        ret.set(2, ret.get(2) + ") ");
                                        List statements = node.statements();
                                        if(debug) System.out.println("{ ");
                                        ret.set(2, ret.get(2) + "{ ");
                                        if(statements != null) {
                                            for(int i = 0; i < statements.size(); i++) {
                                                ((Statement) (statements.get(i))).accept(this);
                                            }
                                        }
                                        if(debug) System.out.println("} ");
                                        ret.set(2, ret.get(2) + "} ");
                                        return false;
                                    }

                                    // switch case
                                    @Override
                                    public boolean visit(SwitchCase node ) {
                                        if(node.isDefault()) {
                                            if(debug) System.out.println("default : ");
                                            ret.set(2, ret.get(2) + "default : ");
                                        }
                                        else {
                                            Expression expression = node.getExpression();
                                            if(debug) System.out.println("case ");
                                            ret.set(2, ret.get(2) + "case ");
                                            if(expression != null) {
                                                expression.accept(this);
                                            }
                                            if(debug) System.out.println(": ");
                                            ret.set(2, ret.get(2) + ": ");
                                        }
                                        return false;
                                    }

                                    @Override
                                    public boolean visit(BreakStatement node) {
                                        if(node != null) {
                                            if(debug) System.out.println("break");
                                            ret.set(2, ret.get(2) + "break ");
                                        }
                                        return false;
                                    }

                                    @Override
                                    public boolean visit(ContinueStatement node) {
                                        if(node != null) {
                                            if(debug) System.out.println("continue");
                                            ret.set(2, ret.get(2) + "continue ");
                                        }
                                        return false;
                                    }

                                    @Override
                                    public boolean visit(ReturnStatement node) {
                                        if(node != null) {
                                            if(debug)
                                                System.out.println("return (");
                                            ret.set(2, ret.get(2) + "return ( ");
                                            Expression expression = node.getExpression();
                                            if(expression != null) {
                                                expression.accept(this);
                                            }
                                        }
                                        return false;
                                    }

                                    @Override
                                    public void endVisit(ReturnStatement node) {
                                        if(node != null) {
                                            if(debug)
                                                System.out.println(") ");
                                            ret.set(2, ret.get(2) + ") ");
                                        }
                                    }

                                    // method invocation
                                    @Override
                                    public void endVisit(MethodInvocation node) {
                                        Expression expression = node.getExpression();
                                        if(expression != null) {
                                            ITypeBinding  typeBinding = expression.resolveTypeBinding();
                                            if (typeBinding != null) {
                                                String qualifiedName = typeBinding.getQualifiedName();
//                                    String name = typeBinding.getName();
//                                    retFull.add(qualifiedName + " " + node.getName());
                                                if(isJdkApi(qualifiedName)) {
//                                                cnt[0] += 1;
                                                    String qualifiedFullName = qualifiedName + "." + node.getName();

                                                    if(debug)
                                                        System.out.println(qualifiedFullName);

                                                    ret.set(2, ret.get(2) +qualifiedFullName + " ");
                                                }


//                                                if(retFull.size() <= 1)
//                                                    retFull.add(qualifiedFullName);
//                                                else
//                                                    retFull.set(1, retFull.get(1) + "|" + qualifiedFullName);
                                    /*if(isJdkApi(qualifiedName)) {
//                                        ret.add(qualifiedName + " " + node.getName());
                                        if(ret.size() <= 2)ret.add(qualifiedFullName);
                                        else ret.set(2, ret.get(2) + "|" + qualifiedFullName);
                                    }*/
                                            }
                                        }
                                    }

                                    // class instance creation
                                    @Override
                                    public void endVisit(ClassInstanceCreation node) {
                                        ITypeBinding binding = node.getType().resolveBinding();
                                        if(binding == null)
                                            return;
                                        String qualifiedName = binding.getQualifiedName();
//                            String name = binding.getName();
                                        if(isJdkApi(qualifiedName)) {
//                                        cnt[0] += 1;
                                            String qualifiedFullName = qualifiedName + ".new ";
                                            if(debug)
                                                System.out.println(qualifiedFullName);
                                            ret.set(2, ret.get(2) +qualifiedFullName);
                                        }
                                    }

                                });
                            }

                            if(ret.size() >= 3 && ret.get(2) != "" && !ret.contains("return ( ) ")){
                                    String api_search = ret.get(1).toString();
                                    if(api_search.length() >= 2){
                                        api_sequence.add(api_search.substring(2, api_search.length()));

                                        //api_sequence.add(ret.get(1));
                                        api_usage.add(ret.get(2).toString());
                                    }
                            }
                            return false;
                        }
                    }
        );
    }


}


