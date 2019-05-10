# This is a project which automatic generate API usage pattern from natural language query

## First: collect the data from the real GitHub Projects.
GHTorrent can provide the GitHub project information for us. You can download the csv files from http://www.ghtorrent.org/downloads.html


## Second: store these csv files into the MYSQL database.

In this project, we collect the real Java projects from the database for further learning.

You can find the url and other attibutes of every projects and transfer these information to an sentence for git clone.

Database table project just like :

![database_table_project](https://github.com/JerrySheep/Automatic-generate-API-usage-pattern-from-natural-language-query/blob/master/img/database_project.png)

From the information provided from the database, you can download the projects with attributes you want. (In my project, I collect the Java Projects which created after 2014 and stars above 10)


## Third: collect the projects with "git clone".

just like : "git clone https://github.com/JerrySheep/Automatic-generate-API-usage-pattern-from-natural-language-query.git"

You can find the information in file "GitCloneWithThread".
In my project, I collect about 45k Java projects. (It took lots of time, you can have another try. If it is possible, tell me~)

## Forth: collect the data set.
In a java file, we collect the annotation as natural language query sentence and collect the source code(transfered by abstrct syntax tree later) as API usage pattern sentence.

You can find the information in file "JavaParserWithThread". (some code is from my senior Tian Yanfei, I might rewrite later)

## Fifth: filter the data.
I collect 500w data and after some operation it only contains 162w data.

Data filter rules:

Rule one: delete some useless control flow sentences. Just like: "if ( ) { } else { }", "for ( ; ; ) { }", "while ( ) { }" and etc.

Rule two: delete the repeate sentences in API usage pattern. Just like: "if ( java.io.File.exists ) { java.io.File.mkdir } if ( java.io.File.exists ) { java.io.File.mkdir }" -> "if ( java.io.File.exists ) { java.io.File.mkdir }" and etc.

Rule three: turn the "if ( API_a ) { } else { API_b }" sentence to "if ( API_a ) { API_b }" sentence. Just like: "if ( java.io.File.exists ) { } else { java.io.File.mkdir }" -> "if ( java.io.File.exists) { java.io.File.mkdir }" and etc.

After filter, my data looks like this:

![data_pic](https://github.com/JerrySheep/Automatic-generate-API-usage-pattern-from-natural-language-query/blob/master/img/data.png)

As the data file size is too large to upload to this location, you can email me for the data. (jerrysheep0308@gmail.com & 435390541@qq.com) Both of them is possible.

## Sixth: train the model.

In my project, I use the seq2seq model to train the model with above data set.

My seq2seq model contains a encoder model and a decoder model with attention layer(The luong attention layer).

As you can see in file "model", it has the training python file and evaluation python file.

In training python file, my parameters is setted as follows:


## Seventh: develop the tool
My tool is generated in the Intellij IDEA and used as a plugin platform tool.

Here is my tool bar GUI:

![tool_bar](https://github.com/JerrySheep/Automatic-generate-API-usage-pattern-from-natural-language-query/blob/master/img/toolBar.png)

Here is my tool seacher GUI:
![tool_search](https://github.com/JerrySheep/Automatic-generate-API-usage-pattern-from-natural-language-query/blob/master/img/toolSearch.png)

Here is my tool result GUI:
![tool_result](https://github.com/JerrySheep/Automatic-generate-API-usage-pattern-from-natural-language-query/blob/master/img/toolResult.png)

You can find the information for generate a plugin tool in file "APITool" and find the needed file "pythonTool.py" in file "model".



