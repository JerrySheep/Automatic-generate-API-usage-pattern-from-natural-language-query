# This is a project which automatic generate API usage pattern from natural language query

The first is collect the data from the real GitHub Projects.
GHTorrent can provide the GitHub project information for us. You can download the csv files from http://www.ghtorrent.org/downloads.html


The second is store these csv files into the MYSQL database.

In this project, we collect the real Java projects from the database for further learning.

You can find the url and other attibutes of every projects and transfer these information to an sentence for git clone.

Database tables just like :

![database_tables](https://github.com/JerrySheep/Automatic-generate-API-usage-pattern-from-natural-language-query/blob/master/img/database_tables.png)

Database table project just like :

![database_table_project](https://github.com/JerrySheep/Automatic-generate-API-usage-pattern-from-natural-language-query/blob/master/img/database_project.png)

From the information provided from the database, you can download the projects with attributes you want. (In my project, I collect the Java Projects which created after 2014 and stars above 10)


The third is collect the projects with "git clone".

just like : "git clone https://github.com/JerrySheep/Automatic-generate-API-usage-pattern-from-natural-language-query.git"

You can find the information in file "GitCloneWithThread".
In my project, I collect about 45k Java projects. (It took lots of time, you can have another try. If it is possible, tell me~)

The forth is collect the data set.
In a java file, we collect the annotation as natural language query sentence and collect the source code(transfered by abstrct syntax tree later) as API usage pattern sentence.

You can find the information in file "JavaParserWithThread". (some code is from my senior Tian Yanfei, I might rewrite later)


