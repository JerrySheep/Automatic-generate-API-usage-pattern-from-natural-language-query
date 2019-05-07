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
