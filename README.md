# Computer-Science-For-Business-Adminstration

This file contains the code for the assignment for Computer Science For Business Administration.

The code is writting in python. It contains a duplicate detection method using LSH as a preselection method and uses cosine similarity to calculate similarities between found candidate pairs. Bootstrappng is provided in the code to give robust results. In the first part of the code data is loaded in and cleaned. Secondly, functions are made which are used for the duplicate detection algorithm. Finally a script is given, whcich performs the duplicate detection. In the list 'bands' the band size for the LSH algorithm can be given. These bands can only be the number {1, 2, 5, 10, 13, 25, 26, 50, 65, 130, 325, 650}. For the classification algorithm a treshold must be manually provided. Please note that it only makes sence to vary this thrseshhold between 0 and 1. 
