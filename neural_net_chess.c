#include<stdlib.h>
#include<stdio.h>

//ideas: have something genrate a list of legal moves and evaluate probability of win for each move, play the move with highest prob.
//before i do this, i want to create a gradient descent algorithm for handwritinng recognition. then build neural net for handwriting 
//recognition using the gradient descent algorithm for previous. 

//so gradient descent for handwriting recognition finally!

int main(int argc, char* argv[]){
	
	//first, we want to be able to set the initial coefficients, simply so that each time we stop we can pick up where we left off, 
	// in terms of the training of the algorithm.
	
	/*
		so we take an argument n, specifying the number of features/variables
		then a binary file containg n values for initial for the coefficients. 
		if file DNE, then initial coefficients are set to random numbers between 0-9.9 each.
		the last argument specifies if we want to step trough the training one by one or train all at once and then go through test data.
	*/
	
	
	
	int n = strtol(argv[1], NULL, 10);
	int *regression_coefficients = malloc(n*sizeof(int));
	FILE *fp = fopen(argv[2], "rb");
	if (fp == NULL){
		// set each initial value to a random number from 0.0 to 9.9
	}
	// read from file and
	for (int i = 0; i<n; i++){
		int *coeff;
		fread(coeff, sizeof(int), 1, fp);
		regression_coefficients[i] = *coeff;
	}
	
	