#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
//ideas: have something genrate a list of legal moves and evaluate probability of win for each move, play the move with highest prob.
//before i do this, i want to create a gradient descent algorithm for handwritinng recognition. then build neural net for handwriting 
//recognition using the gradient descent algorithm for previous. 

//so gradient descent for handwriting recognition finally!

int MAXSIZE = 1000;
int NUMPIXELS = 784;
int MAX_NAME = 128;



float get_label(char *filename) {
 
    char *dash_char = strstr(filename, "-");
    return (float) atoi(dash_char + 1);
}

void load_image(char *filename, float *img) {
    // Open corresponding image file, read in header (which we will discard)
    FILE *f2 = fopen(filename, "r");
    if (f2 == NULL) {
        perror("fopen: load_image");
        exit(1);
    }
	
	int height, width;
	fscanf(f2, "P2 %d %d 255", &width, &height);
	for(int i = 0; i < height;i++){
		for (int j = 0; j<width; j++){
			unsigned char val;
			fscanf(f2, "%hhu", &val);
			img[i*width + j] = (float)val/255.0;
		}
	}

    fclose(f2);
}

int loadDataset(char* file_name, float dataset[MAXSIZE][NUMPIXELS], float* labels){
	
	FILE *f1 = fopen(file_name, "r");
    if (f1 == NULL) {
		
        perror("fopen: loadDataset");
        exit(1);
    }
	int count = 0;
	char line[MAX_NAME+1];
	for (int i = 0;i<MAXSIZE; i++){
		if(fscanf(f1, "%s", line)==0){
			
			break;
		}
		
		load_image(line, dataset[i]);
		labels[i] = get_label(line);
		count+=1;
	}
    fclose(f1);
    return count;
}

float* multiply(float* x, float img[MAXSIZE][NUMPIXELS], int rows){
	float* ret_ptr = malloc(rows*sizeof(float));
	for(int i = 0; i< rows; i++){
		float sum = 0.0;
		for(int j = 0; j<NUMPIXELS; j++){
			sum+= x[j]*img[i][j];
		}
		ret_ptr[i] = sum;
	}
	return ret_ptr;
	
}

//calculates the error in prediction by calculating RS error of all predictions in the dataset
float root_square_error(float* x, float* y, int size){
	double dst_sq = 0.0;
	for (int i = 0; i < size; i++){
		dst_sq += (x[i]-y[i])*(x[i]-y[i]);
	}
    return (float) sqrt(dst_sq);
}

float* gradient_cal(float* actual, float* prediction, float x[MAXSIZE][NUMPIXELS], int size){
	
	float* grad_ptr = malloc(NUMPIXELS*sizeof(float));
	
	for(int j = 0; j<NUMPIXELS; j++){
		float del_E_j = 0;
		for(int i = 0; i<size; i++){
			del_E_j += 2*x[i][j]*(prediction[i] - actual[i]);
		}
		grad_ptr[j] = del_E_j;
	}
	return grad_ptr;
}
	
//y is the actual values of the labels, x is the training dataset, size is the number of imgaes in s	
float gradient_descent(float* coeff, float x[MAXSIZE][NUMPIXELS], float* y, int size, float lr){
	
	
	float* pred_y = multiply(coeff, x, size);
	
	float* gradient = gradient_cal(y, pred_y, x, size);
	
	for(int i = 0; i<NUMPIXELS; i++){
		coeff[i] -= lr*gradient[i];
	}
	float error = root_square_error(y, pred_y, size);
	return error;
}

void test(float* coeff, float data[MAXSIZE][NUMPIXELS], float* labels, int size){
	int num_correct = 0;
	float* pred_y = multiply(coeff, data, size);
	for(int i = 0; i<size; i++){
		if(round(pred_y[i])==(int)labels[i]){
			num_correct+=1;
		}
	}
	float accuracy = (float)num_correct/size;
	printf("accuracy: %f\n", accuracy);
}
	
	
void print_menu(){
	printf("1. continue \n");
	printf("2. test accuracy \n");
	printf("0. exit \n");
	
}

int main(int argc, char* argv[]){
	
	//first, we want to be able to set the initial coefficients, simply so that each time we stop we can pick up where we left off, 
	// in terms of the training of the algorithm.
	
	/*
		so we take an argument n, specifying the number of features/variables
		then a binary file containg n values for initial for the coefficients. 
		if file DNE, then initial coefficients are set to random numbers between 0-9.9 each.
		then the training dataset and test dataset
		
	*/
	float LEARNING_RATE = 0.00002;
	
	float training[MAXSIZE][NUMPIXELS];

	float testing[MAXSIZE][NUMPIXELS];
	float training_labels[MAXSIZE];
	float testing_labels[MAXSIZE];
	
	char* endptr;
	//number of features, also the same as the number of pixels
	
	int n = strtol(argv[1], &endptr, 10);
	
	float *regression_coefficients = malloc(n*sizeof(float));
	if(endptr == argv[1]){
		printf("expecting number");
		exit(1);
	}
	
	
	/*
	FILE *fp = fopen(argv[2], "rb");
	if (fp == NULL){
		// set each initial value to a random number from 0.0 to 9.9
	}
	//read from file and initialise coefficients
	for (int i = 0; i<n; i++){
		float coeff;
		fread(&coeff, sizeof(float), 1, fp);
		regression_coefficients[i] = coeff;
	}
	*/
	
	for (int i = 0; i<n ; i++){
		regression_coefficients[i] = 1;
	}
	
	int training_size = loadDataset(argv[2], training, training_labels); //3rd argument is the file containing list of training dataset
	int testing_size = loadDataset(argv[3], testing, testing_labels); //4th argument is the file containing testing dataset
	if (testing_size == 0){
		printf("%d\n", training_size);
		exit(1);
	}
	float curr_error = gradient_descent(regression_coefficients, training, training_labels, training_size, LEARNING_RATE);
	while(curr_error > 30){
		
		printf("%f\n", curr_error);
		float new_error = gradient_descent(regression_coefficients, training, training_labels, training_size, LEARNING_RATE);
		if ((curr_error - new_error)/new_error <0.1){
			LEARNING_RATE*=1.00000005;
		}

		//hello
		//hello again

		if(new_error > curr_error){
			break;
		}
		
		curr_error = new_error;
		/*
		print_menu();
		char option;
		scanf("%c", &option);
		
		if(option == '1'){
			
		}
		else if(option == '2'){
			test(regression_coefficients, testing, testing_labels, testing_size);
			test(regression_coefficients, training, training_labels, training_size);
		}
		else if(option == '0'){
			exit(0);
		}
		else{continue;}
		*/
	}
	test(regression_coefficients, testing, testing_labels, testing_size);
	test(regression_coefficients, training, training_labels, training_size);
	return 1;
}