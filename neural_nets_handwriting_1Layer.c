#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
//ideas: have something genrate a list of legal moves and evaluate probability of win for each move, play the move with highest prob.
//before i do this, i want to create a gradient descent algorithm for handwritinng recognition. then build neural net for handwriting 
//recognition using the gradient descent algorithm for previous. 

//so gradient descent for handwriting recognition finally!

int MAXSIZE = 5000;
int NUMPIXELS = 784;
int MAX_NAME = 128;
int LAYER1 = 784;
int LAYER2 = 16;
int LAYER3 = 10;

unsigned char get_label(char *filename) {
 
    char *dash_char = strstr(filename, "-");
    return (char) atoi(dash_char + 1);
}

void load_image(char *filename, unsigned char *img) {
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
			//unsigned char val;
			fscanf(f2, "%hhu", &img[i*width + j]);
		}
	}

    fclose(f2);
}
/*
int loadDataset(char* file_name, unsigned char dataset[MAXSIZE][NUMPIXELS], signed char* labels){
	
	FILE *f1 = fopen(file_name, "r");
    if (f1 == NULL) {
		
        perror("fopen: loadDataset");
        exit(1);
    }
	int count = 0;
	char line[MAX_NAME+1];
	int label_count[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	for (int i = 0;i<MAXSIZE; i++){
		if(fscanf(f1, "%s", line)!=1){
			
			break;
		}
		
		load_image(line, dataset[i]);
		unsigned char label = get_label(line);
		labels[i] = label - 5;
		label_count[label]+=1;
		count+=1;
	}
    fclose(f1);
	for(int i = 0; i< 10; i++){
		printf("%d: %d\n", i, label_count[i]);
	}
    return count;
}
*/
int loadDataset(char* file_name, unsigned char **dataset, signed char* labels){
	
	FILE *f1 = fopen(file_name, "r");
    if (f1 == NULL) {
		
        perror("fopen: loadDataset");
        exit(1);
    }
	int count = 0;
	char line[MAX_NAME+1];
	int label_count[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	for (int i = 0;i<MAXSIZE; i++){
		if(fscanf(f1, "%s", line)!=1){
			
			break;
		}
		
		load_image(line, dataset[i]);
		unsigned char label = get_label(line);
		labels[i] = label - 5;
		label_count[label]+=1;
		count+=1;
	}
    fclose(f1);
	for(int i = 0; i< 10; i++){
		printf("%d: %d\n", i, label_count[i]);
	}
    return count;
}

float* multiply(double* x, unsigned char **img, int rows, int cols){
	float* ret_ptr = malloc(rows*sizeof(float));
	for(int i = 0; i< rows; i++){
		float sum = 0.0;
		for(int j = 0; j<cols; j++){
			sum+= x[j]*img[i][j];
		}
		ret_ptr[i] = sum;
	}
	return ret_ptr;
	
}
/*
float* multiply(double* x, double* y, int rows){
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
*/
//calculates the error in prediction by calculating RS error of all predictions in the dataset
float root_square_error(float* x, signed char* y, int size, int true){
	double dst_sq = 0.0;
	for (int i = 0; i < size; i++){
		if(y[i] == true - 5){
			dst_sq += (x[i]-1)*(x[i]-1);
		}
		else{
			dst_sq += (x[i]-0)*(x[i]-0);
		}
	}
    return (float) sqrt(dst_sq);
}

float* gradient_hidden(float* req, float* pred, float** hidden_data, int col){
	
	float* grad_ptr = malloc(LAYER1*sizeof(float));
	
	for(int j = 0; j<LAYER1; j++){
		float del_E_j = 0;
		for(int i = 0; i<col; i++){
			del_E_j += 2*hidden_data[i][j]*(pred[i] - req[i]);
		}
		grad_ptr[j] = del_E_j;
	}
	return grad_ptr;
}

float* gradient_cal(signed char* actual, float* prediction, unsigned char **x, int size, int true){
	
	float* grad_ptr = malloc(LAYER2*sizeof(float));
	
	for(int j = 0; j<LAYER2; j++){
		float del_E_j = 0;
		for(int i = 0; i<size; i++){
			if(true - 5 == actual[i]){
				del_E_j += 2*x[i][j]*(prediction[i] - 1);
			}
			else{
				del_E_j += 2*x[i][j]*(prediction[i] - 0);
			}
		}
		grad_ptr[j] = del_E_j;
	}
	return grad_ptr;
}
	
//y is the actual values of the labels, x is the training dataset, size is the number of imgaes in s	
float gradient_descent(double* coeff, unsigned char **x, signed char* y, int size, double lr, int true_label, int layer){
	
	
	float* pred_y = multiply(coeff, x, size, layer);
	
	float* gradient = gradient_hidden(y, pred_y, x, size, true_label);
	
	for(int i = 0; i<layer; i++){
		coeff[i] -= lr*gradient[i];
	}
	free(gradient);
	
	//float error = root_square_error(pred_y, y, size, true_label);
	//printf("error: %f", error);
	free(pred_y);
	return 0.0;
	//return error;
}

void test(double **coeff1, double** coeff2 unsigned char **data, signed char* labels, int size){
	int num_correct = 0;
	
	for(int i = 0; i<size; i++){
		
		float second_layer[LAYER2];
		for(int k = 0; k<LAYER2; k++){
			float sum = 0;
			for(int j = 0; j<LAYER1; j++){
				sum+= coeff1[k][j]*data[i][j];
			}
			second_layer[k] = sum;
		}
		
		int max_label = 0;
		float max_sum = 0;
		for(int k = 0; k<LAYER3; k++){
			
			float sum = 0;
			for (int j = 0; j< LAYER2; j++){
				sum+= second_layer[j]*coeff2[k][j];
			}
			if (sum > max_sum){
				max_label = k;
				max_sum = sum;
			}
			//printf("%f\n", sum);
		}
		//printf("%d, %d\n", max_label, (int)labels[i]);
		if(max_label == labels[i]+5){
			num_correct++;
		}
	}
	float accuracy = (float)num_correct/size;
	/*
	if(accuracy > 0.845){
		FILE *fp = fopen("H_recog_coeff.txt", "w");
		for (int i = 0; i<NUMPIXELS; i++){
			fprintf(fp, "%f\n", coeff[i]);
		}
		fclose(fp);
		
		exit(1);
		
	}
	*/
	//printf("Total correct: %d\n", num_correct);
	//printf("Total data: %d\n", size);
	printf("accuracy: %f\n", accuracy);
}

void calculate_hidden_layer(float** ret_layer, double** coeff, unsigned char** data, int size){
		
	for(int j = 0; j<LAYER2; j++){
		float* col = multiply(coeff[j], data);
		for(int i = 0; i<size; i++){
			ret_layer[i][j] = col[i];
		}
		free(col);
	}
}		

void get_req_hidden_layer(float** ret_layer, double** coeff, signed char* label, int size){
	for(int i = 0; j<size; j++){
		int index = 5+label[i];
		float max_weight=0.0;
		for(int j = 0; j<LAYER2; j++){
			if(coeff[index][j] > max_weight){
				max_weight = coeff[index][j];
			}
		}
		for(int j = 0; j<LAYER2; j++){
			if(coeff[index][j] >=0){
				ret_layer[i][j] = coeff[index][j]/max_weight;
			}
			else{
				ret_layer[i][j] = 0;
			}
		}
	}
}

void print_menu(){
	printf("1. continue \n");
	printf("2. test accuracy \n");
	printf("0. exit \n");
	
}
/*
	so we take an argument n, specifying the number of features/variables
	then a binary file containg n values for initial for the coefficients. 
	if file DNE, then initial coefficients are set to random numbers between 0-9.9 each.
	then the training dataset and test dataset
	
*/
int main(int argc, char* argv[]){
	
	
	double LEARNING_RATE = 0.000000000003; //0.0000000003 is a good rate for 1k dataset, 0.000000000003 is a good learning rate for 5k dataset
	
	//unsigned char training[MAXSIZE][NUMPIXELS];

	//unsigned char testing[MAXSIZE][NUMPIXELS];
	signed char training_labels[MAXSIZE];
	signed char testing_labels[MAXSIZE];
	
	char* endptr;
	int n = strtol(argv[1], &endptr, 10);
	if(endptr == argv[1]){
		printf("expecting number");
		exit(1);
	}
	int tr_size = strtol(argv[4], &endptr, 10);
	if(endptr == argv[4]){
		printf("expecting number");
		exit(1);
	}
	unsigned char** training = malloc(tr_size*sizeof(unsigned char *));
	if(training == NULL){
		fprintf(stderr, "malloc training:");
		exit(1);
	}
	for(int i = 0; i<tr_size; i++){
		training[i] = malloc(n*sizeof(unsigned char));
		if(training[i] == NULL){
			fprintf(stderr, "malloc training[%d]:", i);
			exit(1);
		}
	}
	
	int te_size = strtol(argv[5], &endptr, 10);
	if(endptr == argv[5]){
		printf("expecting number");
		exit(1);
	}
	unsigned char** testing = malloc(te_size*sizeof(unsigned char *));
	if(testing == NULL){
		fprintf(stderr, "malloc testing:");
		exit(1);
	}
	for(int i = 0; i<te_size; i++){
		testing[i] = malloc(n*sizeof(unsigned char));
		if(training == NULL){
			fprintf(stderr, "malloc testing[%d]:", i);
			exit(1);
		}
	}
	
	double **regression_coefficients1 = malloc(LAYER2*sizeof(double*));
	for(int i = 0; i<LAYER2; i++){
		regression_coefficients1[i] = malloc(LAYER1*sizeof(double));
	}
	
	for (int i = 0; i<LAYER2 ; i++){
		for(int j = 0; j<LAYER1; j++){
			//regression_coefficients[i][j] = 0.5 - (float)(rand()%1000)/1000.0;
			//printf("%f\n", regression_coefficients[i]);
			regression_coefficients1[i][j] = 0;
		}
	}
	
	double **regression_coefficients2 = malloc(LAYER3*sizeof(double*));
	for(int i = 0; i<LAYER3; i++){
		regression_coefficients1[i] = malloc(LAYER2*sizeof(double));
	}
	
	for (int i = 0; i<LAYER3 ; i++){
		for(int j = 0; j<LAYER2; j++){
			//regression_coefficients[i][j] = 0.5 - (float)(rand()%1000)/1000.0;
			//printf("%f\n", regression_coefficients[i]);
			regression_coefficients2[i][j] = 0;
		}
	}
	
	float **hidden_Layer = malloc(tr_size*sizeof(float*));
	for(int i = 0; i<tr_size; i++){
		hidden_Layer[i] = malloc(LAYER2*sizeof(float));
	}
	
	calculate_hidden_layer(hidden_Layer, regression_coefficients1, training, tr_size);
	
	float **req_hidden_Layer = malloc(tr_size*sizeof(float*));
	for(int i = 0; i<tr_size; i++){
		req_hidden_Layer[i] = malloc(LAYER2*sizeof(float));
	}
	
	
	
	printf("loading training dataset...\n");
	int training_size = loadDataset(argv[2], training, training_labels); //3rd argument is the file containing list of training dataset
	printf("loading testing dataset...\n");
	int testing_size = loadDataset(argv[3], testing, testing_labels); //4th argument is the file containing testing dataset
	if (testing_size == 0){
		printf("%d\n", training_size);
		exit(1);
	}
	printf("calculatin error\n");
	//float curr_error = gradient_descent(regression_coefficients, training, training_labels, training_size, LEARNING_RATE);
	while(1){
		//test(regression_coefficients, testing, testing_labels, testing_size);
		//printf("%f\n", curr_error);
		//float new_error = gradient_descent(regression_coefficients, training, training_labels, training_size, LEARNING_RATE);
		//printf("%f\n", new_error);
		/*
		if ((curr_error - new_error)/new_error >0.1){
			LEARNING_RATE*=0.999995;
		}
		
				
		if(new_error > curr_error){
			break;
		}
		
		curr_error = new_error;
		*/
		/*
		print_menu();
		char option;
		scanf("%c", &option);
		
		if(option == '1'){
			for(int i = 0; i<10; i++){
				float err = gradient_descent(regression_coefficients[i], training, training_labels, training_size, LEARNING_RATE, i);
			}
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
		for(int i = 0; i<LAYER3; i++){
			float err = gradient_descent(regression_coefficients2[i], hidden_Layer, training_labels, training_size, LEARNING_RATE, i, LAYER2);
		}
		
		
		get_req_hidden_layer(req_hidden_Layer, regression_coefficients2, training_labels, tr_size);
		
		for(int i = 0; i<LAYER2; i++){
			float err = gradient_descent(regression_coefficients1[i], training, req_hidden_Layer, training_size, LEARNING_RATE, i, LAYER1);
		}
		calculate_hidden_layer(hidden_Layer, regression_coefficients1, training, tr_size);
		//test(regression_coefficients, testing, testing_labels, testing_size);
	}
	test(regression_coefficients1, regression_coefficients2, testing, testing_labels, testing_size);
	test(regression_coefficients1, regression_coefficients2, training, training_labels, training_size);
	return 1;
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