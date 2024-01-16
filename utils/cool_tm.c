#include "cool_tm.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include "fast_rand.h"

struct TsetlinMachine *CreateCoalescedTsetlinMachine(int number_of_classes, int number_of_clauses, int number_of_features, int T, double s ,int boost_true_positive_feedback, int class_sum)
{
	struct TsetlinMachine *tm = (void *)malloc(sizeof(struct TsetlinMachine));

	tm->number_of_classes = number_of_classes;
	tm->number_of_clauses = number_of_clauses;
	tm->number_of_features = number_of_features;
	tm->number_of_weights = tm->number_of_clauses*tm->number_of_classes;

	tm->clauses = (unsigned int *)malloc(sizeof(unsigned int) * tm->number_of_clauses);
	tm->TAs = (unsigned int***)malloc(tm->number_of_clauses* sizeof(unsigned int**));
	for (int i = 0; i < tm->number_of_clauses; i++){
		tm->TAs[i] = (unsigned int**)malloc(tm->number_of_features * sizeof(unsigned int*));
			for (int j = 0; j < tm->number_of_features; j++){
				tm->TAs[i][j] = (unsigned int*)malloc(2 * sizeof(unsigned int));
			}
	}
	tm->weights = (signed int **)malloc(tm->number_of_classes*sizeof(signed int*)); 
	for(int i = 0; i < tm->number_of_classes; i++){
		tm->weights[i] = (signed int*)malloc(tm->number_of_clauses * sizeof(signed int));
		for(int j = 0; j <tm->number_of_clauses; j++){
			// tm->rand_count ++; 
			if(1.0 * rand()/RAND_MAX <= 0.5){
				tm->weights[i][j] = 1; 
			}
			else{
				tm->weights[i][j] = -1; 
			}
		}
	}

	tm->feedback_to_clauses = (unsigned int *)malloc(sizeof(unsigned int) * tm->number_of_clauses);
	tm->T = T;
	tm->s = s;
	tm->class_sum = class_sum;
	tm->rand_count = 0;
	tm->boost_true_positive_feedback = boost_true_positive_feedback;
	tm->clause_1_count = 0;

	tm_initialize(tm);

	return tm;
}

void tm_load_TAs_load_Weights(struct TsetlinMachine *tm, int number_of_TAs, int number_of_Weights, int TAs_file[], int Weights_file[])
{
	int TA = 0; 
	for(int i = 0; i < tm->number_of_clauses; i++){
		for(int j = 0; j< tm->number_of_features; j++){
			for(int k = 0; k < 2; k++){
				tm->TAs[i][j][k] = TAs_file[TA];
				TA++; 
			}
		}
	}
	int Weight = 0;
	for(int i = 0; i < tm->number_of_classes; i++){
		for(int j = 0; j < tm->number_of_clauses; j++){
			tm->weights[i][j] = Weights_file[Weight];
			Weight++; 
		}
	}
}

void tm_initialize(struct TsetlinMachine *tm)
{
	for (int j = 0; j < tm->number_of_clauses; ++j) {
		for (int k = 0; k < tm->number_of_features; ++k) {
			for (int b = 0; b < 2; ++b) {	
				if (1.0 * (float)fast_rand()/(float)FAST_RAND_MAX <= 0.5) {
					tm->TAs[j][k][b] = NUMBER_OF_STATES +1;
				}
				else{
					tm->TAs[j][k][b] = NUMBER_OF_STATES -1; 
				}
			}	
		}	
	}
}

int tm_ta_state(struct TsetlinMachine *tm, int clause, int feature, int literal)
{
	return tm->TAs[clause][feature][literal];
}


int tm_get_clause_weight(struct TsetlinMachine *tm, int class, int clause){
	return tm->weights[class][clause];
}

static inline int action(int state)
{
	return state > NUMBER_OF_STATES;
}

static inline void calculate_clause_output(struct TsetlinMachine *tm, int Xi[], int predict)
{
	int j, k;
	int action_include, action_include_negated;
	int all_exclude;

	for (j = 0; j < tm->number_of_clauses; j++) {
		tm->clauses[j] = 1;
		all_exclude = 1;
		for (k = 0; k < tm->number_of_features; k++) {
			action_include = action(tm->TAs[j][k][0]);
			action_include_negated = action(tm->TAs[j][k][1]);

			all_exclude = all_exclude && !(action_include == 1 || action_include_negated == 1);

			if ((action_include == 1 && Xi[k] == 0) || (action_include_negated == 1 && Xi[k] == 1)) {
				tm->clauses[j] = 0;
				break;
			}
		}
		tm->clauses[j] = tm->clauses[j] && !(predict == PREDICT && all_exclude == 1);
	}
}

static inline int sum_up_class_votes(struct TsetlinMachine *tm, int class)
{
	int class_sum = 0;
	// printf("Running Sum: \n");
	for (int j = 0; j < tm->number_of_clauses; j++) {
		class_sum += tm->weights[class][j]*tm->clauses[j];
		// printf("CS:%d C:%d W:%d ", class_sum, j, tm->weights[class][j]);
		// printf("CS:%d ", class_sum);
	}
	if(class_sum > tm->T){
		class_sum = (signed int)tm->T; 
	}
	else if(class_sum < -1*tm->T){
		class_sum = (signed int)-1*tm->T; 
	}
	// printf("Class Sum: Class[%d]: Sum[%d]\n", class, class_sum);
	return class_sum;
}

static inline void type_i_feedback(struct TsetlinMachine *tm, int Xi[], int j, float s, float update_probability)
{
	if ((((float)fast_rand()/(float)FAST_RAND_MAX)) <= update_probability)  {
		if (tm->clauses[j] == 0)	{
			for (int k = 0; k < tm->number_of_features; k++) {
				
				tm->TAs[j][k][0] -= (tm->TAs[j][k][0] > 1) && (((float)fast_rand()/(float)FAST_RAND_MAX) <= 1.0/s);
				// tm->rand_count ++; 


				tm->TAs[j][k][1] -= (tm->TAs[j][k][1] > 1) && (((float)fast_rand()/(float)FAST_RAND_MAX) <= 1.0/s);
				// tm->rand_count ++; 
			}
		} else if (tm->clauses[j] == 1) {					
			for (int k = 0; k < tm->number_of_features; k++) {
				if (Xi[k] == 1) {
					tm->TAs[j][k][0] += (tm->TAs[j][k][0] < NUMBER_OF_STATES*2) && (tm->boost_true_positive_feedback == 1 || ((float)fast_rand()/(float)FAST_RAND_MAX) <= (s-1)/s);

					// tm->rand_count ++; 

					tm->TAs[j][k][1] -= (tm->TAs[j][k][1] > 1) && (((float)fast_rand()/(float)FAST_RAND_MAX) <= 1.0/s);

					// tm->rand_count ++; 
				} else if (Xi[k] == 0) {
					tm->TAs[j][k][1] += (tm->TAs[j][k][1] < NUMBER_OF_STATES*2) && (tm->boost_true_positive_feedback == 1 || ((float)fast_rand()/(float)FAST_RAND_MAX) <= (s-1)/s);
					
					// tm->rand_count ++; 
					tm->TAs[j][k][0] -= (tm->TAs[j][k][0] > 1) && (((float)fast_rand()/(float)FAST_RAND_MAX) <= 1.0/s);

					// tm->rand_count ++; 
				}
			}
		}
	}
}

static inline void type_ii_feedback(struct TsetlinMachine *tm, int Xi[], int j, float update_probability) {
	int action_include;
	int action_include_negated;
	if ((((float)fast_rand()/(float)FAST_RAND_MAX)) <= update_probability)  {	
		if (tm->clauses[j] == 1) {
			for (int k = 0; k < tm->number_of_features; k++) { 
				action_include = action(tm->TAs[j][k][0]);
				action_include_negated = action(tm->TAs[j][k][1]);

				tm->TAs[j][k][0] += (action_include == 0 && tm->TAs[j][k][0] < NUMBER_OF_STATES*2) && (Xi[k] == 0);
				tm->TAs[j][k][1] += (action_include_negated == 0 && tm->TAs[j][k][1] < NUMBER_OF_STATES*2) && (Xi[k] == 1);
			}
		}
	}
}

void tm_update(struct TsetlinMachine *tm, int Xi[], int target, int class, float s) {

	calculate_clause_output(tm, Xi, UPDATE);
	unsigned int negative_target_class = (unsigned int)tm->number_of_classes * (float)fast_rand()/(float)FAST_RAND_MAX;
	
	while (negative_target_class == class) {
		// tm->rand_count ++; 
		negative_target_class = (unsigned int)tm->number_of_classes * (float)fast_rand()/(float)FAST_RAND_MAX;
	}

	tm->class_sum = sum_up_class_votes(tm, class);
	float update_probability = ((float)(tm->T - tm->class_sum)/(float)(2*tm->T));

	for (int j = 0; j < tm->number_of_clauses; j++) {	
		if(tm->weights[class][j]  >= 0){
			type_i_feedback(tm, Xi, j, s, update_probability);
		}				

		if(tm->weights[class][j]  < 0){
			type_ii_feedback(tm, Xi, j, update_probability);
		}
	}
	for (int k =0; k < tm->number_of_clauses; k++){
		// tm->rand_count ++; 
		if (((((float)fast_rand()/(float)FAST_RAND_MAX)) <= update_probability) && (tm->clauses[k]))
		{
			tm->weights[class][k] += 1; 
		}
	}		
	calculate_clause_output(tm, Xi, UPDATE);
	tm->class_sum = sum_up_class_votes(tm, negative_target_class);
	update_probability = ((float)(tm->T + tm->class_sum)/(float)(2*tm->T));

	for (int j = 0; j < tm->number_of_clauses; j++) {
		if(tm->weights[negative_target_class][j]  < 0){
			type_i_feedback(tm, Xi, j, s, update_probability);
		}
		if(tm->weights[negative_target_class][j]  >= 0){
			type_ii_feedback(tm, Xi, j, update_probability);
		}				
	}
	for (int k =0; k < tm->number_of_clauses; k++){
		// tm->rand_count ++; 
		if (((((float)fast_rand()/(float)FAST_RAND_MAX)) <= update_probability) && (tm->clauses[k]))
		{
			tm->weights[negative_target_class][k] -= 1; 
		}
	}	
}

void tm_analyse_inference(struct TsetlinMachine *tm, int Xi[]){

	// print the TA states
//	for(int i = 0; i < tm->number_of_clauses; i++){
//		for(int j = 0; j < tm->number_of_features; j++){
//			for(int k = 0; < 2; k ++){
//				printf("Clause[%d]Feature[%d]TA[%d]: %d\n", i, j, k, tm->TAs[i][j][k]);
//			}
//		}
//	}


	calculate_clause_output(tm, Xi, PREDICT);
	for(int i = 0; i < tm->number_of_clauses; i++){
		printf("%d", tm->clauses[i]);
	}
	printf("\n");
	int exp_y 		= 0;
	int max_class 	= 0;
	for(int class = 0; class < tm->number_of_classes; class++){
		int y = sum_up_class_votes(tm, class);
		if(y > exp_y){
			max_class = class;
			exp_y = y; 
		} 
	}
	printf("Classification: %d\n", max_class);
}

int get_rands_used(struct TsetlinMachine *tm){
	printf("Random Numbers Used: %lld\n", tm->rand_count);
	return tm->rand_count;
}

void rands_reset(struct TsetlinMachine *tm){
	tm->rand_count = 0;
}

int tm_score(struct TsetlinMachine *tm, int Xi[], int class) {

	calculate_clause_output(tm, Xi, PREDICT);
	return sum_up_class_votes(tm, class);
}
