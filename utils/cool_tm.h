// #include <bits/stdc++.h>

#define PREDICT 1
#define UPDATE 0
#define NUMBER_OF_STATES 400

struct TsetlinMachine { 

	int number_of_clauses;
	int number_of_features;
	int number_of_classes;
	int number_of_weights; 

	unsigned int ***TAs;
	unsigned int *clauses;
	unsigned int *feedback_to_clauses;
	signed int **weights; 

	int T;
	double s;
	signed int class_sum;
	int clause_1_count;
	int boost_true_positive_feedback;
	unsigned long long int rand_count; 
};

struct TsetlinMachine *CreateCoalescedTsetlinMachine();

void tm_initialize(struct TsetlinMachine *tm);

int tm_ta_state(struct TsetlinMachine *tm, int clause, int feature, int ta);

void tm_update(struct TsetlinMachine *tm, int Xi[], int target, int class, float s);

int tm_score(struct TsetlinMachine *tm, int Xi[], int class);

int tm_get_clause_weight(struct TsetlinMachine *tm, int class, int clause); 

int get_rands_used(struct TsetlinMachine *tm);

void rands_reset(struct TsetlinMachine *tm);

void tm_load_TAs_load_Weights(struct TsetlinMachine *tm, int number_of_TAs, int number_of_Weights, int TAs[], int Weights[]);

void tm_analyse_inference(struct TsetlinMachine *tm, int Xi[]);