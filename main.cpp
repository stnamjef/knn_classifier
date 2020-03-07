#include <iostream>
#include <Eigen/Dense>
#include "file_manage.h"
#include "neighbors.h"
#include "model_selection.h"
using namespace std;
using namespace Eigen;

int main()
{
	MatrixXd df;
	VectorXd labels;

	read_csv("iris.csv", df, labels, 150, 5);
	
	KNeighbors knn(4);
	double accuracy = evaluate_model(knn, df, labels, 5);

	cout << "Model accuracy: " << accuracy * 100 << "%" << endl;

	return 0;
}