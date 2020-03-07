#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

class KNeighbors
{
private:
	int K;
	MatrixXd features;
	VectorXd labels;
public:
	KNeighbors(int n_neighbors);
	~KNeighbors() {}
	void fit(const MatrixXd& X, const VectorXd& Y);
	VectorXd predict(const MatrixXd& X);
};

namespace kn
{
	int count_classes(const VectorXd& Y);

	vector<vector<double>> calc_norms(const RowVectorXd& point, const MatrixXd& features,
		const VectorXd& labels);

	double euclidean_norm(const RowVectorXd& p1, const RowVectorXd& p2);

	vector<int> select_K_neighbors(vector<vector<double>>& norms, int K, int n_class);

	void sort_by_norm(vector<vector<double>>& norms);

	double select_most_feaquent(const vector<int>& neighbors);
}

KNeighbors::KNeighbors(int n_neighbors) : K(0)
{
	if (n_neighbors <= 0)
		cout << "Error(KNeighbors::KNeighbors(int)): Invalid argument." << endl;
	else
		K = n_neighbors;
}

void KNeighbors::fit(const MatrixXd& X, const VectorXd& Y)
{
	features = X;
	labels = Y;
}

VectorXd KNeighbors::predict(const MatrixXd& X)
{
	using namespace kn;

	int n_classes = count_classes(labels);

	VectorXd predicts(X.rows());
	for (int i = 0; i < X.rows(); i++)
	{
		vector<vector<double>> norms = calc_norms(X.row(i), features, labels);
		vector<int> neighbors = select_K_neighbors(norms, K, n_classes);
		predicts[i] = select_most_feaquent(neighbors);
	}
	return predicts;
}

int kn::count_classes(const VectorXd& Y)
{
	return *std::max_element(Y.data(), Y.data() + Y.size()) + 1;
}

vector<vector<double>> kn::calc_norms(const RowVectorXd& point, const MatrixXd& features,
	const VectorXd& labels)
{
	vector<vector<double>> norms(features.rows(), vector<double>());
	for (int i = 0; i < features.rows(); i++)
	{
		norms[i].push_back(euclidean_norm(point, features.row(i)));
		norms[i].push_back(labels[i]);
	}
	return norms;
}

double kn::euclidean_norm(const RowVectorXd& p1, const RowVectorXd& p2)
{
	if (p1.size() != p2.size())
	{
		cout << "Error(KNeighbors::euclidean_norm(cosnt RowVectorXd&, const RowVectorXd&)): " <<
			"Vectors are not compatible." << endl;
		return 0.0;
	}

	double sum = 0;
	for (int i = 0; i < p1.size(); i++)
		sum += std::pow((p1[i] - p2[i]), 2);

	return std::sqrt(sum);
}

vector<int> kn::select_K_neighbors(vector<vector<double>>& norms, int K, int n_classes)
{
	sort_by_norm(norms);

	vector<int> neighbors(n_classes, 0);
	for (int i = 0; i < K; i++)
	{
		int label = (int)norms[i][1];
		neighbors[label]++;
	}
	return neighbors;
}

void kn::sort_by_norm(vector<vector<double>>& norms)
{
	std::sort(norms.begin(), norms.end(), [](const auto& a, const auto& b) { return a[0] < b[0]; });
}

double kn::select_most_feaquent(const vector<int>& neighbors)
{
	return (double)std::distance(neighbors.begin(), std::max_element(neighbors.begin(), neighbors.end()));
}