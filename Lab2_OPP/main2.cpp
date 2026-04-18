#include <iostream>
#include <vector>
#include <cmath>
#include "Matrix.h"
#include <omp.h>
#include <chrono>
#include <thread>
#include <math.h>

const double EPS = 1e-5;
const double tau = 0.0005;
int maxx = 100000;
int N_NOW = 100;


Vector_full operator-(const Vector_full& a, const Vector_full& b) {
	int N = a.v_full_size;
	Vector_full result(N);

	for (int i = 0; i < N; i++) {
		result.data[i] = a.data[i] - b.data[i];
	}

	return result;
}
Vector_divined operator-(const Vector_divined& a, const Vector_divined& b) {
	int N = a.v_full_size;
	Vector_divined result(N, 0, 1);

	for (int i = 0; i < N; i++) {
		result.data[i] = a.data[i] - b.data[i];
	}

	return result;
}
double norm(const Vector_full& v) {
	double s = 0;

	#pragma omp parallel for reduction(+:s) schedule(static)
	for (int i = 0; i < v.v_full_size; i++) {
		s += v.data[i] * v.data[i];
	}
	return sqrt(s);
}
double norm(const Vector_divined& v) {
	double s = 0;

	#pragma omp parallel for reduction(+:s) schedule(static)
	for (int i = 0; i < v.process_elem_count; i++) {
		s += v.data[i] * v.data[i];
	}
	return sqrt(s);
}


void multiply_div_mat_on_full_vec(const Matrix_divined& A, const Vector_full& x, Vector_full& res) {
	int N = A.m_full_size;
	int M = A.process_rows_count;

	res.v_full_size = N;
	
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < M; i++) {//локальные строки

		double sum = 0.0;
		for (int j = 0; j < N; j++) {//столбец
			sum += A.grid[A.index(i, j)] * x.data[j];
		}
		res.data[i] = sum;
	}
}
void solve_Var1(int N, int rank, int size, Matrix_divined& A, Vector_full& x, Vector_full& b) {
	bool conver = false;
	int it = 0;

	Vector_full Ax_full_vector(N);
	Vector_full x_next(N);

	double b_norm = norm(b);
	double start_time = omp_get_wtime();

	while (!conver && it < maxx) {
		it++;

		multiply_div_mat_on_full_vec(A, x, Ax_full_vector);
		double r2 = 0.0;

		#pragma omp parallel for reduction(+:r2) schedule(static)
		for (int i = 0; i < N; i++) {
			double r = (Ax_full_vector.data[i] - b.data[i]);
			r2 += r * r;
			x_next.data[i] = x.data[i] - tau * r;
		}
		conver = (sqrt(r2) / b_norm < EPS);
		x = x_next;
	}

	double end_time = omp_get_wtime();

	std::cout << "------------------------------------" << std::endl;
	std::cout << "Количество итераций: " << it << std::endl;
	if (conver) {
		std::cout << "Метод сошёлся по точности EPS." << std::endl;
	}
	else {
		std::cout << "Достигнут лимит итераций (maxx)." << std::endl;
	}
	std::cout << "Elapsed time (Var1): " << end_time - start_time << " seconds" << std::endl;
	std::cout << "------------------------------------" << std::endl;
}


void solve_Var2(int N, int rank, int size, Matrix_divined& A, Vector_divined& x,Vector_divined& b){
	Vector_divined Ax(N, rank, size);
	Vector_divined x_next(N, rank, size);


	double b2 = 0.0;
	#pragma omp parallel for reduction(+:b2) schedule(static)
	for (int i = 0; i < N; i++) {
		b2 += b.data[i] * b.data[i];	
	}
	double b_norm = sqrt(b2);


	double r2 = 0.0;
	int it = 0;
	bool conver = false;

	double start_time = omp_get_wtime();

	#pragma omp parallel shared(A, x, b, Ax, x_next, conver, it, b_norm)
{
		while (true) {
			#pragma omp single
			{
				r2 = 0.0;
				if (conver || it >= maxx) {
					conver = true;
				}
			}

			#pragma omp barrier//иначе другие потоки могут успеть прочитать старое значение conver
			if (conver) break;

			#pragma omp for schedule(static)//без reduction, ибо локальная на поток переменная
			for (int i = 0; i < A.process_rows_count; i++) {
				double sum = 0.0;
				for (int j = 0; j < N; j++) {
					sum += A.grid[A.index(i, j)] * x.data[j];
				}
				Ax.data[i] = sum;
			}


			#pragma omp for reduction(+:r2) schedule(static)//ибо объявлена вне потоко
			for (int i = 0; i < N; i++) {
				double r = Ax.data[i] - b.data[i];
				r2 += r * r;
				x_next.data[i] = x.data[i] - tau * r;
			}

			#pragma omp single
			{
				conver = (sqrt(r2) / b_norm < EPS);
				x = x_next;
				it++;
			}

			#pragma omp barrier//также из-за сингла сверху
		}
	}

	double end_time = omp_get_wtime();

	std::cout << "------------------------------------\n";
	std::cout << "Итого итераций: " << it << "\n";
	if (it < maxx)
		std::cout << "Метод сошёлся по EPS\n";
	else
		std::cout << "Достигнут лимит итераций\n";

	std::cout << "Elapsed time (Var2): "
		<< end_time - start_time << " s\n";
	std::cout << "------------------------------------\n";
}

int main(int argc, char** argv) {
	int variant = 1;
	if (argc > 1) {
		variant = atoi(argv[1]);
	}

	int N = N_NOW;
	if (argc > 2) {
		N = atoi(argv[2]);
	}

	int threads = 1;
	if (argc > 3) {
		threads = atoi(argv[3]);
	}

	if (threads > 0) {
		omp_set_num_threads(threads);
	}

	int rank = 0;
	int size = 1;

	Matrix_divined A(N, rank, size);
	A.m_fill();

	for (int i = 0; i < 3; i++) {
		int global_row = i;
		double diag = A.grid[A.index(i, global_row)];
		double first = A.grid[A.index(i, 0)];

		std::cout << "Строка " << global_row
			<< ": первый=" << first
			<< ", диагональ=" << diag << "\n";
	}

	if (variant == 1) {
		Vector_full x(N), b(N);

		#pragma omp parallel for schedule(static)
		for (int i = 0; i < N; i++) {
			x.data[i] = 0.0;
			b.data[i] = N + 1.0;
		}

		solve_Var1(N, rank, size, A, x, b);

		for (int i = 0; i < 30; i++) {
			std::cout << "x[" << i << "] = " << x.data[i] << std::endl;
		}
	}
	else if (variant == 2) {
		Vector_divined x_block(N, rank, size);
		Vector_divined b_block(N, rank, size);

		#pragma omp parallel for schedule(static)
		for (int i = 0; i < N; i++) {
			x_block.data[i] = 0.0;
			b_block.data[i] = N + 1.0;
		}

		solve_Var2(N, rank, size, A, x_block, b_block);

		for (int i = 0; i < 30; i++) {
			std::cout << "x[" << i << "] = " << x_block.data[i] << std::endl;
		}
	}

	return 0;
}
