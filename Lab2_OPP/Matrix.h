#pragma once
#include <vector>

class Matrix_divined {
public:
	int m_full_size;
	int process_rows_count;
	int process_start_row;

	std::vector<double> grid;
	int index(int row, int column) const;

	Matrix_divined(int N, int rank, int pros_count);// n- размер матрицы, rank - номер процесса, size - кол-во процессов
	void m_fill();

};

class Vector_divined {
public:
	int v_full_size;
	int process_elem_count;
	int process_start_elem;

	std::vector<double> data;

	Vector_divined(int N, int rank, int pros_count);//тоже самое что выше
};

class Vector_full {
public:
	int v_full_size;
	std::vector<double> data;

	Vector_full(int N);
};