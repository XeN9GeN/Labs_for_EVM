#include "Matrix.h"
#include <random>
#include <cmath>

Matrix_divined::Matrix_divined(int N, int rank, int pros_count) {
	m_full_size = N;

	int base_rows = N / pros_count;//колво строк
	int rest = N % pros_count;//остаток

	if (rank < rest) {//проверка, что процесс получит на 1 строку больше
		process_rows_count = base_rows + 1;

		//индекс начальной строки в блоке равно сумме строк у всех предыдущих таких увеличенных процессов
		process_start_row = rank * (base_rows + 1);
	}
	else {//если не должен получить доп строку
		process_rows_count = base_rows;

		//начало блока = строки из процессов доп строкой(base+1) +
		// строки у предыдущих процессов без доп строки, которые перед нами(их кол-во = rank - rest, по base строк каждая)
		process_start_row = rest * (base_rows + 1) + (rank - rest) * base_rows;
	}


	grid.resize(process_rows_count * m_full_size);//свой блок строк, но общие столбцы
}

int Matrix_divined::index(int proc_row, int col) const {
	return proc_row * m_full_size + col;
}

void Matrix_divined::m_fill() {
	int N = m_full_size;
	std::fill(grid.begin(), grid.end(), 1.0);

	for (int i = 0; i < process_rows_count; i++) {
		int j = process_start_row + i;//глобальный номер строки
		grid[index(i, j)] = 2.0;
	}
}


//Каждая строка матрицы A умножается на вектор x 
// и даёт один элемент результата Ax

//Тогда строки Ax и x должны быть одинаковы для каждого процесса
//Поэтому разделение должно быть одинаковым
Vector_divined::Vector_divined(int N, int rank, int pros_count) {
	v_full_size = N;
	int base_elem = N / pros_count;
	int rest = N % pros_count;

	if (rank < rest) {
		process_elem_count = base_elem + 1;
		process_start_elem = rank * (base_elem + 1);
	}
	else {
		process_elem_count = base_elem;
		process_start_elem = rest * (base_elem + 1) + (rank - rest) * base_elem;
	}
	data.resize(process_elem_count);
}

Vector_full::Vector_full(int N) {
	v_full_size = N;
	data.resize(N);

}